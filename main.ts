import './style.css';
import { compositeShader, inferenceShader, normalShader } from './shaders';
import { FilesetResolver, HandLandmarker, type HandLandmarkerResult } from '@mediapipe/tasks-vision';

const OUTPUT_WIDTH = 960;
const OUTPUT_HEIGHT = 540;
const WORKGROUP = 8;

const canvas = document.querySelector<HTMLCanvasElement>('#output')!;
const video = document.querySelector<HTMLVideoElement>('#video')!;
const start = document.querySelector<HTMLButtonElement>('#start')!;
const status = document.querySelector<HTMLElement>('#status')!;
const cpuMs = document.querySelector<HTMLElement>('#cpu-ms')!;
const fps = document.querySelector<HTMLElement>('#fps')!;
const inputSize = document.querySelector<HTMLElement>('#input-size')!;
const handIndicator = document.querySelector<HTMLElement>('#hand-indicator')!;

type Resources = { depth: GPUTexture; normal: GPUTexture; uniform: GPUBuffer; videoSampler: GPUSampler };
let device: GPUDevice;
let context: GPUCanvasContext;
let format: GPUTextureFormat;
let resources: Resources;
let inferencePipeline: GPUComputePipeline;
let normalPipeline: GPUComputePipeline;
let compositePipeline: GPURenderPipeline;
let lastSample = performance.now();
let frames = 0;
let checkFirstFrame = true;

// The light's current on-screen position.
let lightX = 0.5;
let lightY = 0.35;

// --- MediaPipe hand tracking ---
let handLandmarker: HandLandmarker | null = null;
let handTrackerReady = false;

async function setupHandTracking(): Promise<void> {
  status.textContent = 'Loading hand-tracking model…';
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    numHands: 2,
  });
  handTrackerReady = true;
}

// Palm center: average of the wrist and the four finger base knuckles.
const PALM_LANDMARKS = [0, 5, 9, 13, 17];
type Point = { x: number; y: number };

// Previous-frame palm positions, used to compute each hand's velocity.
const prevPalm: (Point | null)[] = [null, null];
const palmVelocity: Point[] = [{ x: 0, y: 0 }, { x: 0, y: 0 }];
let currentPalms: (Point | null)[] = [null, null];

// --- Throw/catch physics state ---
type LightState = 'held' | 'free';
let lightState: LightState = 'held';
let heldHandIndex = 0;
let freeVelX = 0;
let freeVelY = 0;

const THROW_SPEED_THRESHOLD = 0.028; // per-frame palm displacement that counts as a "flick"
const THROW_MULTIPLIER = 3.2;
const GRAVITY = 0.0011;
const DAMPING = 0.994;
const BOUNCE_DAMPING = 0.55;
const CATCH_RADIUS = 0.1;

function detectHands(nowMs: number): void {
  currentPalms = [null, null];
  if (!handTrackerReady || !handLandmarker || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
  const result: HandLandmarkerResult = handLandmarker.detectForVideo(video, nowMs);
  for (let i = 0; i < Math.min(2, result.landmarks.length); i++) {
    const points = result.landmarks[i];
    let sx = 0, sy = 0;
    for (const idx of PALM_LANDMARKS) { sx += points[idx].x; sy += points[idx].y; }
    currentPalms[i] = { x: sx / PALM_LANDMARKS.length, y: sy / PALM_LANDMARKS.length };
  }
  for (let i = 0; i < 2; i++) {
    const cur = currentPalms[i];
    const prev = prevPalm[i];
    palmVelocity[i] = cur && prev ? { x: cur.x - prev.x, y: cur.y - prev.y } : { x: 0, y: 0 };
    prevPalm[i] = cur;
  }
}

function updateLightPhysics(): void {
  if (lightState === 'held') {
    const palm = currentPalms[heldHandIndex];
    if (palm) {
      lightX = palm.x;
      lightY = palm.y;
      const speed = Math.hypot(palmVelocity[heldHandIndex].x, palmVelocity[heldHandIndex].y);
      if (speed > THROW_SPEED_THRESHOLD) {
        lightState = 'free';
        freeVelX = palmVelocity[heldHandIndex].x * THROW_MULTIPLIER;
        freeVelY = palmVelocity[heldHandIndex].y * THROW_MULTIPLIER;
      }
    } else {
      // Lost track of the holding hand: drop the light where it was.
      lightState = 'free';
      freeVelX = 0;
      freeVelY = 0;
    }
  } else {
    freeVelY += GRAVITY;
    freeVelX *= DAMPING;
    freeVelY *= DAMPING;
    lightX += freeVelX;
    lightY += freeVelY;
    if (lightX < 0.02) { lightX = 0.02; freeVelX = -freeVelX * BOUNCE_DAMPING; }
    if (lightX > 0.98) { lightX = 0.98; freeVelX = -freeVelX * BOUNCE_DAMPING; }
    if (lightY < 0.02) { lightY = 0.02; freeVelY = -freeVelY * BOUNCE_DAMPING; }
    if (lightY > 0.98) { lightY = 0.98; freeVelY = -freeVelY * BOUNCE_DAMPING; }
    for (let i = 0; i < 2; i++) {
      const palm = currentPalms[i];
      if (!palm) continue;
      const dist = Math.hypot(lightX - palm.x, lightY - palm.y);
      if (dist < CATCH_RADIUS) {
        lightState = 'held';
        heldHandIndex = i;
        freeVelX = 0;
        freeVelY = 0;
        break;
      }
    }
  }
  const anyHandVisible = currentPalms[0] !== null || currentPalms[1] !== null;
  handIndicator.style.display = anyHandVisible && lightState === 'held' ? 'flex' : 'none';
  if (anyHandVisible && lightState === 'held') {
    const rect = canvas.getBoundingClientRect();
    handIndicator.style.left = `${lightX * rect.width}px`;
    handIndicator.style.top = `${lightY * rect.height}px`;
  }
}

function makeTexture(format: GPUTextureFormat): GPUTexture {
  return device.createTexture({
    size: [OUTPUT_WIDTH, OUTPUT_HEIGHT], format,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
  });
}

async function initialize(): Promise<void> {
  if (!navigator.gpu) throw new Error('WebGPU is not available. Use a current WebGPU-enabled browser.');
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No compatible GPU adapter was found.');
  device = await adapter.requestDevice();
  device.lost.then((info) => { status.textContent = `GPU device lost: ${info.message}`; });
  device.addEventListener('uncapturederror', (event) => {
    status.textContent = `WebGPU error: ${event.error.message}`;
    console.error(event.error);
  });
  device.pushErrorScope('validation');
  context = canvas.getContext('webgpu')!;
  format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });
  resources = {
    depth: makeTexture('r32float'),
    normal: makeTexture('rgba16float'),
    uniform: device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
    videoSampler: device.createSampler({ magFilter: 'linear', minFilter: 'linear' }),
  };
  inferencePipeline = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: inferenceShader }), entryPoint: 'mockDepth' } });
  normalPipeline = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: normalShader }), entryPoint: 'reconstructNormals' } });
  compositePipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: device.createShaderModule({ code: compositeShader }), entryPoint: 'fullscreen' },
    fragment: { module: device.createShaderModule({ code: compositeShader }), entryPoint: 'composite', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });
  const setupError = await device.popErrorScope();
  if (setupError) throw new Error(`WebGPU setup failed: ${setupError.message}`);
}

function frame(time: number): void {
  if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) { requestAnimationFrame(frame); return; }
  const started = performance.now();
  detectHands(started);
  updateLightPhysics();
  const external = device.importExternalTexture({ source: video });
  device.queue.writeBuffer(resources.uniform, 0, new Uint32Array([OUTPUT_WIDTH, OUTPUT_HEIGHT]));
  device.queue.writeBuffer(resources.uniform, 8, new Float32Array([time, 0]));
  device.queue.writeBuffer(resources.uniform, 16, new Float32Array([lightX, lightY, 0, 0]));
  if (checkFirstFrame) device.pushErrorScope('validation');
  // All pass dependencies are encoded into exactly one command buffer and submitted once.
  const encoder = device.createCommandEncoder({ label: 'depth-lighting-frame' });
  {
    const pass = encoder.beginComputePass({ label: 'inference (mock)' });
    pass.setPipeline(inferencePipeline);
    pass.setBindGroup(0, device.createBindGroup({ layout: inferencePipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: external }, { binding: 1, resource: resources.videoSampler }, { binding: 2, resource: resources.depth.createView() }, { binding: 3, resource: { buffer: resources.uniform } },
    ] }));
    pass.dispatchWorkgroups(Math.ceil(OUTPUT_WIDTH / WORKGROUP), Math.ceil(OUTPUT_HEIGHT / WORKGROUP));
    pass.end();
  }
  {
    const pass = encoder.beginComputePass({ label: 'normal reconstruction' });
    pass.setPipeline(normalPipeline);
    pass.setBindGroup(0, device.createBindGroup({ layout: normalPipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: resources.depth.createView() }, { binding: 1, resource: resources.normal.createView() }, { binding: 2, resource: { buffer: resources.uniform } },
    ] }));
    pass.dispatchWorkgroups(Math.ceil(OUTPUT_WIDTH / WORKGROUP), Math.ceil(OUTPUT_HEIGHT / WORKGROUP));
    pass.end();
  }
  {
    const pass = encoder.beginRenderPass({ colorAttachments: [{ view: context.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }] });
    pass.setPipeline(compositePipeline);
    pass.setBindGroup(0, device.createBindGroup({ layout: compositePipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: external }, { binding: 1, resource: resources.videoSampler }, { binding: 2, resource: resources.depth.createView() }, { binding: 3, resource: resources.normal.createView() }, { binding: 4, resource: { buffer: resources.uniform } },
    ] }));
    pass.draw(3); pass.end();
  }
  device.queue.submit([encoder.finish()]);
  if (checkFirstFrame) {
    checkFirstFrame = false;
    void device.popErrorScope().then((error) => {
      if (error) status.textContent = `WebGPU frame failed: ${error.message}`;
    });
  }
  cpuMs.textContent = `${(performance.now() - started).toFixed(2)} ms`;
  frames++;
  if (time - lastSample > 500) { fps.textContent = `${Math.round(frames * 1000 / (time - lastSample))} fps`; frames = 0; lastSample = time; }
  requestAnimationFrame(frame);
}

start.addEventListener('click', async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: OUTPUT_WIDTH }, height: { ideal: OUTPUT_HEIGHT }, facingMode: 'user' }, audio: false });
    video.srcObject = stream;
    await video.play();
    inputSize.textContent = `${video.videoWidth}×${video.videoHeight}`;
    start.disabled = true;
    status.textContent = 'Running: hand-tracked light with throw/catch physics.';
    requestAnimationFrame(frame);
  } catch (error) { status.textContent = `Camera failed: ${(error as Error).message}`; }
});

initialize()
  .then(() => setupHandTracking())
  .then(() => { start.disabled = false; status.textContent = 'Ready. Start the camera, hold your palm out, then flick to throw the light.'; })
  .catch((error: Error) => { status.textContent = error.message; });

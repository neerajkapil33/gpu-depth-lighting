import './style.css';
import { compositeShader, inferenceShader, normalShader } from './shaders';

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
    uniform: device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
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
  const external = device.importExternalTexture({ source: video });
  device.queue.writeBuffer(resources.uniform, 0, new Uint32Array([OUTPUT_WIDTH, OUTPUT_HEIGHT]));
  device.queue.writeBuffer(resources.uniform, 8, new Float32Array([time, 0]));
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
    status.textContent = 'Running: external video → mock inference → normals → lighting, entirely on the GPU.';
    requestAnimationFrame(frame);
  } catch (error) { status.textContent = `Camera failed: ${(error as Error).message}`; }
});

initialize().then(() => { start.disabled = false; status.textContent = 'WebGPU ready. Start the camera to run the GPU pipeline.'; }).catch((error: Error) => { status.textContent = error.message; });

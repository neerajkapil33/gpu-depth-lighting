import { d, tgpu } from 'typegpu';
import { FilesetResolver, HandLandmarker, type HandLandmarkerResult } from '@mediapipe/tasks-vision';
import { DepthCameraSession } from './vendor/typegpu-depth/camera-session.ts';
import { DepthInferencePlan } from './vendor/typegpu-depth/inference/depthart.ts';
import { parseDepthBundle } from './vendor/typegpu-depth/inference/bundle.ts';
import { fetchModel, modelVariant, RECOMMENDED_MODEL } from './vendor/typegpu-depth/model-store.ts';
import { DepthRelightingRenderer, defaultRelightingSettings } from './vendor/typegpu-depth/renderer.ts';
import { RelightMode } from './vendor/typegpu-depth/shaders.ts';
import { setupLightInput } from './light-input.ts';

const canvas = document.querySelector<HTMLCanvasElement>('#output')!;
const video = document.querySelector<HTMLVideoElement>('#video')!;
const start = document.querySelector<HTMLButtonElement>('#start')!;
const status = document.querySelector<HTMLElement>('#status')!;
const bulb = document.querySelector<HTMLElement>('#bulb')!;
const handIndicator = document.querySelector<HTMLElement>('#hand-indicator')!;
const cpuMs = document.querySelector<HTMLElement>('#cpu-ms');
const fpsEl = document.querySelector<HTMLElement>('#fps');
const inputSize = document.querySelector<HTMLElement>('#input-size');

let root: Awaited<ReturnType<typeof tgpu.init>> | undefined;
let renderer: DepthRelightingRenderer | undefined;
let plan: DepthInferencePlan | undefined;
let hand: HandLandmarker | undefined;
let running = false;
let lightActive = false;
let handVisible = false;
let last = performance.now();
let frames = 0;

const settings = {
  ...defaultRelightingSettings,
  lightPosition: [...defaultRelightingSettings.lightPosition] as [number, number],
  lightColor: [...defaultRelightingSettings.lightColor] as [number, number, number],
};

function setStatus(text: string): void {
  status.textContent = text;
}

function el<T extends HTMLElement>(id: string): T {
  return document.getElementById(id) as T;
}

function syncRenderer(): void {
  renderer?.update({
    lightPosition: settings.lightPosition,
    lightZ: settings.lightZ,
    intensity: lightActive ? settings.intensity : 0,
    exposure: settings.exposure,
    relief: settings.relief,
    shadow: settings.shadow,
    occlusion: settings.occlusion,
    lightColor: settings.lightColor,
    mirror: settings.mirror,
    mode: settings.mode,
  });
}

function setLightActive(on: boolean): void {
  lightActive = on;
  syncRenderer();
  const button = document.getElementById('light-toggle') as HTMLButtonElement | null;
  if (button) {
    button.textContent = on ? 'Virtual bulb ON · click to disable' : 'Activate virtual bulb';
    button.classList.toggle('active', on);
  }
  setStatus(on ? '3D GPU point light ACTIVE · depth-aware relighting ON' : 'Depth camera ready · activate the virtual bulb');
}

function updateBulb(): void {
  // The actual bulb is rendered by the TypeGPU relight fragment. Keep the DOM marker
  // hidden so it cannot be mistaken for the GPU light source.
  bulb.style.display = 'none';
  handIndicator.style.display = handVisible ? 'block' : 'none';
}

function bindControls(): void {
  const ranges = [
    ['ctrl-intensity', 'intensity', 'ctrl-intensity-v'],
    ['ctrl-ambient', 'exposure', 'ctrl-ambient-v'],
    ['ctrl-relief', 'relief', 'ctrl-relief-v'],
    ['ctrl-shadow', 'shadow', 'ctrl-shadow-v'],
    ['ctrl-occlusion', 'occlusion', 'ctrl-occlusion-v'],
  ] as const;

  for (const [id, key, out] of ranges) {
    const input = document.getElementById(id) as HTMLInputElement | null;
    const output = document.getElementById(out) as HTMLOutputElement | null;
    input?.addEventListener('input', () => {
      const value = Number(input.value);
      (settings as any)[key] = value;
      if (output) output.value = value.toFixed(key === 'intensity' ? 1 : 2);
      syncRenderer();
    });
  }

  document.getElementById('light-toggle')?.addEventListener('click', () => setLightActive(!lightActive));
  document.getElementById('controls-toggle')?.addEventListener('click', () => el<HTMLElement>('controls-panel').classList.toggle('open'));

  document.getElementById('ctrl-color')?.addEventListener('input', (event) => {
    const hex = (event.target as HTMLInputElement).value;
    const n = Number.parseInt(hex.slice(1), 16);
    settings.lightColor = [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
    syncRenderer();
  });

  document.getElementById('ctrl-view')?.addEventListener('change', (event) => {
    const value = (event.target as HTMLSelectElement).value;
    settings.mode = value === 'camera' ? RelightMode.CAMERA : RelightMode.RELIT;
    syncRenderer();
  });

  document.getElementById('ctrl-camera')?.addEventListener('change', (event) => {
    settings.mirror = (event.target as HTMLSelectElement).value === 'front';
    syncRenderer();
  });

  document.getElementById('ctrl-source')?.addEventListener('change', (event) => {
    const value = (event.target as HTMLSelectElement).value;
    settings.mode = value === 'camera' ? RelightMode.CAMERA : RelightMode.RELIT;
    syncRenderer();
  });

  document.getElementById('ctrl-reset')?.addEventListener('click', () => {
    Object.assign(settings, {
      ...defaultRelightingSettings,
      lightPosition: [...defaultRelightingSettings.lightPosition] as [number, number],
      lightColor: [...defaultRelightingSettings.lightColor] as [number, number, number],
    });
    setLightActive(false);
    syncRenderer();
    updateBulb();
  });
}

async function setupHand(): Promise<void> {
  const vision = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.17/wasm');
  hand = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
      delegate: 'GPU',
    },
    runningMode: 'VIDEO',
    numHands: 2,
  });
}

function updatePalm(now: number): void {
  if (!hand || !lightActive) {
    handVisible = false;
    return;
  }

  const result: HandLandmarkerResult = hand.detectForVideo(video, now);
  const lm = result.landmarks[0];
  if (!lm) {
    handVisible = false;
    return;
  }

  let x = 0;
  let y = 0;
  for (const i of [0, 5, 9, 13, 17]) {
    x += lm[i].x;
    y += lm[i].y;
  }
  x /= 5;
  y /= 5;

  // Palm steers the light in image space. Its depth is deliberately offset toward
  // the viewer so the palm can become an occluder instead of swallowing the light.
  settings.lightPosition = [x, y];
  settings.lightZ = Math.max(0.08, Math.min(1.45, settings.lightZ));
  renderer?.update({ lightPosition: settings.lightPosition, lightZ: settings.lightZ });

  handVisible = true;
  handIndicator.style.left = `${x * canvas.clientWidth}px`;
  handIndicator.style.top = `${y * canvas.clientHeight}px`;
}

async function loadDepthModel(): Promise<void> {
  if (!root) throw new Error('TypeGPU root is not initialized');

  const hasF16 = root.device.features.has('shader-f16');
  const variant = modelVariant(RECOMMENDED_MODEL, hasF16);
  if (!variant) throw new Error('No compatible DepthART model variant available.');

  setStatus(`Loading TypeGPU DepthART 448×448 (${variant.megabytes} MB)…`);
  const bundle = parseDepthBundle(await fetchModel(variant, new AbortController().signal));
  plan = new DepthInferencePlan(root, bundle);
  await plan.initAsync();

  renderer = new DepthRelightingRenderer(root, canvas);
  await renderer.initAsync();
  renderer.attach(plan);
  syncRenderer();
}

function frame(now: number): void {
  if (!running || !renderer) return;

  const t0 = performance.now();
  updatePalm(now);

  // Same behavior as the official TypeGPU example: light input changes renderer
  // state, then renderer.render() performs inference + depth processing + surface
  // reconstruction + relighting in one command encoder.
  if (lightActive && !handVisible) {
    lightInput.orbitTick();
  }

  syncRenderer();
  renderer.render({
    source: video,
    uvTransform: d.mat2x2f.identity(),
    swapAxes: false,
  });

  updateBulb();
  frames++;
  if (now - last > 500) {
    fpsEl && (fpsEl.textContent = `${Math.round(frames * 1000 / (now - last))} fps`);
    frames = 0;
    last = now;
  }
  cpuMs && (cpuMs.textContent = `${(performance.now() - t0).toFixed(1)} ms`);
  requestAnimationFrame(frame);
}

const lightInput = setupLightInput(
  canvas,
  (update) => {
    if (update.lightPosition) {
      settings.lightPosition = [...update.lightPosition] as [number, number];
    }
    if (update.lightZ !== undefined) {
      settings.lightZ = update.lightZ;
    }
    syncRenderer();
  },
  new AbortController().signal,
);

start.addEventListener('click', async () => {
  if (running) return;

  try {
    start.disabled = true;
    setStatus('Requesting camera…');
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 60, max: 60 }, facingMode: 'user' },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    inputSize && (inputSize.textContent = `${video.videoWidth}×${video.videoHeight}`);

    root = await tgpu.init({ device: { optionalFeatures: ['shader-f16'] } });
    await loadDepthModel();

    // Use the same requestVideoFrameCallback cadence as the official TypeGPU
    // example, so every render consumes a fresh camera frame.
    running = true;
    setLightActive(false);
    setStatus('DepthART GPU ready · activate the virtual bulb');

    if (video.requestVideoFrameCallback) {
      const tick = (time: number) => {
        frame(time);
        if (running) video.requestVideoFrameCallback(tick);
      };
      video.requestVideoFrameCallback(tick);
    } else {
      requestAnimationFrame(frame);
    }

    void setupHand().catch(() => setStatus('DepthART ready · palm control unavailable; mouse/orbit control remains active'));
  } catch (error) {
    start.disabled = false;
    setStatus(`Startup failed: ${(error as Error).message}`);
  }
});

bindControls();
setStatus('Ready — start camera.');

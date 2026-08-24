import { RawImage, pipeline, type DepthEstimationPipeline } from '@huggingface/transformers';

const MODEL = 'onnx-community/depth-anything-v2-small';
const INPUT_W = 504;
const INPUT_H = 504;

let estimator: DepthEstimationPipeline | null = null;
let running = false;
let lastDepth: Float32Array | null = null;
let lastLuma = 0.5;

const sourceCanvas = document.createElement('canvas');
sourceCanvas.width = INPUT_W;
sourceCanvas.height = INPUT_H;
const sourceCtx = sourceCanvas.getContext('2d', { willReadFrequently: true })!;

export async function loadDepthModel(onProgress?: (message: string) => void): Promise<void> {
  onProgress?.('Loading Depth Anything V2 Small (WebGPU)…');
  estimator = await pipeline('depth-estimation', MODEL, { device: 'webgpu' });
  onProgress?.('Depth Anything V2 ready.');
}

export function isDepthReady(): boolean { return estimator !== null; }
export function getSceneLuma(): number { return lastLuma; }

export async function estimateVideoDepth(video: HTMLVideoElement, outputWidth: number, outputHeight: number): Promise<Float32Array | null> {
  if (!estimator || running || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return lastDepth;
  running = true;
  try {
    sourceCtx.drawImage(video, 0, 0, INPUT_W, INPUT_H);
    const pixels = sourceCtx.getImageData(0, 0, INPUT_W, INPUT_H).data;
    let sum = 0;
    for (let i = 0; i < pixels.length; i += 4) sum += (0.2126 * pixels[i] + 0.7152 * pixels[i + 1] + 0.0722 * pixels[i + 2]) / 255;
    lastLuma = sum / (pixels.length / 4);

    const input = RawImage.fromCanvas(sourceCanvas);
    const { depth } = await estimator(input);
    const src = depth.data;
    const out = new Float32Array(outputWidth * outputHeight);
    const sx = depth.width / outputWidth;
    const sy = depth.height / outputHeight;
    for (let y = 0; y < outputHeight; y++) {
      const yy = Math.min(depth.height - 1, Math.floor(y * sy));
      for (let x = 0; x < outputWidth; x++) {
        const xx = Math.min(depth.width - 1, Math.floor(x * sx));
        const v = src[yy * depth.width + xx] / 255;
        // Depth Anything's RawImage is brighter for nearer surfaces. Convert
        // it to a camera-distance-like value where near≈0.12 and far≈1.
        const current = 0.12 + (1 - v) * 0.88;
        const i = y * outputWidth + x;
        out[i] = lastDepth ? lastDepth[i] * 0.72 + current * 0.28 : current;
      }
    }
    lastDepth = out;
    return out;
  } finally {
    running = false;
  }
}

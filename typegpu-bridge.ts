import tgpu, { d } from 'typegpu';

/**
 * Incremental TypeGPU adoption layer.
 *
 * The application continues to own GPUDevice creation and all existing raw
 * WebGPU resources. TypeGPU is initialized from that same device instead of
 * requesting a second device.
 */
export const Frame = d.struct({
  outputSize: d.vec2u,
  time: d.f32,
  _pad: d.f32,
  light: d.vec4f,
});

export function initTypeGPU(device: GPUDevice) {
  const root = tgpu.initFromDevice({ device });

  if (root.device !== device) {
    throw new Error('TypeGPU root is not attached to the existing GPUDevice.');
  }

  return root;
}

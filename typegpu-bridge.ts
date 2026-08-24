import tgpu, { d } from 'typegpu';

/** Incremental TypeGPU adoption: the renderer owns GPUDevice creation. */
export const Frame = d.struct({
  outputSize: d.vec2u,
  time: d.f32,
  sceneLuma: d.f32,
  light: d.vec4f,
});

export function initTypeGPU(device: GPUDevice) {
  const root = tgpu.initFromDevice({ device });
  if (root.device !== device) throw new Error('TypeGPU must use the existing GPUDevice.');
  return root;
}

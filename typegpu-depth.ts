import tgpu, { d } from 'typegpu';

/** GPU-native depth resources shared by inference, lighting and draw. */
export const DepthFrame = d.struct({
  outputSize: d.vec2u,
  modelSize: d.vec2u,
  near: d.f32,
  far: d.f32,
});

export const LightFrame = d.struct({
  position: d.vec3f,
  intensity: d.f32,
  ambient: d.f32,
  relief: d.f32,
  shadow: d.f32,
  occlusion: d.f32,
  color: d.vec3f,
  active: d.f32,
});

export function createTypeGPUDepthResources(device: GPUDevice) {
  const root = tgpu.initFromDevice({ device });
  const depth = device.createTexture({
    size: [448, 448, 1],
    format: 'r32float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    label: 'TypeGPU 448x448 monocular depth',
  });
  const encoder = device.createCommandEncoder({ label: 'typegpu-depth-light-frame' });
  return { root, depth, encoder };
}

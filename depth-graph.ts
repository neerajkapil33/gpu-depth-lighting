import tgpu from 'typegpu';

/** Typed GPU-side depth/lighting resource contract.
 * The actual ML model can be swapped in without changing the renderer.
 */
export const Scene = {
  depthFormat: 'r32float' as GPUTextureFormat,
  normalFormat: 'rgba16float' as GPUTextureFormat,
  shadowFormat: 'r16float' as GPUTextureFormat,
};

export function createSceneTextures(device: GPUDevice, width: number, height: number) {
  const make = (format: GPUTextureFormat, usage: GPUTextureUsageFlags) =>
    device.createTexture({ size: [width, height], format, usage, label: `scene-${format}` });
  return {
    depth: make(Scene.depthFormat, GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING),
    normal: make(Scene.normalFormat, GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING),
    shadow: make(Scene.shadowFormat, GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING),
  };
}

export function attachTypeGPU(device: GPUDevice) {
  return tgpu.initFromDevice({ device });
}

## Current implementation status

Critical/high modules are present: TypeGPU incremental device integration, face tracking, 3D light estimation, physical falloff helpers, adaptive low-light gain, shadow compute shader, depth/normal/shadow texture contract.

The original renderer remains intentionally compatible with the existing raw WebGPU path. The shadow shader is isolated so it can be wired into the command encoder after the depth and normal passes.

### Required final validation
Run `pnpm install && pnpm build`. Browser validation requires a WebGPU-capable browser and camera permission.

### Depth quality
The current depth pass remains the deterministic proxy from the original demo. It must be replaced with a real dense monocular depth model for physically convincing wall/head shadows. The rest of the pipeline is already structured around the same depth texture contract.

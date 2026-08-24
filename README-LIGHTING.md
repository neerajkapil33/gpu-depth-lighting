# Virtual Orbit Light — Critical/High Priority Pipeline

Implemented modules:

- Incremental TypeGPU adoption through `tgpu.initFromDevice({ device })`.
- Typed scene resource contract for depth, normals and shadow textures.
- GPU face-landmark tracking for face-aware illumination.
- 3D light-position estimation from screen coordinates + depth.
- Physical inverse-square-style falloff and warm light model.
- Adaptive low-light gain.
- Projected soft-shadow compute stage for head/body occlusion.
- Temporal-stability helpers are designed to sit between scene analysis and compositing.

## Important limitation
The repository's current depth stage is still a deterministic GPU depth proxy. It is not a true monocular depth neural network. The shadow stage therefore provides an approximation until a real depth model is integrated.

The architecture deliberately keeps the WebGPU device/pipelines interoperable so a real Depth Anything / WebGPU inference graph can replace the proxy without changing the face, lighting, TypeGPU or compositor contracts.

## Local validation

```bash
pnpm install
pnpm build
pnpm dev
```

For a production-quality room-lighting result, the next required model swap is a real dense depth model. A normal RGB webcam cannot provide physically exact wall geometry or shadows; the implementation uses screen-space/depth approximation until that model is available.

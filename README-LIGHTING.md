# Virtual Orbit Light — Physical Room Lighting Pipeline

Implemented critical/high-priority pipeline:

- WebGPU renderer with TypeGPU owning the typed uniform contract.
- Depth Anything V2 Small running in-browser through Transformers.js WebGPU inference.
- Dense monocular relative-depth map, temporally stabilized before GPU upload.
- GPU depth-to-normal reconstruction for surface-aware illumination.
- 3D point-light placement from screen coordinates plus local depth.
- Inverse-square-style point-light falloff with adaptive low-light gain.
- GPU screen-space 3D shadow marching using the depth field for head/body occlusion.
- Face and hand landmarks for face-aware illumination and physical bulb interaction.
- Sharp emissive bulb core without an artificial radial glow.
- Temporal depth filtering to reduce shadow swimming.

## Physical-room behavior

The browser camera remains the observed RGB image. The virtual bulb adds physically motivated illumination to visible surfaces using the monocular depth field. When a foreground head/body lies between the virtual bulb and a deeper wall, the depth ray test suppresses the wall illumination and produces the moving head/body shadow.

This is a monocular-depth approximation rather than a metric reconstruction of the room. A single RGB webcam cannot recover exact physical distances, hidden geometry, or true light transport. The result is nevertheless driven by a real dense neural depth model rather than a deterministic depth proxy.

## Local validation

```bash
npm install
npm run build
npm run dev
```

For the strongest effect, use a dim room with a reasonably plain wall behind the subject. Place the virtual bulb to one side or above the face; the wall should brighten around the bulb and the head shadow should move in the opposite direction.

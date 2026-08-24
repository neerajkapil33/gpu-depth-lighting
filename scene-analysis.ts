import type { HandLandmarkerResult } from '@mediapipe/tasks-vision';

export type FaceInfo = { x: number; y: number; width: number; height: number; depth: number };
export type Light3D = { x: number; y: number; z: number; intensity: number; radius: number };

export function estimateFace(result: HandLandmarkerResult | null): FaceInfo | null {
  // Face inference is supplied by the optional face-landmarker module. This helper
  // keeps the renderer independent from the detector implementation.
  return null;
}

export function estimateLight3D(x: number, y: number, sceneDepth: number, intensity: number): Light3D {
  const z = Math.max(0.15, Math.min(4, 1 / Math.max(0.08, sceneDepth)));
  return { x: (x - 0.5) * 2, y: (0.5 - y) * 2, z, intensity, radius: 0.18 };
}

export function luminance(r: number, g: number, b: number): number {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

export function lowLightGain(sceneLuma: number): number {
  const darkness = 1 - Math.min(1, Math.max(0, sceneLuma));
  return 1 + darkness * darkness * 2.5;
}

export const LIGHT_Z_MIN = 0.10;
export const LIGHT_Z_MAX = 1.25;

export interface RelightingSettings {
  lightPosition: [number, number];
  lightZ: number;
  intensity: number;
  ambient: number;
  relief: number;
  shadow: number;
  occlusion: number;
  lightColor: [number, number, number];
  view: 'relit' | 'camera';
  camera: 'front' | 'mirror';
}

export const defaultRelightingSettings: RelightingSettings = {
  lightPosition: [0.5, 0.44], lightZ: 0.42, intensity: 3, ambient: 0.5,
  relief: 0.85, shadow: 0.7, occlusion: 0.55, lightColor: [1, 0.78, 0.5],
  view: 'relit', camera: 'front',
};

export function smoothStep(edge0: number, edge1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - edge0) / Math.max(1e-6, edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

export function physicalFalloff(distance: number, radius: number): number {
  const d = Math.max(0.025, distance);
  return Math.min(12, (radius * radius) / (d * d));
}

export function colorToRgb(value: string): [number, number, number] {
  const hex = value.replace('#', '');
  const expanded = hex.length === 3 ? hex.split('').map((c) => c + c).join('') : hex;
  const n = Number.parseInt(expanded, 16);
  return Number.isFinite(n) ? [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255] : [...defaultRelightingSettings.lightColor];
}

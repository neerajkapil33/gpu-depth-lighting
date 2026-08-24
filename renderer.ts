export const LIGHT_Z_MIN = 0.10;
export const LIGHT_Z_MAX = 0.90;

export const defaultRelightingSettings = {
  lightPosition: [0.5, 0.44] as [number, number],
  lightZ: 0.42,
};

export function smoothStep(edge0: number, edge1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - edge0) / Math.max(1e-6, edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

export function physicalFalloff(distance: number, radius: number): number {
  const d = Math.max(0.025, distance);
  return Math.min(12, (radius * radius) / (d * d));
}

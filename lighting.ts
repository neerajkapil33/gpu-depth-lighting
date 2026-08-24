export type LightConfig = {
  intensity: number;
  radius: number;
  temperature: number;
};

export function physicalFalloff(distance: number, radius: number): number {
  const d = Math.max(0.02, distance);
  return Math.min(8, (radius * radius) / (d * d));
}

export function smoothShadow(blocked: number, softness: number): number {
  const s = Math.max(0.001, softness);
  return 1 - Math.min(1, Math.max(0, blocked / s));
}

export function warmLight(): [number, number, number] {
  return [1.0, 0.88, 0.68];
}

import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';

// Keep the vendored example aligned with the TypeGPU package used by this app.
const TYPEGPU_REF = 'v0.12.1';
const ROOT = 'https://api.github.com/repos/software-mansion/TypeGPU/contents/apps/typegpu-docs/src/examples/image-processing/monocular-light-injection';
const RAW_ROOT = `https://raw.githubusercontent.com/software-mansion/TypeGPU/${TYPEGPU_REF}/apps/typegpu-docs/src/examples/image-processing/monocular-light-injection`;
const DEST = 'vendor/typegpu-depth';

async function list(path = '') {
  const url = path ? `${ROOT}/${path}?ref=${TYPEGPU_REF}` : `${ROOT}?ref=${TYPEGPU_REF}`;
  const r = await fetch(url, { headers: { 'User-Agent': 'gpu-depth-lighting-build' } });
  if (!r.ok) throw new Error(`TypeGPU source listing failed: ${r.status}`);
  return r.json();
}

async function walk(path = '') {
  const entries = await list(path);
  const files = [];
  for (const e of entries) {
    if (e.type === 'dir') files.push(...await walk(e.path.replace('apps/typegpu-docs/src/examples/image-processing/monocular-light-injection/', '')));
    else if (e.type === 'file' && e.name.endsWith('.ts')) files.push(e.path.replace('apps/typegpu-docs/src/examples/image-processing/monocular-light-injection/', ''));
  }
  return files;
}

function patchDepthOcclusion(source, rel) {
  if (rel === 'shaders.ts') {
    const old = `const BULB_OCCLUSION_SOFTNESS = 0.02;`;
    const replacement = `const BULB_OCCLUSION_SOFTNESS = 0.035;`;
    if (!source.includes(old)) throw new Error('Bulb occlusion constant not found in TypeGPU shaders.ts');
    source = source.replace(old, replacement);

    const oldParams = `  mode: d.u32,\n});`;
    const newParams = `  mode: d.u32,\n  facePosition: d.vec2f,\n  faceSize: d.vec2f,\n  faceActive: d.f32,\n  handPosition: d.vec2f,\n  handSize: d.vec2f,\n  handActive: d.f32,\n});`;
    if (!source.includes(oldParams)) throw new Error('RelightParams shape not found in TypeGPU shaders.ts');
    source = source.replace(oldParams, newParams);

    const oldSurface = `  const front = relightLayout.$.params.lightZ + BULB_WORLD_RADIUS * dome;\n  const solid = std.smoothstep(d.f32(0), BULB_OCCLUSION_SOFTNESS, front - surfaceZ(depth));`;
    const newSurface = `  // Convert interactive lightZ into the same camera-space Z as the depth surface.\n  // lightZ=0.10 is far/behind; larger values move the bulb toward the camera.\n  // This depth test is retained for the body/hand occlusion path.\n  const bulbSceneZ = std.mix(\n    d.f32(SURFACE_FAR_Z + 0.02),\n    d.f32(NEAR_Z + 0.005),\n    std.saturate((relightLayout.$.params.lightZ - d.f32(0.10)) / d.f32(1.15)),\n  );\n  const front = bulbSceneZ + BULB_WORLD_RADIUS * dome;\n  const depthSolid = std.smoothstep(d.f32(0), BULB_OCCLUSION_SOFTNESS, front - surfaceZ(depth));\n\n  // Face is a deliberate foreground exception for the visible bulb. The\n  // exception is restricted to the tracked face ellipse and is suppressed\n  // where the tracked hand overlaps it, so a palm crossing the face still\n  // occludes the bulb. The illumination calculation remains unchanged.\n  const faceDelta = (uv - relightLayout.$.params.facePosition) / std.max(relightLayout.$.params.faceSize, d.vec2f(0.001));\n  const faceRegion = (1 - std.smoothstep(0.78, 1.08, std.length(faceDelta))) * relightLayout.$.params.faceActive;\n  const handDelta = (uv - relightLayout.$.params.handPosition) / std.max(relightLayout.$.params.handSize, d.vec2f(0.001));\n  const handRegion = (1 - std.smoothstep(0.72, 1.12, std.length(handDelta))) * relightLayout.$.params.handActive;\n  const facePass = faceRegion * (1 - handRegion);\n  const solid = std.max(depthSolid, facePass);`;
    if (!source.includes(oldSurface)) throw new Error('Bulb depth surface test not found in TypeGPU shaders.ts');
    source = source.replace(oldSurface, newSurface);
    return source;
  }

  if (rel === 'renderer.ts') {
    const oldState = `  readonly mode: number;\n}`;
    const newState = `  readonly mode: number;\n  readonly facePosition: readonly [number, number];\n  readonly faceSize: readonly [number, number];\n  readonly faceActive: number;\n  readonly handPosition: readonly [number, number];\n  readonly handSize: readonly [number, number];\n  readonly handActive: number;\n}`;
    if (!source.includes(oldState)) throw new Error('RelightingState shape not found in TypeGPU renderer.ts');
    source = source.replace(oldState, newState);

    const oldDefaults = `  mode: RelightMode.RELIT,\n};`;
    const newDefaults = `  mode: RelightMode.RELIT,\n  facePosition: [0.5, 0.42],\n  faceSize: [0.22, 0.28],\n  faceActive: 0,\n  handPosition: [0.5, 0.5],\n  handSize: [0.18, 0.22],\n  handActive: 0,\n};`;
    if (!source.includes(oldDefaults)) throw new Error('Relighting defaults not found in TypeGPU renderer.ts');
    source = source.replace(oldDefaults, newDefaults);

    const oldWrite = `      mode: this.#settings.mode,\n    });`;
    const newWrite = `      mode: this.#settings.mode,\n      facePosition: d.vec2f(...this.#settings.facePosition),\n      faceSize: d.vec2f(...this.#settings.faceSize),\n      faceActive: this.#settings.faceActive,\n      handPosition: d.vec2f(...this.#settings.handPosition),\n      handSize: d.vec2f(...this.#settings.handSize),\n      handActive: this.#settings.handActive,\n    });`;
    if (!source.includes(oldWrite)) throw new Error('Relight parameter write block not found in TypeGPU renderer.ts');
    source = source.replace(oldWrite, newWrite);
    return source;
  }

  return source;
}

const files = await walk();
for (const rel of files) {
  const r = await fetch(`${RAW_ROOT}/${rel}`, { headers: { 'User-Agent': 'gpu-depth-lighting-build' } });
  if (!r.ok) throw new Error(`TypeGPU source download failed for ${rel}: ${r.status}`);
  const dest = join(DEST, rel);
  await mkdir(dirname(dest), { recursive: true });
  let source = await r.text();
  source = patchDepthOcclusion(source, rel);
  await writeFile(dest, `// @ts-nocheck\n${source}`, 'utf8');
}
console.log(`Synced ${files.length} TypeGPU monocular-depth source files from ${TYPEGPU_REF} into ${DEST}`);

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
  if (rel !== 'shaders.ts') return source;

  // The TypeGPU depth surface is in camera-space Z [-0.7, 0], while the
  // application's interactive lightZ is [0.10, 1.25]. Convert only the
  // visible-bulb depth test into that same surface space. The actual light
  // position/intensity/shadow calculations are intentionally untouched.
  const old = `const BULB_OCCLUSION_SOFTNESS = 0.02;`;
  const replacement = `const BULB_OCCLUSION_SOFTNESS = 0.035;`;
  if (!source.includes(old)) throw new Error('Bulb occlusion constant not found in TypeGPU shaders.ts');
  source = source.replace(old, replacement);

  const oldSurface = `  const front = relightLayout.$.params.lightZ + BULB_WORLD_RADIUS * dome;\n  const solid = std.smoothstep(d.f32(0), BULB_OCCLUSION_SOFTNESS, front - surfaceZ(depth));`;
  const newSurface = `  // Convert interactive lightZ into the same camera-space Z as the depth surface.\n  // lightZ=0.10 is far/behind the visible surface; lightZ=1.25 is near/front.\n  // This affects only bulb visibility: illumination remains driven by the original light.\n  const bulbSceneZ = std.mix(\n    d.f32(SURFACE_FAR_Z + 0.02),\n    d.f32(NEAR_Z + 0.005),\n    std.saturate((relightLayout.$.params.lightZ - d.f32(0.10)) / d.f32(1.15)),\n  );\n  const front = bulbSceneZ + BULB_WORLD_RADIUS * dome;\n  const solid = std.smoothstep(d.f32(0), BULB_OCCLUSION_SOFTNESS, front - surfaceZ(depth));`;
  if (!source.includes(oldSurface)) throw new Error('Bulb depth-occlusion surface test not found in TypeGPU shaders.ts');
  source = source.replace(oldSurface, newSurface);

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

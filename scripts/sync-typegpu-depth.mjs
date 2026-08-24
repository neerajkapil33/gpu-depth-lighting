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

  // TypeGPU's reference shader keeps the virtual bulb in a positive camera-Z
  // space while the monocular surface is represented in [-0.7, 0]. That makes
  // the reference bulb test always report the bulb as in front of the surface.
  // Map the user-controlled lightZ into the same surface-Z space for the bulb
  // visibility test only. Lighting/shadows keep the original light coordinates.
  const old = `const front = relightLayout.$.params.lightZ + BULB_WORLD_RADIUS * dome;\n  const solid = std.smoothstep(d.f32(0), BULB_OCCLUSION_SOFTNESS, front - surfaceZ(depth));`;
  const replacement = `const bulbSceneZ = std.mix(\n    d.f32(SURFACE_FAR_Z + 0.02),\n    d.f32(NEAR_Z - 0.005),\n    std.saturate((relightLayout.$.params.lightZ - d.f32(0.10)) / d.f32(1.15)),\n  );\n  const front = bulbSceneZ + BULB_WORLD_RADIUS * dome;\n  const solid = std.smoothstep(d.f32(0), BULB_OCCLUSION_SOFTNESS, front - surfaceZ(depth));`;

  if (!source.includes(old)) {
    throw new Error('Depth-occlusion patch target not found in TypeGPU shaders.ts');
  }
  return source.replace(old, replacement);
}

const files = await walk();
for (const rel of files) {
  const r = await fetch(`${RAW_ROOT}/${rel}`, { headers: { 'User-Agent': 'gpu-depth-lighting-build' } });
  if (!r.ok) throw new Error(`TypeGPU source download failed for ${rel}: ${r.status}`);
  const dest = join(DEST, rel);
  await mkdir(dirname(dest), { recursive: true });
  let source = await r.text();
  source = patchDepthOcclusion(source, rel);
  // TypeGPU's GPU DSL relies on operator overloads that strict tsc cannot
  // validate reliably across generated WGSL vector types. Vite still transpiles
  // this executable vendor source; the application's own code remains strict.
  await writeFile(dest, `// @ts-nocheck\n${source}`, 'utf8');
}
console.log(`Synced ${files.length} TypeGPU monocular-depth source files from ${TYPEGPU_REF} into ${DEST}`);

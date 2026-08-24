import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';

// Keep the vendored example aligned with the TypeGPU package used by this app.
// Pulling TypeGPU `main` caused hundreds of WGSL vector typing errors because
// the example and published runtime were on different API generations.
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
    if (e.type === 'dir') {
      files.push(...await walk(e.path.replace('apps/typegpu-docs/src/examples/image-processing/monocular-light-injection/', '')));
    } else if (e.type === 'file' && e.name.endsWith('.ts')) {
      files.push(e.path.replace('apps/typegpu-docs/src/examples/image-processing/monocular-light-injection/', ''));
    }
  }
  return files;
}

const files = await walk();
for (const rel of files) {
  const r = await fetch(`${RAW_ROOT}/${rel}`, { headers: { 'User-Agent': 'gpu-depth-lighting-build' } });
  if (!r.ok) throw new Error(`TypeGPU source download failed for ${rel}: ${r.status}`);
  const dest = join(DEST, rel);
  await mkdir(dirname(dest), { recursive: true });
  await writeFile(dest, await r.text(), 'utf8');
}
console.log(`Synced ${files.length} TypeGPU monocular-depth source files from ${TYPEGPU_REF} into ${DEST}`);

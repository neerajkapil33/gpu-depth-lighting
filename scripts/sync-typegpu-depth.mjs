import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';

const API_ROOT='https://api.github.com/repos/software-mansion/TypeGPU/contents/apps/typegpu-docs/src/examples/image-processing/monocular-light-injection';
const RAW_ROOT='https://raw.githubusercontent.com/software-mansion/TypeGPU/main/apps/typegpu-docs/src/examples/image-processing/monocular-light-injection';
// main.ts imports the runtime from the repository root, so vendor it there.
const DEST='vendor/typegpu-depth';

async function list(path=''){
  const url=path?`${API_ROOT}/${path}`:API_ROOT;
  const r=await fetch(url,{headers:{'User-Agent':'gpu-depth-lighting-build'}});
  if(!r.ok) throw new Error(`TypeGPU source listing failed: ${r.status}`);
  return r.json();
}
async function walk(path=''){
  const entries=await list(path); const files=[];
  for(const e of entries){
    if(e.type==='dir') files.push(...await walk(e.path.replace('apps/typegpu-docs/src/examples/image-processing/monocular-light-injection/','')));
    else if(e.type==='file' && e.name.endsWith('.ts')) files.push(e.path.replace('apps/typegpu-docs/src/examples/image-processing/monocular-light-injection/',''));
  }
  return files;
}
const files=await walk();
for(const rel of files){
  const r=await fetch(`${RAW_ROOT}/${rel}`,{headers:{'User-Agent':'gpu-depth-lighting-build'}});
  if(!r.ok) throw new Error(`TypeGPU source download failed for ${rel}: ${r.status}`);
  const dest=join(DEST,rel); await mkdir(dirname(dest),{recursive:true}); await writeFile(dest,await r.text(),'utf8');
}
console.log(`Synced ${files.length} TypeGPU monocular-depth source files into ${DEST}`);

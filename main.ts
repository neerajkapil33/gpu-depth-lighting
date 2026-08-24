import { d, tgpu } from 'typegpu';
import { FilesetResolver, HandLandmarker, type HandLandmarkerResult } from '@mediapipe/tasks-vision';
import { DepthInferencePlan } from './vendor/typegpu-depth/inference/depthart.ts';
import { parseDepthBundle } from './vendor/typegpu-depth/inference/bundle.ts';
import { fetchModel, modelVariant, RECOMMENDED_MODEL } from './vendor/typegpu-depth/model-store.ts';
import { DepthRelightingRenderer, defaultRelightingSettings } from './vendor/typegpu-depth/renderer.ts';
import { RelightMode } from './vendor/typegpu-depth/shaders.ts';
import { setupLightInput } from './light-input.ts';

const canvas=document.querySelector<HTMLCanvasElement>('#output')!;
const video=document.querySelector<HTMLVideoElement>('#video')!;
const start=document.querySelector<HTMLButtonElement>('#start')!;
const status=document.querySelector<HTMLElement>('#status')!;
const handIndicator=document.querySelector<HTMLElement>('#hand-indicator');
const cpuMs=document.querySelector<HTMLElement>('#cpu-ms');
const fpsEl=document.querySelector<HTMLElement>('#fps');
const inputSize=document.querySelector<HTMLElement>('#input-size');
const environment=document.querySelector<HTMLElement>('#environment');
let root:Awaited<ReturnType<typeof tgpu.init>>|undefined;
let renderer:DepthRelightingRenderer|undefined;
let plan:DepthInferencePlan|undefined;
let hand:HandLandmarker|undefined;
let running=false;
let lightActive=false;
let palmControl=false;
let handVisible=false;
let mediaRecorder:MediaRecorder|undefined;
let recordingChunks:BlobPart[]=[];
let recordingBlob:Blob|undefined;
let recordingStream:MediaStream|undefined;
let last=performance.now();
let frames=0;
const settings={...defaultRelightingSettings,lightPosition:[...defaultRelightingSettings.lightPosition] as [number,number],lightColor:[...defaultRelightingSettings.lightColor] as [number,number,number]};
function setStatus(text:string){status.textContent=text;}
function syncRenderer(){renderer?.update({lightPosition:settings.lightPosition,lightZ:settings.lightZ,intensity:lightActive?settings.intensity:0,exposure:settings.exposure,relief:settings.relief,shadow:settings.shadow,occlusion:settings.occlusion,lightColor:settings.lightColor,mirror:settings.mirror,mode:settings.mode});}
function setLightActive(on:boolean){lightActive=on;syncRenderer();const button=document.getElementById('light-toggle') as HTMLButtonElement|null;if(button){button.textContent=on?'Virtual bulb ON · click to disable':'Activate virtual bulb';button.classList.toggle('active',on);}setStatus(on?'3D GPU point light ACTIVE · depth-aware relighting ON':'DepthART ready · activate the virtual bulb');updateBulb();}
function updateBulb(){if(handIndicator)handIndicator.style.display=handVisible?'block':'none';}
function setPalmControl(on:boolean){palmControl=on;const button=document.getElementById('palm-toggle') as HTMLButtonElement|null;if(button){button.textContent=on?'Palm control: ON · move your hand':'Palm control: OFF';button.classList.toggle('active',on);}if(on&&!hand)void setupHand();}
function setEnvironment(name:string){if(!environment)return;environment.className=`env-${name}`;document.querySelectorAll<HTMLButtonElement>('.env-option').forEach(b=>b.classList.toggle('active',b.dataset.env===name));if(name==='camera'){environment.style.display='none';}else{environment.style.display='block';}setStatus(name==='camera'?(lightActive?'3D GPU point light ACTIVE · camera environment':'DepthART ready · camera environment'): `${name.replace(/^./,c=>c.toUpperCase())} environment selected`);}
function bindControls(){
 const ranges=[['ctrl-intensity','intensity','ctrl-intensity-v'],['ctrl-ambient','exposure','ctrl-ambient-v'],['ctrl-relief','relief','ctrl-relief-v'],['ctrl-shadow','shadow','ctrl-shadow-v'],['ctrl-occlusion','occlusion','ctrl-occlusion-v']] as const;
 for(const [id,key,out] of ranges){const input=document.getElementById(id) as HTMLInputElement|null;const output=document.getElementById(out) as HTMLOutputElement|null;input?.addEventListener('input',()=>{const value=Number(input.value);(settings as any)[key]=value;if(output)output.value=value.toFixed(key==='intensity'?1:2);syncRenderer();});}
 document.querySelectorAll<HTMLButtonElement>('.tab').forEach(tab=>tab.addEventListener('click',()=>{document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('active',t===tab));document.querySelectorAll('.tab-panel').forEach(p=>p.classList.toggle('active',p.id===tab.dataset.tab));}));
 document.querySelectorAll<HTMLButtonElement>('.env-option').forEach(button=>button.addEventListener('click',()=>setEnvironment(button.dataset.env||'white')));
 document.getElementById('light-toggle')?.addEventListener('click',()=>setLightActive(!lightActive));
 document.getElementById('palm-toggle')?.addEventListener('click',()=>setPalmControl(!palmControl));
 document.getElementById('ctrl-color')?.addEventListener('input',event=>{const hex=(event.target as HTMLInputElement).value;const n=Number.parseInt(hex.slice(1),16);settings.lightColor=[((n>>16)&255)/255,((n>>8)&255)/255,(n&255)/255];syncRenderer();});
 document.getElementById('ctrl-view')?.addEventListener('change',event=>{settings.mode=(event.target as HTMLSelectElement).value==='camera'?RelightMode.CAMERA:RelightMode.RELIT;syncRenderer();});
 document.getElementById('ctrl-camera')?.addEventListener('change',event=>{settings.mirror=(event.target as HTMLSelectElement).value==='front';syncRenderer();});
 document.getElementById('ctrl-source')?.addEventListener('change',event=>{settings.mode=(event.target as HTMLSelectElement).value==='camera'?RelightMode.CAMERA:RelightMode.RELIT;syncRenderer();});
 document.getElementById('ctrl-reset')?.addEventListener('click',()=>{Object.assign(settings,{...defaultRelightingSettings,lightPosition:[...defaultRelightingSettings.lightPosition] as [number,number],lightColor:[...defaultRelightingSettings.lightColor] as [number,number,number]});setLightActive(true);syncRenderer();});
 const recordButton=document.getElementById('record') as HTMLButtonElement|null;const downloadButton=document.getElementById('download-recording') as HTMLButtonElement|null;
 recordButton?.addEventListener('click',()=>{if(mediaRecorder?.state==='recording'){mediaRecorder.stop();recordButton.textContent='🔴 Record session';recordButton.classList.remove('recording');return;}if(!('MediaRecorder' in window)){setStatus('Recording is not supported by this browser.');return;}recordingChunks=[];recordingBlob=undefined;downloadButton&&(downloadButton.disabled=true);const stream=canvas.captureStream(60);recordingStream=stream;let mime='video/webm;codecs=vp9';if(!MediaRecorder.isTypeSupported(mime))mime='video/webm';mediaRecorder=new MediaRecorder(stream,{mimeType:mime});mediaRecorder.ondataavailable=e=>{if(e.data.size)recordingChunks.push(e.data);};mediaRecorder.onstop=()=>{recordingBlob=new Blob(recordingChunks,{type:mime});if(downloadButton)downloadButton.disabled=false;recordingStream?.getTracks().forEach(t=>t.stop());setStatus('Recording ready · download the rendered session');};mediaRecorder.start(250);recordButton.textContent='⏹ Stop recording';recordButton.classList.add('recording');setStatus('Recording TypeGPU rendered session…');});
 downloadButton?.addEventListener('click',()=>{if(!recordingBlob)return;const url=URL.createObjectURL(recordingBlob);const a=document.createElement('a');a.href=url;a.download=`virtual-orbit-light-${new Date().toISOString().replace(/[:.]/g,'-')}.webm`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
}
async function setupHand(){if(hand)return;try{const vision=await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.17/wasm');hand=await HandLandmarker.createFromOptions(vision,{baseOptions:{modelAssetPath:'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',delegate:'GPU'},runningMode:'VIDEO',numHands:1});setStatus(palmControl?'Palm control ready · move your palm to move the bulb':'DepthART ready · mouse/orbit control active');}catch(error){palmControl=false;const button=document.getElementById('palm-toggle') as HTMLButtonElement|null;if(button)button.textContent='Palm control: unavailable';setStatus(`Palm control unavailable: ${(error as Error).message}`);}}
function updatePalm(now:number){if(!palmControl||!hand||!lightActive){handVisible=false;return;}const result:HandLandmarkerResult=hand.detectForVideo(video,now);const lm=result.landmarks[0];if(!lm){handVisible=false;return;}let x=0,y=0;for(const i of [0,5,9,13,17]){x+=lm[i].x;y+=lm[i].y;}x/=5;y/=5;settings.lightPosition=[x,y];renderer?.update({lightPosition:settings.lightPosition});handVisible=true;if(handIndicator){handIndicator.style.left=`${x*canvas.clientWidth}px`;handIndicator.style.top=`${y*canvas.clientHeight}px`;}}
async function loadDepthModel(){if(!root)throw new Error('TypeGPU root is not initialized');const variant=modelVariant(RECOMMENDED_MODEL,root.device.features.has('shader-f16'));if(!variant)throw new Error('No compatible DepthART model variant available.');setStatus(`Loading TypeGPU DepthART 448×448 (${variant.megabytes} MB)…`);const bundle=parseDepthBundle(await fetchModel(variant,new AbortController().signal));plan=new DepthInferencePlan(root,bundle);await plan.initAsync();renderer=new DepthRelightingRenderer(root,canvas);await renderer.initAsync();renderer.attach(plan);syncRenderer();}
function renderFrame(now:number){if(!running||!renderer)return;const t0=performance.now();updatePalm(now);if(lightActive&&!palmControl)lightInput.orbitTick();syncRenderer();renderer.render({source:video,uvTransform:d.mat2x2f.identity(),swapAxes:false});updateBulb();frames++;if(now-last>500){fpsEl&&(fpsEl.textContent=`${Math.round(frames*1000/(now-last))} fps`);frames=0;last=now;}cpuMs&&(cpuMs.textContent=`${(performance.now()-t0).toFixed(1)} ms`);}
const lightInput=setupLightInput(canvas,update=>{if(update.lightPosition)settings.lightPosition=[...update.lightPosition] as [number,number];if(update.lightZ!==undefined)settings.lightZ=update.lightZ;syncRenderer();updateBulb();},new AbortController().signal);
start.addEventListener('click',async()=>{if(running)return;try{start.disabled=true;setStatus('Requesting camera…');const stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720},frameRate:{ideal:60,max:60},facingMode:'user'},audio:false});video.srcObject=stream;await video.play();inputSize&&(inputSize.textContent=`${video.videoWidth}×${video.videoHeight}`);root=await tgpu.init({device:{optionalFeatures:['shader-f16']}});await loadDepthModel();running=true;setLightActive(true);setEnvironment('white');const tick=(time:number)=>{renderFrame(time);if(running){if(video.requestVideoFrameCallback)video.requestVideoFrameCallback(tick);else requestAnimationFrame(tick);}};if(video.requestVideoFrameCallback)video.requestVideoFrameCallback(tick);else requestAnimationFrame(tick);void setupHand();}catch(error){start.disabled=false;setStatus(`Startup failed: ${(error as Error).message}`);}});
bindControls();setEnvironment('white');setStatus('Ready — start camera.');
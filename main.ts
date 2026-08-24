import './style.css';
import { compositeShader, normalShader, shadowShader } from './shaders';
import { FilesetResolver, FaceLandmarker, HandLandmarker, type FaceLandmarkerResult, type HandLandmarkerResult } from '@mediapipe/tasks-vision';
import { d } from 'typegpu';
import { Frame, initTypeGPU } from './typegpu-bridge';
import { estimateVideoDepth, getDepthAt, getSceneLuma, loadDepthModel } from './depth-estimator';

const W=960,H=540,WG=8;
const canvas=document.querySelector<HTMLCanvasElement>('#output')!;
const video=document.querySelector<HTMLVideoElement>('#video')!;
const start=document.querySelector<HTMLButtonElement>('#start')!;
const status=document.querySelector<HTMLElement>('#status')!;
const cpuMs=document.querySelector<HTMLElement>('#cpu-ms')!;
const fpsEl=document.querySelector<HTMLElement>('#fps')!;
const inputSize=document.querySelector<HTMLElement>('#input-size')!;
const handIndicator=document.querySelector<HTMLElement>('#hand-indicator')!;
const intensityControl=document.querySelector<HTMLInputElement>('#intensity');
const shadowControl=document.querySelector<HTMLInputElement>('#shadow');

let userIntensity=Number(intensityControl?.value??3);
let shadowStrength=Number(shadowControl?.value??0.94);
intensityControl?.addEventListener('input',()=>{userIntensity=Number(intensityControl.value);const v=document.querySelector('#intensity-value');if(v)v.textContent=userIntensity.toFixed(1);});
shadowControl?.addEventListener('input',()=>{shadowStrength=Number(shadowControl.value);const v=document.querySelector('#shadow-value');if(v)v.textContent=shadowStrength.toFixed(2);});

type Gpu={device:GPUDevice;context:GPUCanvasContext;format:GPUTextureFormat;depth:GPUTexture;normal:GPUTexture;shadow:GPUTexture;sampler:GPUSampler;normals:GPUComputePipeline;shadows:GPUComputePipeline;composite:GPURenderPipeline};
let gpu:Gpu;let root:ReturnType<typeof initTypeGPU>;let uniform:ReturnType<ReturnType<typeof initTypeGPU>['createUniform']>;
let hand:HandLandmarker|null=null,face:FaceLandmarker|null=null;
let lightX=.5,lightY=.32,lightZ=.35,lightIntensity=3;
let lastDepthAt=0,depthInFlight=false,lastFrame=performance.now(),frameCount=0,started=false;
let held=true,handIndex=0,vx=0,vy=0;
const prev:[{x:number,y:number}|null,{x:number,y:number}|null]=[null,null];const palms:[{x:number,y:number}|null,{x:number,y:number}|null]=[null,null];

async function visionSetup(){const v=await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.17/wasm');hand=await HandLandmarker.createFromOptions(v,{baseOptions:{modelAssetPath:'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',delegate:'GPU'},runningMode:'VIDEO',numHands:2});face=await FaceLandmarker.createFromOptions(v,{baseOptions:{modelAssetPath:'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',delegate:'GPU'},runningMode:'VIDEO',numFaces:1,outputFacialTransformationMatrixes:true});}
function track(now:number){
 palms[0]=palms[1]=null;
 if(hand){const r:HandLandmarkerResult=hand.detectForVideo(video,now);for(let i=0;i<Math.min(2,r.landmarks.length);i++){let x=0,y=0;for(const j of [0,5,9,13,17]){x+=r.landmarks[i][j].x;y+=r.landmarks[i][j].y;}palms[i]={x:x/5,y:y/5};}}
 for(let i=0;i<2;i++){const p=palms[i],q=prev[i];if(p&&q){vx=p.x-q.x;vy=p.y-q.y;}prev[i]=p;}
 const p=palms[handIndex];
 if(held&&p){lightX=p.x;lightY=p.y;if(Math.hypot(vx,vy)>.028){held=false;vx*=3.2;vy*=3.2;}}
 else if(!held){vy+=.0011;vx*=.994;vy*=.994;lightX+=vx;lightY+=vy;if(lightX<.02||lightX>.98){lightX=Math.max(.02,Math.min(.98,lightX));vx=-vx*.55;}if(lightY<.02||lightY>.98){lightY=Math.max(.02,Math.min(.98,lightY));vy=-vy*.55;}for(let i=0;i<2;i++){const q=palms[i];if(q&&Math.hypot(lightX-q.x,lightY-q.y)<.1){held=true;handIndex=i;vx=vy=0;break;}}}
 // The bulb lives just in front of the surface it is visually sitting against.
 // If it is over the face, getDepthAt() places it in front of the face, which
 // makes the face receive direct light while the wall behind remains shadowable.
 const surfaceAtLight=getDepthAt(lightX,lightY); lightZ=Math.max(.045,Math.min(.78,surfaceAtLight-.16));
 if(face){const fr:FaceLandmarkerResult=face.detectForVideo(video,now);const pts=fr.faceLandmarks[0];if(pts?.length){const c=pts[1]??pts[0];const faceDepth=getDepthAt(c.x,c.y);const faceDist=Math.hypot(lightX-c.x,lightY-c.y);lightIntensity=userIntensity*(1+Math.max(0,1-faceDist*4)*.35)*(1+Math.max(0,.45-faceDepth)*.25);}}
 else lightIntensity=userIntensity;
 const visible=palms[0]||palms[1];handIndicator.style.display=visible&&held?'flex':'none';if(visible&&held){handIndicator.style.left=`${lightX*canvas.clientWidth}px`;handIndicator.style.top=`${lightY*canvas.clientHeight}px`;}
}
function tex(format:GPUTextureFormat,usage:GPUTextureUsageFlags){return gpu.device.createTexture({size:[W,H],format,usage,label:`orbit-${format}`});}
async function init(){
 if(!navigator.gpu)throw Error('WebGPU is not available. Use Chrome or Edge with WebGPU enabled.');const adapter=await navigator.gpu.requestAdapter();if(!adapter)throw Error('No GPU adapter.');const device=await adapter.requestDevice();root=initTypeGPU(device);uniform=root.createUniform(Frame);
 const context=canvas.getContext('webgpu')!;const format=navigator.gpu.getPreferredCanvasFormat();context.configure({device,format,alphaMode:'opaque'});
 gpu={device,context,format,depth:null as never,normal:null as never,shadow:null as never,sampler:null as never,normals:null as never,shadows:null as never,composite:null as never};
 gpu.depth=tex('r32float',GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST);gpu.normal=tex('rgba16float',GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING);gpu.shadow=tex('r16float',GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING);gpu.sampler=device.createSampler({magFilter:'linear',minFilter:'linear'});
 const n=device.createShaderModule({code:normalShader}),s=device.createShaderModule({code:shadowShader}),c=device.createShaderModule({code:compositeShader});
 gpu.normals=device.createComputePipeline({layout:'auto',compute:{module:n,entryPoint:'reconstructNormals'}});gpu.shadows=device.createComputePipeline({layout:'auto',compute:{module:s,entryPoint:'projectShadow'}});gpu.composite=device.createRenderPipeline({layout:'auto',vertex:{module:c,entryPoint:'fullscreen'},fragment:{module:c,entryPoint:'composite',targets:[{format}]},primitive:{topology:'triangle-list'}});
}
function uploadDepth(depth:Float32Array){gpu.device.queue.writeTexture({texture:gpu.depth},depth,{bytesPerRow:W*4,rowsPerImage:H},{width:W,height:H,depthOrArrayLayers:1});}
function render(now:number){
 if(video.readyState<HTMLMediaElement.HAVE_CURRENT_DATA){requestAnimationFrame(render);return;}track(now);
 if(!depthInFlight&&now-lastDepthAt>110){depthInFlight=true;lastDepthAt=now;estimateVideoDepth(video,W,H).then(d=>{if(d)uploadDepth(d);}).catch(e=>{console.error('depth',e);status.textContent='Depth inference error — retrying…';}).finally(()=>depthInFlight=false);}
 uniform.write({outputSize:d.vec2u(W,H),time:now/1000,sceneLuma:getSceneLuma(),light:d.vec4f(lightX,lightY,lightZ,lightIntensity)});
 const dev=gpu.device,enc=dev.createCommandEncoder({label:'virtual-orbit-point-light'}),ub={buffer:uniform.buffer};
 const np=enc.beginComputePass({label:'depth normals'});np.setPipeline(gpu.normals);np.setBindGroup(0,dev.createBindGroup({layout:gpu.normals.getBindGroupLayout(0),entries:[{binding:0,resource:gpu.depth.createView()},{binding:1,resource:gpu.normal.createView()},{binding:2,resource:ub}]}));np.dispatchWorkgroups(Math.ceil(W/WG),Math.ceil(H/WG));np.end();
 const sp=enc.beginComputePass({label:'sharp cast shadows'});sp.setPipeline(gpu.shadows);sp.setBindGroup(0,dev.createBindGroup({layout:gpu.shadows.getBindGroupLayout(0),entries:[{binding:0,resource:gpu.depth.createView()},{binding:1,resource:gpu.shadow.createView()},{binding:2,resource:ub}]}));sp.dispatchWorkgroups(Math.ceil(W/WG),Math.ceil(H/WG));sp.end();
 const pass=enc.beginRenderPass({colorAttachments:[{view:gpu.context.getCurrentTexture().createView(),loadOp:'clear',storeOp:'store',clearValue:[0,0,0,1]}]});pass.setPipeline(gpu.composite);pass.setBindGroup(0,dev.createBindGroup({layout:gpu.composite.getBindGroupLayout(0),entries:[{binding:0,resource:dev.importExternalTexture({source:video})},{binding:1,resource:gpu.sampler},{binding:2,resource:gpu.depth.createView()},{binding:3,resource:gpu.normal.createView()},{binding:4,resource:gpu.shadow.createView()},{binding:5,resource:ub}]}));pass.draw(3);pass.end();dev.queue.submit([enc.finish()]);
 const dt=performance.now()-now;cpuMs.textContent=`${dt.toFixed(1)} ms`;frameCount++;if(now-lastFrame>500){fpsEl.textContent=`${Math.round(frameCount*1000/(now-lastFrame))} fps`;frameCount=0;lastFrame=now;}requestAnimationFrame(render);
}
start.addEventListener('click',async()=>{try{const stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720},facingMode:'user'},audio:false});video.srcObject=stream;await video.play();inputSize.textContent=`${video.videoWidth}×${video.videoHeight}`;start.disabled=true;status.textContent='Loading dense depth model…';started=true;requestAnimationFrame(render);}catch(e){status.textContent=`Camera failed: ${(e as Error).message}`;}});
(async()=>{try{status.textContent='Initializing WebGPU + TypeGPU…';await init();status.textContent='Loading real dense monocular depth…';await loadDepthModel(m=>status.textContent=m);status.textContent='Loading face + hand tracking…';await visionSetup();start.disabled=false;status.textContent='Ready — start camera.';}catch(e){status.textContent=(e as Error).message;console.error(e);}})();

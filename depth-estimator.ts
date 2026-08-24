import { RawImage, pipeline, type DepthEstimationPipeline } from '@huggingface/transformers';
const MODEL='onnx-community/depth-anything-v2-small';
const INPUT_W=518, INPUT_H=518;
let estimator:DepthEstimationPipeline|null=null, running=false, lastDepth:Float32Array|null=null, lastLuma=.5;
const sourceCanvas=document.createElement('canvas'); sourceCanvas.width=INPUT_W; sourceCanvas.height=INPUT_H;
const sourceCtx=sourceCanvas.getContext('2d',{willReadFrequently:true})!;
export async function loadDepthModel(onProgress?:(message:string)=>void):Promise<void>{
 onProgress?.('Loading Depth Anything V2 Small · WebGPU q4f16…');
 estimator=await pipeline('depth-estimation',MODEL,{device:'webgpu',dtype:'q4f16',progress_callback:(p:any)=>{if(p?.status==='progress'||p?.status==='progress_total'){const pct=typeof p.progress==='number'?` ${Math.round(p.progress)}%`:'';onProgress?.(`Loading Depth Anything V2 Small · WebGPU${pct}`);}}});
 onProgress?.('Depth Anything V2 ready · WebGPU q4f16.');
}
export function isDepthReady(){return estimator!==null;}
export function getSceneLuma(){return lastLuma;}
export async function estimateVideoDepth(video:HTMLVideoElement,W:number,H:number):Promise<Float32Array|null>{
 if(!estimator||running||video.readyState<HTMLMediaElement.HAVE_CURRENT_DATA)return lastDepth; running=true;
 try{
  sourceCtx.drawImage(video,0,0,INPUT_W,INPUT_H); const px=sourceCtx.getImageData(0,0,INPUT_W,INPUT_H).data; let sum=0;
  for(let i=0;i<px.length;i+=4)sum+=(.2126*px[i]+.7152*px[i+1]+.0722*px[i+2])/255; lastLuma=sum/(px.length/4);
  const result:any=await estimator(RawImage.fromCanvas(sourceCanvas)); const depth=result.depth as RawImage; const src=depth.data;
  let min=255,max=0;for(let i=0;i<src.length;i++){const v=src[i];if(v<min)min=v;if(v>max)max=v;}const range=Math.max(1,max-min);
  const out=new Float32Array(W*H),sx=depth.width/W,sy=depth.height/H;
  for(let y=0;y<H;y++){const yy=Math.min(depth.height-1,Math.floor(y*sy));for(let x=0;x<W;x++){const xx=Math.min(depth.width-1,Math.floor(x*sx));const brightness=(src[yy*depth.width+xx]-min)/range;const cameraZ=.08+(1-brightness)*.92;const i=y*W+x;out[i]=lastDepth?lastDepth[i]*.82+cameraZ*.18:cameraZ;}}
  lastDepth=out;return out;
 }finally{running=false;}
}

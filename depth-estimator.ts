import { RawImage, pipeline } from '@huggingface/transformers';

const MODEL='onnx-community/depth-anything-v2-small';
const INPUT_W=518;
const INPUT_H=518;
const DEPTH_NEAR=0.06;
const DEPTH_FAR=1.0;

let estimator:any=null;
let running=false;
let lastDepth:Float32Array|null=null;
let lastW=0;
let lastH=0;
let lastLuma=.5;

const sourceCanvas=document.createElement('canvas');
sourceCanvas.width=INPUT_W;
sourceCanvas.height=INPUT_H;
const sourceCtx=sourceCanvas.getContext('2d',{willReadFrequently:true})!;

export async function loadDepthModel(onProgress?:(message:string)=>void){
  if(estimator)return;
  onProgress?.('Loading Depth Anything V2 Small · WebGPU q4f16…');
  estimator=await pipeline('depth-estimation',MODEL,{
    device:'webgpu',
    dtype:'q4f16',
    progress_callback:(p:any)=>{
      if(p?.status==='progress'||p?.status==='progress_total'){
        const pct=typeof p.progress==='number'?` ${Math.round(p.progress)}%`:'';
        onProgress?.(`Loading Depth Anything V2 Small · WebGPU${pct}`);
      }
    }
  });
  onProgress?.('Depth Anything V2 ready · dense monocular depth · WebGPU.');
}

export function isDepthReady(){return estimator!==null;}
export function getSceneLuma(){return lastLuma;}

export function getDepthAt(nx:number,ny:number){
  if(!lastDepth||!lastW||!lastH)return .65;
  const x=Math.max(0,Math.min(lastW-1,Math.floor(nx*lastW)));
  const y=Math.max(0,Math.min(lastH-1,Math.floor(ny*lastH)));
  return lastDepth[y*lastW+x];
}

export async function estimateVideoDepth(video:HTMLVideoElement,W:number,H:number):Promise<Float32Array|null>{
  if(!estimator||running||video.readyState<HTMLMediaElement.HAVE_CURRENT_DATA)return lastDepth;
  running=true;
  try{
    sourceCtx.drawImage(video,0,0,INPUT_W,INPUT_H);
    const px=sourceCtx.getImageData(0,0,INPUT_W,INPUT_H).data;
    let sum=0;
    for(let i=0;i<px.length;i+=4)sum+=(.2126*px[i]+.7152*px[i+1]+.0722*px[i+2])/255;
    lastLuma=sum/(px.length/4);

    const result:any=await estimator(RawImage.fromCanvas(sourceCanvas));
    const depth=result?.depth as RawImage|undefined;
    if(!depth?.data||!depth.width||!depth.height)return lastDepth;

    const src=depth.data as ArrayLike<number>;
    let min=Infinity,max=-Infinity;
    for(let i=0;i<src.length;i++){
      const v=Number(src[i]);
      if(Number.isFinite(v)){min=Math.min(min,v);max=Math.max(max,v);}
    }
    if(!Number.isFinite(min)||!Number.isFinite(max)||max-min<1e-6)return lastDepth;
    const range=max-min;

    const out=new Float32Array(W*H);
    const sx=depth.width/W;
    const sy=depth.height/H;
    for(let y=0;y<H;y++){
      const yy=Math.min(depth.height-1,Math.floor(y*sy));
      for(let x=0;x<W;x++){
        const xx=Math.min(depth.width-1,Math.floor(x*sx));
        const raw=(Number(src[yy*depth.width+xx])-min)/range;
        // Depth Anything relative-depth convention: higher predicted depth is nearer.
        const cameraZ=DEPTH_NEAR+(1-Math.max(0,Math.min(1,raw)))*(DEPTH_FAR-DEPTH_NEAR);
        const i=y*W+x;
        // Strong enough temporal filtering to stop shadow swimming, while still following a moving head.
        out[i]=lastDepth?lastDepth[i]*.72+cameraZ*.28:cameraZ;
      }
    }
    lastDepth=out;
    lastW=W;
    lastH=H;
    return out;
  }finally{
    running=false;
  }
}

export const normalShader = /* wgsl */ `
struct Frame { width:u32, height:u32, time:f32, sceneLuma:f32, lightX:f32, lightY:f32, lightZ:f32, lightIntensity:f32 }
@group(0) @binding(0) var depthIn:texture_2d<f32>;
@group(0) @binding(1) var normalsOut:texture_storage_2d<rgba16float,write>;
@group(0) @binding(2) var<uniform> frame:Frame;
fn z(p:vec2i)->f32{let m=vec2i(i32(frame.width)-1,i32(frame.height)-1);return textureLoad(depthIn,clamp(p,vec2i(0),m),0).x;}
@compute @workgroup_size(8,8)
fn reconstructNormals(@builtin(global_invocation_id) id:vec3u){
 if(id.x>=frame.width||id.y>=frame.height){return;}
 let p=vec2i(id.xy);
 let dx=z(p+vec2i(2,0))-z(p-vec2i(2,0));
 let dy=z(p+vec2i(0,2))-z(p-vec2i(0,2));
 let n=normalize(vec3f(-dx*1.7,-dy*1.7,0.62));
 textureStore(normalsOut,p,vec4f(n*0.5+0.5,1));
}`;

export const shadowShader = /* wgsl */ `
struct Frame { width:u32, height:u32, time:f32, sceneLuma:f32, lightX:f32, lightY:f32, lightZ:f32, lightIntensity:f32 }
@group(0) @binding(0) var depth:texture_2d<f32>;
@group(0) @binding(1) var shadowOut:texture_storage_2d<r16float,write>;
@group(0) @binding(2) var<uniform> frame:Frame;
fn z(p:vec2i)->f32{let m=vec2i(i32(frame.width)-1,i32(frame.height)-1);return textureLoad(depth,clamp(p,vec2i(0),m),0).x;}
@compute @workgroup_size(8,8)
fn projectShadow(@builtin(global_invocation_id) id:vec3u){
 if(id.x>=frame.width||id.y>=frame.height){return;}
 let p=vec2i(id.xy); let uv=(vec2f(id.xy)+0.5)/vec2f(f32(frame.width),f32(frame.height));
 let receiver=z(p); let toLight=vec2f(frame.lightX,frame.lightY)-uv; let dist=length(toLight);
 var blocked=0.0;
 // Trace from receiver toward the point light. A nearer depth sample between
 // the receiver and the light means an occluder (head/body/hand).
 for(var i:i32=1;i<=16;i++){
   let t=f32(i)/17.0;
   let q=vec2i((uv+toLight*t)*vec2f(f32(frame.width),f32(frame.height)));
   let sampleZ=z(q);
   blocked=max(blocked,smoothstep(0.012,0.060,receiver-sampleZ));
 }
 // Keep shadows crisp near the silhouette; only a tiny 1-pixel antialias remains.
 let edge=smoothstep(0.0,0.06,dist);
 textureStore(shadowOut,p,vec4f(blocked*(0.96+0.04*edge),0,0,1));
}`;

export const compositeShader = /* wgsl */ `
struct Frame { width:u32, height:u32, time:f32, sceneLuma:f32, lightX:f32, lightY:f32, lightZ:f32, lightIntensity:f32 }
@group(0) @binding(0) var camera:texture_external;
@group(0) @binding(1) var videoSampler:sampler;
@group(0) @binding(2) var depth:texture_2d<f32>;
@group(0) @binding(3) var normals:texture_2d<f32>;
@group(0) @binding(4) var shadow:texture_2d<f32>;
@group(0) @binding(5) var<uniform> frame:Frame;
struct VSOut{@builtin(position)position:vec4f;@location(0)uv:vec2f}
@vertex fn fullscreen(@builtin(vertex_index)i:u32)->VSOut{var p=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));var o:VSOut;o.position=vec4f(p[i],0,1);o.uv=vec2f(p[i].x*.5+.5,1-(p[i].y*.5+.5));return o;}

fn scenePoint(uv:vec2f,z:f32)->vec3f{
  let aspect=f32(frame.width)/f32(frame.height);
  let x=(uv.x-0.5)*aspect;
  let y=(0.5-uv.y);
  return vec3f(x,y,z*1.7);
}

@fragment fn composite(in:VSOut)->@location(0)vec4f{
 let src=textureSampleBaseClampToEdge(camera,videoSampler,in.uv).rgb;
 let p=vec2i(clamp(in.uv*vec2f(f32(frame.width),f32(frame.height)),vec2f(0),vec2f(f32(frame.width-1u),f32(frame.height-1u))));
 let z=textureLoad(depth,p,0).x;
 let n=normalize(textureLoad(normals,p,0).xyz*2.0-1.0);
 let sh=textureLoad(shadow,p,0).x;

 // Reconstruct an approximate camera-space point and place the bulb in the same space.
 let surface=scenePoint(in.uv,z);
 let lp=scenePoint(vec2f(frame.lightX,frame.lightY),frame.lightZ);
 let L=lp-surface;
 let distance=max(0.035,length(L));
 let ldir=L/distance;
 let lambert=max(dot(n,ldir),0.0);

 // Point-bulb inverse-square falloff. No broad image-wide scattering.
 let inverseSquare=1.0/(distance*distance+0.055);
 let darkness=1.0-clamp(frame.sceneLuma,0.0,1.0);
 let darkRoomGain=1.0+darkness*darkness*1.8;
 let depthSurfaceGain=0.55+0.75*smoothstep(0.95,0.18,z);
 let direct=lambert*inverseSquare*frame.lightIntensity*darkRoomGain*0.075*depthSurfaceGain;
 let lit=src + src*direct*(1.0-sh*0.94);

 // The bulb is a physical-looking emissive core: a small hard circular disk,
 // deliberately with no radial glow/bloom/scattering.
 let aspect=f32(frame.width)/f32(frame.height);
 let d=length(vec2f((frame.lightX-in.uv.x)*aspect,frame.lightY-in.uv.y));
 let coreRadius=0.018 + 0.006*frame.lightZ;
 let core=smoothstep(coreRadius+0.002,coreRadius,d);
 let bulb=vec3f(1.0,0.955,0.82)*(1.0-core)*1.35;

 return vec4f(clamp(lit+bulb,vec3f(0),vec3f(1)),1);
}`;

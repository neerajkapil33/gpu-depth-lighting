export const normalShader = /* wgsl */ `
struct Frame { width:u32, height:u32, time:f32, sceneLuma:f32, lightX:f32, lightY:f32, lightZ:f32, lightIntensity:f32 }
@group(0) @binding(0) var depthIn:texture_2d<f32>;
@group(0) @binding(1) var normalsOut:texture_storage_2d<rgba16float,write>;
@group(0) @binding(2) var<uniform> frame:Frame;
fn z(p:vec2i)->f32{let m=vec2i(i32(frame.width)-1,i32(frame.height)-1);return textureLoad(depthIn,clamp(p,vec2i(0),m),0).x;}
fn safeNormal(p:vec2i)->vec3f{let dx=(z(p+vec2i(3,0))-z(p-vec2i(3,0)))/6.0;let dy=(z(p+vec2i(0,3))-z(p-vec2i(0,3)))/6.0;return normalize(vec3f(-dx*3.6,-dy*3.6,1.0));}
@compute @workgroup_size(8,8)
fn reconstructNormals(@builtin(global_invocation_id) id:vec3u){if(id.x>=frame.width||id.y>=frame.height){return;}let n=safeNormal(vec2i(id.xy));textureStore(normalsOut,vec2i(id.xy),vec4f(n*0.5+0.5,1));}`;

export const shadowShader = /* wgsl */ `
struct Frame { width:u32, height:u32, time:f32, sceneLuma:f32, lightX:f32, lightY:f32, lightZ:f32, lightIntensity:f32 }
@group(0) @binding(0) var depth:texture_2d<f32>;
@group(0) @binding(1) var shadowOut:texture_storage_2d<r32float,write>;
@group(0) @binding(2) var<uniform> frame:Frame;
fn z(p:vec2i)->f32{let m=vec2i(i32(frame.width)-1,i32(frame.height)-1);return textureLoad(depth,clamp(p,vec2i(0),m),0).x;}
fn rayOcclusion(uv:vec2f,receiver:f32,offset:vec2f)->f32{let lightUv=vec2f(frame.lightX,frame.lightY)+offset;let toLight=lightUv-uv;var occlusion=0.0;for(var i:i32=1;i<=32;i++){let t=f32(i)/33.0;let qUv=uv+toLight*t;let q=vec2i(qUv*vec2f(f32(frame.width),f32(frame.height)));let sampled=z(q);let expectedZ=mix(receiver,frame.lightZ,t);let bias=0.008+0.018*t;let hit=smoothstep(0.006,0.032,expectedZ-sampled-bias);occlusion=max(occlusion,hit);}return occlusion;}
@compute @workgroup_size(8,8)
fn projectShadow(@builtin(global_invocation_id) id:vec3u){if(id.x>=frame.width||id.y>=frame.height){return;}let uv=(vec2f(id.xy)+0.5)/vec2f(f32(frame.width),f32(frame.height));let receiver=z(vec2i(id.xy));let px=1.35/vec2f(f32(frame.width),f32(frame.height));let c=rayOcclusion(uv,receiver,vec2f(0));let a=rayOcclusion(uv,receiver,vec2f(px.x*2.0,px.y));let b=rayOcclusion(uv,receiver,vec2f(-px.x*2.0,-px.y));let occlusion=c*.55+(a+b)*.225;let distanceToLight=length(vec2f((frame.lightX-uv.x)*f32(frame.width)/f32(frame.height),frame.lightY-uv.y));let contact=1.0-smoothstep(0.015,0.28,distanceToLight);let strength=0.76+0.24*contact;textureStore(shadowOut,vec2i(id.xy),vec4f(clamp(occlusion*strength,0.0,1.0),0,0,1));}`;

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
fn scenePoint(uv:vec2f,z:f32)->vec3f{let aspect=f32(frame.width)/f32(frame.height);return vec3f((uv.x-0.5)*aspect,(0.5-uv.y),z*1.7);}
@fragment fn composite(in:VSOut)->@location(0)vec4f{let src=textureSampleBaseClampToEdge(camera,videoSampler,in.uv).rgb;let p=vec2i(clamp(in.uv*vec2f(f32(frame.width),f32(frame.height)),vec2f(0),vec2f(f32(frame.width-1u),f32(frame.height-1u))));let z=textureLoad(depth,p,0).x;let n=normalize(textureLoad(normals,p,0).xyz*2.0-1.0);let sh=textureLoad(shadow,p,0).x;let surface=scenePoint(in.uv,z);let lp=scenePoint(vec2f(frame.lightX,frame.lightY),frame.lightZ);let L=lp-surface;let distance=max(0.035,length(L));let ldir=L/distance;let lambert=max(dot(n,ldir),0.0);let inverseSquare=1.0/(distance*distance+0.045);let darkness=1.0-clamp(frame.sceneLuma,0.0,1.0);let darkRoomGain=1.0+darkness*darkness*3.8;let depthGain=0.78+0.52*(1.0-smoothstep(0.08,0.92,z));let wrappedDiffuse=0.18+0.82*lambert;let direct=wrappedDiffuse*inverseSquare*frame.lightIntensity*darkRoomGain*0.075*depthGain;let visibleLight=1.0-sh*0.985;let warm=vec3f(1.0,0.78,0.52);let neutral=vec3f(1.0,0.96,0.90);let viewDir=normalize(vec3f(0.0,0.0,-1.0)-surface*0.08);let halfDir=normalize(ldir+viewDir);let spec=pow(max(dot(n,halfDir),0.0),34.0)*0.12*inverseSquare*frame.lightIntensity;let ambientLift=darkness*0.018;let lit=src*(1.0+direct*visibleLight+ambientLift)+warm*direct*0.10*visibleLight+neutral*spec*visibleLight;let aspect=f32(frame.width)/f32(frame.height);let d=length(vec2f((frame.lightX-in.uv.x)*aspect,frame.lightY-in.uv.y));let coreRadius=.010+.007*frame.lightZ;let haloRadius=coreRadius*3.4;let core=1.0-smoothstep(coreRadius,coreRadius+.0025,d);let halo=(1.0-smoothstep(coreRadius,haloRadius,d))*0.10;let bulb=vec3f(1.0,.96,.84)*(core*1.65+halo);return vec4f(clamp(lit+bulb,vec3f(0),vec3f(1)),1);}`;

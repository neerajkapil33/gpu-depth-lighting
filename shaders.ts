export const normalShader = /* wgsl */ `
struct Frame { width:u32, height:u32, time:f32, sceneLuma:f32, lightX:f32, lightY:f32, lightZ:f32, lightIntensity:f32 }
@group(0) @binding(0) var depthIn:texture_2d<f32>;
@group(0) @binding(1) var normalsOut:texture_storage_2d<rgba16float,write>;
@group(0) @binding(2) var<uniform> frame:Frame;
fn z(p:vec2i)->f32{let m=vec2i(i32(frame.width)-1,i32(frame.height)-1);return textureLoad(depthIn,clamp(p,vec2i(0),m),0).x;}
@compute @workgroup_size(8,8)
fn reconstructNormals(@builtin(global_invocation_id) id:vec3u){if(id.x>=frame.width||id.y>=frame.height){return;}let p=vec2i(id.xy);let dx=z(p+vec2i(2,0))-z(p-vec2i(2,0));let dy=z(p+vec2i(0,2))-z(p-vec2i(0,2));let n=normalize(vec3f(-dx*1.25,-dy*1.25,0.72));textureStore(normalsOut,p,vec4f(n*0.5+0.5,1));}
`;

export const shadowShader = /* wgsl */ `
struct Frame { width:u32, height:u32, time:f32, sceneLuma:f32, lightX:f32, lightY:f32, lightZ:f32, lightIntensity:f32 }
@group(0) @binding(0) var depth:texture_2d<f32>;
@group(0) @binding(1) var shadowOut:texture_storage_2d<r16float,write>;
@group(0) @binding(2) var<uniform> frame:Frame;
fn z(p:vec2i)->f32{let m=vec2i(i32(frame.width)-1,i32(frame.height)-1);return textureLoad(depth,clamp(p,vec2i(0),m),0).x;}
@compute @workgroup_size(8,8)
fn projectShadow(@builtin(global_invocation_id) id:vec3u){
 if(id.x>=frame.width||id.y>=frame.height){return;}
 let uv=(vec2f(id.xy)+0.5)/vec2f(f32(frame.width),f32(frame.height));
 let here=z(vec2i(id.xy)); let toLight=vec2f(frame.lightX,frame.lightY)-uv; var blocked=0.0;
 // Multi-tap screen-space ray test. Real dense depth makes the silhouette track
 // head/body geometry; the soft threshold prevents crawling hard edges.
 for(var i:i32=1;i<=12;i++){let t=f32(i)/13.0;let q=vec2i((uv+toLight*t)*vec2f(f32(frame.width),f32(frame.height)));let sampleZ=z(q);blocked=max(blocked,smoothstep(0.008,0.055,here-sampleZ));}
 let softness=smoothstep(0.02,0.30,length(toLight));
 textureStore(shadowOut,vec2i(id.xy),vec4f(blocked*(1.0-softness*0.45),0,0,1));
}
`;

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
@fragment fn composite(in:VSOut)->@location(0)vec4f{
 let src=textureSampleBaseClampToEdge(camera,videoSampler,in.uv).rgb;
 let p=vec2i(clamp(in.uv*vec2f(f32(frame.width),f32(frame.height)),vec2f(0),vec2f(f32(frame.width-1u),f32(frame.height-1u))));
 let z=textureLoad(depth,p,0).x;let n=normalize(textureLoad(normals,p,0).xyz*2-1);let sh=textureLoad(shadow,p,0).x;
 let aspect=f32(frame.width)/f32(frame.height);let toLight=vec2f((frame.lightX-in.uv.x)*aspect,frame.lightY-in.uv.y);let dist=length(toLight);
 let lightDir=normalize(vec3f(toLight.x,-toLight.y,max(0.12,frame.lightZ)));let diffuse=max(dot(n,lightDir),0);
 let falloff=1.0/max(0.12,dist*dist+frame.lightZ*frame.lightZ);
 let darkness=1.0-clamp(frame.sceneLuma,0,1);let adaptive=1.0+darkness*darkness*2.8;
 // Foreground depth receives stronger local illumination while the projected
 // shadow suppresses wall/background light behind the occluder.
 let nearBoost=1.0+smoothstep(0.82,0.12,z)*0.55;
 let boost=diffuse*falloff*frame.lightIntensity*adaptive*0.075*nearBoost*(1.0-sh*0.90);
 let orb=smoothstep(0.075,0.0,dist)*1.9;
 let warm=vec3f(1.0,0.88,0.67);
 let shaded=src*(0.58+boost)+warm*orb;
 return vec4f(clamp(shaded,vec3f(0),vec3f(1)),1);
}
`;

export const shadowShader = /* wgsl */ `
struct Frame { width:u32, height:u32, time:f32, sceneLuma:f32, lightX:f32, lightY:f32, lightZ:f32, lightIntensity:f32 }
@group(0) @binding(0) var depth:texture_2d<f32>;
@group(0) @binding(1) var shadowOut:texture_storage_2d<r16float,write>;
@group(0) @binding(2) var<uniform> frame:Frame;
fn dz(p:vec2i)->f32{let m=vec2i(i32(frame.width)-1,i32(frame.height)-1);return textureLoad(depth,clamp(p,vec2i(0),m),0).x;}
@compute @workgroup_size(8,8)
fn projectShadow(@builtin(global_invocation_id) id:vec3u){
 if(id.x>=frame.width||id.y>=frame.height){return;}
 let uv=(vec2f(id.xy)+0.5)/vec2f(f32(frame.width),f32(frame.height));
 let toLight=vec2f(frame.lightX,frame.lightY)-uv; let dist=length(toLight); let here=dz(vec2i(id.xy)); var blocked=0.0;
 for(var i:i32=1;i<=8;i++){let t=f32(i)/9.0;let p=vec2i((uv+toLight*t)*vec2f(f32(frame.width),f32(frame.height)));blocked=max(blocked,smoothstep(0.008,0.075,dz(p)-here));}
 textureStore(shadowOut,vec2i(id.xy),vec4f(blocked*(1.0-smoothstep(0.0,0.35,dist)*0.55),0,0,1));
}`;

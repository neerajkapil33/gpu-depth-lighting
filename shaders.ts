export const inferenceShader = /* wgsl */ `
struct Frame { width: u32, height: u32, time: f32, _pad: f32 }
@group(0) @binding(0) var camera: texture_external;
@group(0) @binding(1) var videoSampler: sampler;
// r32float is a portable writable WebGPU storage-texture format.
@group(0) @binding(2) var depthOut: texture_storage_2d<r32float, write>;
@group(0) @binding(3) var<uniform> frame: Frame;

// REPLACE THIS ENTRY POINT with the compiled/generative inference graph for
// DepthAnything (preprocess → encoder → decoder → depth). Its output contract
// is a normalized linear depth value in depthOut, and it must remain GPU-only.
@compute @workgroup_size(8, 8)
fn mockDepth(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= frame.width || id.y >= frame.height) { return; }
  let uv = (vec2f(id.xy) + 0.5) / vec2f(f32(frame.width), f32(frame.height));
  let rgb = textureSampleBaseClampToEdge(camera, videoSampler, uv).rgb;
  let luminance = dot(rgb, vec3f(0.2126, 0.7152, 0.0722));
  // This is intentionally NOT ML inference: a stable pseudo-depth proxy.
  let center = 1.0 - length(uv - vec2f(0.5)) * 1.15;
  let depth = clamp(0.22 + luminance * 0.38 + center * 0.4, 0.0, 1.0);
  textureStore(depthOut, vec2i(id.xy), vec4f(depth, 0.0, 0.0, 1.0));
}`;

export const normalShader = /* wgsl */ `
struct Frame { width: u32, height: u32, time: f32, _pad: f32 }
@group(0) @binding(0) var depthIn: texture_2d<f32>;
@group(0) @binding(1) var normalsOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> frame: Frame;
fn readDepth(p: vec2i) -> f32 {
  let maxP = vec2i(i32(frame.width) - 1, i32(frame.height) - 1);
  return textureLoad(depthIn, clamp(p, vec2i(0), maxP), 0).x;
}
@compute @workgroup_size(8, 8)
fn reconstructNormals(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= frame.width || id.y >= frame.height) { return; }
  let p = vec2i(id.xy);
  // Sample across a wider span so fine camera noise does not turn into harsh relief.
  let dx = readDepth(p + vec2i(2, 0)) - readDepth(p - vec2i(2, 0));
  let dy = readDepth(p + vec2i(0, 2)) - readDepth(p - vec2i(0, 2));
  let n = normalize(vec3f(-dx * 0.65, -dy * 0.65, 0.70));
  textureStore(normalsOut, p, vec4f(n * 0.5 + 0.5, 1.0));
}`;

export const compositeShader = /* wgsl */ `
struct Frame { width: u32, height: u32, time: f32, _pad0: f32, lightX: f32, lightY: f32, _pad1: f32, _pad2: f32 }
@group(0) @binding(0) var camera: texture_external;
@group(0) @binding(1) var videoSampler: sampler;
@group(0) @binding(2) var depth: texture_2d<f32>;
@group(0) @binding(3) var normals: texture_2d<f32>;
@group(0) @binding(4) var<uniform> frame: Frame;

struct VSOut { @builtin(position) position: vec4f, @location(0) uv: vec2f }
@vertex fn fullscreen(@builtin(vertex_index) i: u32) -> VSOut {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var out: VSOut;
  out.position = vec4f(p[i], 0.0, 1.0);
  out.uv = vec2f(p[i].x * 0.5 + 0.5, 1.0 - (p[i].y * 0.5 + 0.5));
  return out;
}
@fragment fn composite(in: VSOut) -> @location(0) vec4f {
  let src = textureSampleBaseClampToEdge(camera, videoSampler, in.uv).rgb;
  let p = vec2i(clamp(
    in.uv * vec2f(f32(frame.width), f32(frame.height)),
    vec2f(0.0),
    vec2f(f32(frame.width - 1u), f32(frame.height - 1u))
  ));
  let z = textureLoad(depth, p, 0).x;
  let n = normalize(textureLoad(normals, p, 0).xyz * 2.0 - 1.0);
  let aspect = f32(frame.width) / f32(frame.height);
  // Screen-space vector from this pixel to the virtual light (cursor-controlled orb).
  let toLight2D = vec2f((frame.lightX - in.uv.x) * aspect, frame.lightY - in.uv.y);
  let dist = length(toLight2D);
  let lightDir = normalize(vec3f(toLight2D.x, -toLight2D.y, 0.28));
  let diffuse = max(dot(n, lightDir), 0.0);
  // Falloff: bright near the light, fading out with distance across the frame.
  let falloff = clamp(1.0 - dist * 1.35, 0.0, 1.0);
  // Local depth discontinuity is a cheap screen-space contact-occlusion proxy.
  let neighbor = textureLoad(depth, min(p + vec2i(3, 3), vec2i(i32(frame.width - 1u), i32(frame.height - 1u))), 0).x;
  let occlusion = 1.0 - smoothstep(0.015, 0.15, abs(neighbor - z));
  // Warm glowing orb drawn directly at the light position.
  let orb = smoothstep(0.05, 0.0, dist) * 1.6;
  let illumination = 0.30 + diffuse * falloff * falloff * 1.6;
  let shaded = src * illumination * mix(0.7, 1.0, occlusion) + vec3f(1.0, 0.92, 0.75) * orb;
  return vec4f(clamp(shaded, vec3f(0.0), vec3f(1.0)), 1.0);
}`;

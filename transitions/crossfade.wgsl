// gl-transitions compatible WGSL compute shader for linear-light crossfade
struct Params {
    progress: f32,
    ratio: f32,
    width: u32,
    height: u32,
    custom: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> from_buf: array<u32>;
@group(0) @binding(2) var<storage, read> to_buf: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_buf: array<u32>;

fn unpack_rgba(p: u32) -> vec4<f32> {
    let r = f32(p & 0xFFu) / 255.0;
    let g = f32((p >> 8u) & 0xFFu) / 255.0;
    let b = f32((p >> 16u) & 0xFFu) / 255.0;
    let a = f32((p >> 24u) & 0xFFu) / 255.0;
    return vec4<f32>(pow(r, 2.2), pow(g, 2.2), pow(b, 2.2), a);
}

fn pack_rgba(c: vec4<f32>) -> u32 {
    let r = u32(clamp(pow(max(c.r, 0.0), 1.0 / 2.2) * 255.0 + 0.5, 0.0, 255.0));
    let g = u32(clamp(pow(max(c.g, 0.0), 1.0 / 2.2) * 255.0 + 0.5, 0.0, 255.0));
    let b = u32(clamp(pow(max(c.b, 0.0), 1.0 / 2.2) * 255.0 + 0.5, 0.0, 255.0));
    let a = u32(clamp(c.a * 255.0 + 0.5, 0.0, 255.0));
    return r | (g << 8u) | (b << 16u) | (a << 24u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = params.width * params.height;
    let idx = gid.x;
    if (idx >= total) {
        return;
    }
    let c0 = unpack_rgba(from_buf[idx]);
    let c1 = unpack_rgba(to_buf[idx]);
    let mixed = mix(c0, c1, params.progress);
    out_buf[idx] = pack_rgba(mixed);
}

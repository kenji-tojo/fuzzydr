// shaders/rasterize_line_bresen.vert
//
// Simple 2-vertex-per-line vertex shader for Bresenham (1-px) line rasterization.
//
// Use with VK_PRIMITIVE_TOPOLOGY_LINE_LIST.  Draw call:
//   vkCmdDraw(cmd, num_lines * 2, 1, 0, 0)
//
// Reads endpoint indices from LinesBuf (binding 3) and vertex attributes from
// VertAttrBuf (binding 4).  Line width is fixed at 1 pixel by the rasterizer,
// so the `radius` field (vattr[off+3]) is not read.
//
// The output varying interface matches rasterize_line_quads.vert, so
// rasterize.frag and opacity_grad.frag are shared between the two:
//
//   v_t       - 0.0 at vertex 0, 1.0 at vertex 1 (Vulkan interpolates in-between)
//   v_s       - always 0.0
//   v_prim_id - original line index (flat, so both endpoints carry the same value)
//
// Vertex interpolation in the fragment/grad shaders:
//   bw0 = 1 - t,  bw1 = t   (same formula as the quad pipeline)

#version 450

layout(set=0, binding=0) uniform UBO {
    mat4  viewproj;
    vec3  campos;
    float tau;
    uint  seed;
    uint  _pad0;
    uint  _pad1;
    uint  _pad2;
} ubo;

layout(set=0, binding=3, std430) readonly buffer LinesBuf {
    uint lines[];            // [num_lines * 2]
} linesBuf;

layout(set=0, binding=4, std430) readonly buffer VertAttrBuf {
    float vattr[];           // [num_verts * 7]  (x, y, z, radius, r, g, b)
} vattrBuf;

layout(push_constant) uniform Push {
    uint  width;
    uint  height;
    uint  prim_type;         // always 1 for bresen lines
    uint  _pad;
} push;

layout(location=0) out vec3  v_rgb;
layout(location=1) out float v_t;       // along-line in [0,1]
layout(location=2) out float v_s;       // always 0.0
layout(location=3) flat out int v_prim_id;

void main() {
    uint lineIdx     = uint(gl_VertexIndex) / 2u;
    uint endpointIdx = uint(gl_VertexIndex) % 2u;

    uint vi  = linesBuf.lines[lineIdx * 2u + endpointIdx];
    uint off = vi * 7u;

    vec3 pos = vec3(vattrBuf.vattr[off + 0u],
                    vattrBuf.vattr[off + 1u],
                    vattrBuf.vattr[off + 2u]);
    // vattr[off + 3] = radius - ignored; Bresenham lines are always 1 px wide.
    vec3 rgb = vec3(vattrBuf.vattr[off + 4u],
                    vattrBuf.vattr[off + 5u],
                    vattrBuf.vattr[off + 6u]);

    gl_Position = ubo.viewproj * vec4(pos, 1.0);
    v_rgb       = rgb;
    v_t         = float(endpointIdx);  // 0.0 or 1.0; rasterizer interpolates
    v_s         = 0.0;
    v_prim_id   = int(lineIdx);
}

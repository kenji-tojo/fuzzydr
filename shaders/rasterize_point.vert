// shaders/rasterize_point.vert
//
// Single-pixel point vertex shader for VK_PRIMITIVE_TOPOLOGY_POINT_LIST.
//
// Draw call:
//   vkCmdDraw(cmd, num_points, 1, 0, 0)
//
// Reads point indices from PointsBuf (binding 12) and vertex attributes from
// VertAttrBuf (binding 4).  The radius field (vattr[off+3]) is not read; each
// point covers one pixel.
//
// The output varying interface matches the line vertex shaders so that
// rasterize.frag and opacity_grad.frag are shared without modification:
//
//   v_t       - always 0.0
//   v_s       - always 0.0
//   v_prim_id - original point index (flat)
//
// Vertex interpolation in the grad shaders:
//   bw0 = 1.0  (single vertex per primitive)

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

layout(set=0, binding=4, std430) readonly buffer VertAttrBuf {
    float vattr[];           // [num_verts * 7]  (x, y, z, radius, r, g, b)
} vattrBuf;

// Binding 12 is the same in the forward and opacity_grad descriptor set
// layouts, so this vertex shader serves both passes.
layout(set=0, binding=12, std430) readonly buffer PointsBuf {
    uint points[];           // [num_points]
} pointsBuf;

layout(push_constant) uniform Push {
    uint  width;
    uint  height;
    uint  prim_type;         // always 3 for points
    uint  _pad;
} push;

layout(location=0) out vec3  v_rgb;
layout(location=1) out float v_t;       // always 0.0
layout(location=2) out float v_s;       // always 0.0
layout(location=3) flat out int v_prim_id;

void main() {
    uint pointIdx = uint(gl_VertexIndex);

    uint vi  = pointsBuf.points[pointIdx];
    uint off = vi * 7u;

    vec3 pos = vec3(vattrBuf.vattr[off + 0u],
                    vattrBuf.vattr[off + 1u],
                    vattrBuf.vattr[off + 2u]);
    // vattr[off + 3] = radius - ignored; points are always 1 px.
    vec3 rgb = vec3(vattrBuf.vattr[off + 4u],
                    vattrBuf.vattr[off + 5u],
                    vattrBuf.vattr[off + 6u]);

    gl_Position = ubo.viewproj * vec4(pos, 1.0);
    gl_PointSize = 1.0;
    v_rgb       = rgb;
    v_t         = 0.0;
    v_s         = 0.0;
    v_prim_id   = int(pointIdx);
}

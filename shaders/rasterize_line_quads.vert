// shaders/rasterize_line_quads.vert
//
// Instanced-quad vertex shader for line rasterization.
//
// Each line primitive is drawn as one instance of a 6-vertex quad
// (two TRIANGLE_LIST triangles).  The shader reads line endpoint
// indices from the SSBO (binding 3) and vertex attributes from the
// SSBO (binding 4), computes screen-space quad corners by projecting
// the world-space cylinder radius to screen pixels, and emits varyings
// (v_rgb, v_t, v_s, v_prim_id) consumed by the fragment shader.
//
// Draw call:  vkCmdDraw(6, num_lines, 0, 0)
// Topology:   VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
// Vertex input state:  EMPTY (no VBO bindings - all data from SSBOs)

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
    uint  prim_type;         // always 2 for quad lines
    uint  _pad;
} push;

// Output varyings consumed by the fragment shader (same interface for
// both rasterize.frag and opacity_grad.frag).
layout(location=0) out vec3  v_rgb;
layout(location=1) out float v_t;       // along-line in [0,1]
layout(location=2) out float v_s;       // across-line in [-1,1]
layout(location=3) flat out int v_prim_id;

void main() {
    uint lineIdx    = gl_InstanceIndex;
    uint vertInQuad = gl_VertexIndex;     // 0..5

    // ---- Read line endpoint indices ----
    uint i0 = linesBuf.lines[lineIdx * 2u + 0u];
    uint i1 = linesBuf.lines[lineIdx * 2u + 1u];

    // ---- Read per-endpoint attributes (7 floats each) ----
    uint off0 = i0 * 7u;
    uint off1 = i1 * 7u;

    vec3  pos0 = vec3(vattrBuf.vattr[off0],      vattrBuf.vattr[off0 + 1u], vattrBuf.vattr[off0 + 2u]);
    float rad0 =      vattrBuf.vattr[off0 + 3u];
    vec3  rgb0 = vec3(vattrBuf.vattr[off0 + 4u],  vattrBuf.vattr[off0 + 5u], vattrBuf.vattr[off0 + 6u]);

    vec3  pos1 = vec3(vattrBuf.vattr[off1],      vattrBuf.vattr[off1 + 1u], vattrBuf.vattr[off1 + 2u]);
    float rad1 =      vattrBuf.vattr[off1 + 3u];
    vec3  rgb1 = vec3(vattrBuf.vattr[off1 + 4u],  vattrBuf.vattr[off1 + 5u], vattrBuf.vattr[off1 + 6u]);

    // ---- Transform endpoints to clip space ----
    vec4 clip0 = ubo.viewproj * vec4(pos0, 1.0);
    vec4 clip1 = ubo.viewproj * vec4(pos1, 1.0);

    // ---- Screen-space positions ----
    vec2 ndc0 = clip0.xy / clip0.w;
    vec2 ndc1 = clip1.xy / clip1.w;

    vec2 screenSize = vec2(float(push.width), float(push.height));
    vec2 screen0 = (ndc0 * 0.5 + 0.5) * screenSize;
    vec2 screen1 = (ndc1 * 0.5 + 0.5) * screenSize;

    // ---- Screen-space direction and perpendicular ----
    vec2 dir = screen1 - screen0;
    float len = length(dir);
    vec2 tangent = (len > 1e-6) ? (dir / len) : vec2(1.0, 0.0);
    vec2 normal  = vec2(-tangent.y, tangent.x);   // perpendicular

    // ---- Project world-space cylinder radius to screen-space pixels ----
    //
    // For each endpoint, compute a world-space vector perpendicular to both
    // the line tangent and the camera-to-endpoint direction, then project
    // (pos + right * radius) to screen space and measure the pixel distance.
    // This correctly handles perspective foreshortening.
    //
    vec3 lineDir3 = pos1 - pos0;
    float lineLen3 = length(lineDir3);
    vec3 lineTan3 = (lineLen3 > 1e-12) ? (lineDir3 / lineLen3) : vec3(1.0, 0.0, 0.0);

    // Endpoint 0
    vec3 toCamera0 = ubo.campos - pos0;
    vec3 right0 = cross(lineTan3, toCamera0);
    float rightLen0 = length(right0);
    right0 = (rightLen0 > 1e-12) ? (right0 / rightLen0) : vec3(0.0, 1.0, 0.0);

    vec4 clipOff0 = ubo.viewproj * vec4(pos0 + right0 * rad0, 1.0);
    vec2 screenOff0 = (clipOff0.xy / clipOff0.w * 0.5 + 0.5) * screenSize;
    float screenRad0 = length(screenOff0 - screen0);

    // Endpoint 1
    vec3 toCamera1 = ubo.campos - pos1;
    vec3 right1 = cross(lineTan3, toCamera1);
    float rightLen1 = length(right1);
    right1 = (rightLen1 > 1e-12) ? (right1 / rightLen1) : vec3(0.0, 1.0, 0.0);

    vec4 clipOff1 = ubo.viewproj * vec4(pos1 + right1 * rad1, 1.0);
    vec2 screenOff1 = (clipOff1.xy / clipOff1.w * 0.5 + 0.5) * screenSize;
    float screenRad1 = length(screenOff1 - screen1);

    // ---- Decode (t, s) from the 6 quad vertices ----
    //
    //   Quad layout (two CCW triangles):
    //
    //      s=-1  s=+1
    //   t=0  A-----B        tri 0 = A B C  (verts 0,1,2)
    //        |   / |        tri 1 = A C D  (verts 3,4,5)
    //   t=1  D-----C
    //
    float t, s;
    switch (vertInQuad) {
        case 0u: t = 0.0; s = -1.0; break;
        case 1u: t = 0.0; s =  1.0; break;
        case 2u: t = 1.0; s =  1.0; break;
        case 3u: t = 0.0; s = -1.0; break;
        case 4u: t = 1.0; s =  1.0; break;
        default: t = 1.0; s = -1.0; break;   // vert 5
    }

    // ---- Compute screen-space position of this quad corner ----
    float screenRadius = mix(screenRad0, screenRad1, t);
    vec2  offset = normal * (s * screenRadius);
    vec2  screenPt = mix(screen0, screen1, t) + offset;

    // ---- Back to clip space (preserves perspective-correct interpolation) ----
    vec2  ndcPt = (screenPt / screenSize) * 2.0 - 1.0;
    float wPt   = mix(clip0.w, clip1.w, t);
    float zPt   = mix(clip0.z, clip1.z, t);

    gl_Position = vec4(ndcPt * wPt, zPt, wPt);

    // ---- Varyings ----
    v_rgb     = mix(rgb0, rgb1, t);
    v_t       = t;
    v_s       = s;
    v_prim_id = int(lineIdx);
}

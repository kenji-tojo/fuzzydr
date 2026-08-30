// shaders/viewer_line_quads.vert
// Instanced-quad vertex shader for thick line rasterization.
//
// Adapted from fuzzydr's rasterize_line_quads.vert with the backward-pass
// machinery (opacity, tau, seed, bary output attachments) stripped out.
//
// Each line is drawn as one instance of 6 vertices (two CCW triangles).
// All data comes from SSBOs; the pipeline has no vertex input bindings.
//
// Draw call:  vkCmdDraw(cmd, 6, num_lines, 0, 0)
// Topology:   VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
//
// Quad corner layout (t  in  {0,1} along line, s  in  {-1,+1} across line):
//
//      s=-1  s=+1
//   t=0  A-----B        tri 0 = A B C  (verts 0,1,2)
//        |   / |        tri 1 = A C D  (verts 3,4,5)
//   t=1  D-----C
//
// Screen-space half-width is derived by projecting (pos + right*radius) into
// screen pixels, giving perspective-correct thick lines.
// A 0.5-pixel minimum guarantees that radius=0 produces a 1-pixel hairline
// without requiring a separate Bresenham pass.

#version 450

layout(push_constant) uniform PC {
    mat4  viewproj;  // offset  0
    vec3  campos;    // offset 64  - camera world position for radius projection
    uint  width;     // offset 76
    uint  height;    // offset 80
} pc;

// set=0, binding=0 : vertex attributes  [num_verts * 7]  (x,y,z,radius,r,g,b)
layout(set=0, binding=0, std430) readonly buffer VertAttrBuf {
    float vattr[];
} vattrBuf;

// set=0, binding=1 : line endpoint index pairs  [num_lines * 2]
layout(set=0, binding=1, std430) readonly buffer LinesBuf {
    uint lines[];
} linesBuf;

layout(location=0) out vec3 fragColor;

void main() {
    uint lineIdx    = uint(gl_InstanceIndex);
    uint vertInQuad = uint(gl_VertexIndex);   // 0..5

    // ---- Endpoint indices ----
    uint i0 = linesBuf.lines[lineIdx * 2u + 0u];
    uint i1 = linesBuf.lines[lineIdx * 2u + 1u];

    // ---- Per-endpoint attributes (7 floats each) ----
    uint off0 = i0 * 7u;
    uint off1 = i1 * 7u;

    vec3  pos0 = vec3(vattrBuf.vattr[off0],     vattrBuf.vattr[off0+1u], vattrBuf.vattr[off0+2u]);
    float rad0 =      vattrBuf.vattr[off0+3u];
    vec3  rgb0 = vec3(vattrBuf.vattr[off0+4u],  vattrBuf.vattr[off0+5u], vattrBuf.vattr[off0+6u]);

    vec3  pos1 = vec3(vattrBuf.vattr[off1],     vattrBuf.vattr[off1+1u], vattrBuf.vattr[off1+2u]);
    float rad1 =      vattrBuf.vattr[off1+3u];
    vec3  rgb1 = vec3(vattrBuf.vattr[off1+4u],  vattrBuf.vattr[off1+5u], vattrBuf.vattr[off1+6u]);

    // ---- Clip and screen-space endpoints ----
    vec4 clip0 = pc.viewproj * vec4(pos0, 1.0);
    vec4 clip1 = pc.viewproj * vec4(pos1, 1.0);

    vec2 screenSize = vec2(float(pc.width), float(pc.height));
    vec2 screen0 = (clip0.xy / clip0.w * 0.5 + 0.5) * screenSize;
    vec2 screen1 = (clip1.xy / clip1.w * 0.5 + 0.5) * screenSize;

    // ---- Screen-space tangent and its perpendicular (the quad's normal axis) ----
    vec2 dir     = screen1 - screen0;
    float dlen   = length(dir);
    vec2 tangent = (dlen > 1e-6) ? (dir / dlen) : vec2(1.0, 0.0);
    vec2 normal  = vec2(-tangent.y, tangent.x);

    // ---- Perspective-correct radius projection ----
    // For each endpoint compute a world-space vector perpendicular to both the
    // line tangent and the camera direction, project (pos + right*radius) to
    // screen space, and measure the pixel distance.
    vec3 lineDir = pos1 - pos0;
    float llen   = length(lineDir);
    vec3 lineTan = (llen > 1e-12) ? (lineDir / llen) : vec3(1.0, 0.0, 0.0);

    // endpoint 0
    vec3 right0 = cross(lineTan, pc.campos - pos0);
    float rlen0 = length(right0);
    right0 = (rlen0 > 1e-12) ? (right0 / rlen0) : vec3(0.0, 1.0, 0.0);
    vec4 cr0 = pc.viewproj * vec4(pos0 + right0 * rad0, 1.0);
    float screenRad0 = length((cr0.xy / cr0.w * 0.5 + 0.5) * screenSize - screen0);

    // endpoint 1
    vec3 right1 = cross(lineTan, pc.campos - pos1);
    float rlen1 = length(right1);
    right1 = (rlen1 > 1e-12) ? (right1 / rlen1) : vec3(0.0, 1.0, 0.0);
    vec4 cr1 = pc.viewproj * vec4(pos1 + right1 * rad1, 1.0);
    float screenRad1 = length((cr1.xy / cr1.w * 0.5 + 0.5) * screenSize - screen1);

    // ---- Quad corner from vertex index ----
    float t, s;
    switch (vertInQuad) {
        case 0u: t = 0.0; s = -1.0; break;
        case 1u: t = 0.0; s =  1.0; break;
        case 2u: t = 1.0; s =  1.0; break;
        case 3u: t = 0.0; s = -1.0; break;
        case 4u: t = 1.0; s =  1.0; break;
        default: t = 1.0; s = -1.0; break;   // vertex 5
    }

    // 0.5 px minimum so that radius=0 still produces a visible hairline
    float screenRadius = max(0.5, mix(screenRad0, screenRad1, t));
    vec2  screenPt     = mix(screen0, screen1, t) + normal * (s * screenRadius);

    // ---- Back to clip space (preserves perspective-correct depth) ----
    vec2  ndcPt = (screenPt / screenSize) * 2.0 - 1.0;
    float wPt   = mix(clip0.w, clip1.w, t);
    float zPt   = mix(clip0.z, clip1.z, t);
    gl_Position = vec4(ndcPt * wPt, zPt, wPt);

    fragColor = mix(rgb0, rgb1, t);
}

// shaders/viewer_gauss_resolve.vert
// Fullscreen triangle with no vertex input - covers the clip-space quad with a
// single oversized tri, the standard trick that avoids a VBO and produces zero
// overdraw at the screen edges.

#version 450

void main() {
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}

// shaders/rasterize.vert
//
// Vertex shader for the face (triangle) pipeline.
// Transforms world-space vertices to clip space and forwards per-vertex
// color to the fragment shader.
//
// The line pipeline uses separate vertex shaders that read data from SSBOs:
// rasterize_line_quads.vert (instanced quads) and rasterize_line_bresen.vert
// (1-px lines).

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

// Packed attrs from C++:
//   location 0: vec4 (x,y,z,radius)   - world-space position + per-vertex radius
//   location 1: vec3 (r,g,b)
layout(location=0) in vec4 in_pos;
layout(location=1) in vec3 in_rgb;

// To fragment shader (must match line pipeline's output interface):
layout(location=0) out vec3  v_rgb;
layout(location=1) out float v_t;          // zeroed; only meaningful in line pipeline
layout(location=2) out float v_s;          // zeroed; only meaningful in line pipeline
layout(location=3) flat out int v_prim_id; // zeroed; only meaningful in line pipeline

void main() {
    gl_Position = ubo.viewproj * vec4(in_pos.xyz, 1.0);
    v_rgb       = in_rgb;
    v_t         = 0.0;
    v_s         = 0.0;
    v_prim_id   = 0;
}

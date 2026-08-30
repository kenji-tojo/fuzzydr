// shaders/viewer.vert
// Face (triangle) + Bresenham line vertex shader.
//
// Push constant block is 84 bytes, matching viewer_line_quads.vert, so both
// pipelines can share one VkPipelineLayout.  This shader reads only viewproj
// (bytes 0-63); campos/width/height are declared for layout compatibility only.
//
// Vertex layout: stride 28 bytes (7 floats per vertex, same as fuzzydr)
//   offset  0 : vec3  position
//   offset 12 : float radius  (unused here - Bresenham ignores it)
//   offset 16 : vec3  color

#version 450

layout(push_constant) uniform PC {
    mat4  viewproj;  // offset  0
    vec3  campos;    // offset 64  (not used here)
    uint  width;     // offset 76  (not used here)
    uint  height;    // offset 80  (not used here)
} pc;

layout(location=0) in vec3 inPos;
layout(location=1) in vec3 inColor;   // VBO attribute at byte offset 16

layout(location=0) out vec3 fragColor;

void main() {
    gl_Position = pc.viewproj * vec4(inPos, 1.0);
    fragColor   = inColor;
}

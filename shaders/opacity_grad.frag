// shaders/opacity_grad.frag
#version 450

#extension GL_EXT_fragment_shader_barycentric : require
#extension GL_EXT_shader_atomic_float : require

layout(push_constant) uniform Push {
    uint  width;
    uint  height;
    // Primitive class (matches src/rasterize.cpp):
    //   0 = face, 1 = bresen line, 2 = quad line, 3 = point.
    uint  prim_type;
    float eps;           // e.g. 1e-6 (for opacity denom clamp)
} pc;

layout(set=0, binding=0) uniform UBO {
    mat4  viewproj;
    vec3  campos;
    float tau;
    uint  seed;
    uint  _pad0;
    uint  _pad1;
    uint  _pad2;
} ubo;

// Forward outputs / inputs for backward (all SSBO)
layout(set=0, binding=1, std430) readonly buffer PrimIdBuf {
    uint prim_id[];             // [H*W], background = 0xFFFFFFFF
} primIdBuf;

layout(set=0, binding=2, std430) readonly buffer BaryDepthBuf {
    float bary_depth[];         // [H*W*4] -> (w0, w1, depth, prim_type)
} bdBuf;

// Bindings 3 (lines) and 4 (vert_attrs) are read by the line and point vertex
// shaders and are omitted here.

layout(set=0, binding=5, std430) readonly buffer FaceOpacityBuf {
    float face_opacity[];       // [M]
} fopBuf;

layout(set=0, binding=6, std430) buffer GradFaceOpacityBuf {
    float grad_face_opacity[];  // [M]
} gfopBuf;

layout(set=0, binding=7, std430) readonly buffer ErrorBuf {
    float err[];                // [H*W] per-pixel scalar error
} errBuf;

layout(set=0, binding=8, std430) readonly buffer LineOpacityBuf {
    float line_opacity[];       // [L]
} lopBuf;

layout(set=0, binding=9, std430) buffer GradLineOpacityBuf {
    float grad_line_opacity[];  // [L]
} glopBuf;

layout(set=0, binding=10, std430) readonly buffer PointOpacityBuf {
    float point_opacity[];       // [P]
} popBuf;

layout(set=0, binding=11, std430) buffer GradPointOpacityBuf {
    float grad_point_opacity[];  // [P]
} gpopBuf;

// --- Varying from the line vertex shader (line pipeline only) ---
layout(location=3) flat in int v_prim_id;

// Dummy attachment to keep a normal graphics render pass happy
layout(location=0) out uint outDummy;

void main() {
    uint px = uint(gl_FragCoord.x);
    uint py = uint(gl_FragCoord.y);
    if (px >= pc.width || py >= pc.height) {
        outDummy = 0u;
        return;
    }

    uint idx = py * pc.width + px;

    uint winner = primIdBuf.prim_id[idx];
    bool has_winner = (winner != 0xFFFFFFFFu);

    // Reference depth: winner depth if hit, else background=1.0
    float winner_depth = has_winner ? bdBuf.bary_depth[idx * 4u + 2u] : 1.0;

    float e = errBuf.err[idx];

    // --- Resolve primitive ID ---
    // Face: gl_PrimitiveID is the hardware face index.
    // Line: v_prim_id carries the original input line index from the line VS.
    uint prim;
    if (pc.prim_type == 0u) {
        prim = uint(gl_PrimitiveID);
    } else {
        prim = uint(v_prim_id);
    }

    // Look up per-primitive opacity.
    float op;
    if (pc.prim_type == 0u) {
        op = fopBuf.face_opacity[prim];
    } else if (pc.prim_type == 3u) {
        op = popBuf.point_opacity[prim];
    } else {
        // Lines: prim_type in {1, 2}  (1 = bresen, 2 = quads).
        op = lopBuf.line_opacity[prim];
    }

    float grad_op = 0.0;
    float weight = 1.0 / float(pc.width * pc.height);

    if (has_winner && prim == winner) {
        // Check that winner's prim_type matches this draw's prim_type.
        float stored_prim_type = bdBuf.bary_depth[idx * 4u + 3u];
        if (uint(stored_prim_type) != pc.prim_type) {
            // Primitive IDs are per-class, so this is a different primitive
            // that happens to share an ID with the winner.
            outDummy = 0u;
            return;
        }
        // winner: + e / op
        grad_op = weight * e / max(op, pc.eps);
    } else if (gl_FragCoord.z < winner_depth) {
        // occluder in front of reference: - e / (1 - op)
        grad_op = -weight * e / max(1.0 - op, pc.eps);
    } else {
        outDummy = 0u;
        return;
    }

    // Accumulate gradient into the appropriate per-primitive buffer.
    if (pc.prim_type == 0u) {
        atomicAdd(gfopBuf.grad_face_opacity[prim], grad_op);
    } else if (pc.prim_type == 3u) {
        atomicAdd(gpopBuf.grad_point_opacity[prim], grad_op);
    } else {
        // Lines: prim_type in {1, 2}  (1 = bresen, 2 = quads).
        atomicAdd(glopBuf.grad_line_opacity[prim], grad_op);
    }

    outDummy = 0u;
}

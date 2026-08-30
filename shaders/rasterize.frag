// shaders/rasterize.frag
#version 450
#extension GL_EXT_fragment_shader_barycentric : require

layout(set=0, binding=0) uniform UBO {
    mat4  viewproj;
    vec3  campos;
    float tau;
    uint  seed;
    uint  _pad0;
    uint  _pad1;
    uint  _pad2;
} ubo;

layout(set=0, binding=1, std430) readonly buffer FaceOpacity {
    float fop[];
} faceOpacity;

layout(set=0, binding=2, std430) readonly buffer LineOpacity {
    float lop[];
} lineOpacity;

// Bindings 3 (lines), 4 (vert_attrs), and 12 (points) are read by the line and
// point vertex shaders and are omitted here.

layout(set=0, binding=5, std430) readonly buffer PointOpacity {
    float pop[];
} pointOpacity;

layout(push_constant) uniform Push {
    uint  width;
    uint  height;
    // Primitive class (matches src/rasterize.cpp):
    //   0 = face, 1 = bresen line, 2 = quad line, 3 = point.
    uint  prim_type;
    // Face primitive IDs are offset by this to form the RNG counter, so that
    // faces and lines do not share counter values.  Points cannot be mixed
    // with faces or lines, so point IDs need no offset.
    uint  num_lines;
} push;

// --- Varyings from vertex shader ---
layout(location=0) in vec3  v_rgb;
layout(location=1) in float v_t;          // along-line in [0,1]  (line pipeline)
layout(location=2) in float v_s;          // across-line in [-1,1] (line pipeline)
layout(location=3) flat in int v_prim_id; // original line index  (line pipeline)

layout(location=0) out uint outPrimId;
layout(location=1) out vec4 outBaryDepth;
layout(location=2) out vec4 outRGBA;

// ---- Philox 2x32-10 (uses umulExtended) ----
uvec2 philox_round(uvec2 ctr, uint key) {
    uint hi, lo;
    umulExtended(0xD2511F53u, ctr.x, hi, lo);
    return uvec2(hi ^ key ^ ctr.y, lo);
}

uint philox_bumpkey(uint key) {
    return key + 0x9E3779B9u;
}

uvec2 philox2x32_10(uvec2 ctr, uint key) {
    for (int i = 0; i < 10; ++i) {
        ctr = philox_round(ctr, key);
        key = philox_bumpkey(key);
    }
    return ctr;
}

float rng01(uint prim, uint seed) {
    uvec2 r = philox2x32_10(uvec2(prim, seed), seed);
    return float(r.x >> 8) * (1.0 / 16777216.0); // 2^24: maps 24-bit mantissa to [0, 1)
}

void main() {
    // --- Resolve primitive ID ---
    // For faces: gl_PrimitiveID is the hardware face index.
    // For lines: v_prim_id carries the original input line index
    //   from the line vertex shader, because gl_PrimitiveID would
    //   reflect the instanced-quad triangle index instead.
    uint primId;
    if (push.prim_type == 0u) {
        primId = uint(gl_PrimitiveID);
    } else {
        primId = uint(v_prim_id);
    }

    outPrimId = primId;

    // Look up per-primitive opacity.
    float op;
    if (push.prim_type == 0u) {
        op = faceOpacity.fop[primId];
    } else if (push.prim_type == 3u) {
        op = pointOpacity.pop[primId];
    } else {
        // Lines: prim_type in {1, 2}  (1 = bresen, 2 = quads).
        op = lineOpacity.lop[primId];
    }

    float tau_val = ubo.tau;
    if (tau_val == -1.0) {
        // Primitive IDs are numbered per class, so face IDs are offset by
        // num_lines here.  The offset applies to the RNG counter only; primId
        // is used unchanged elsewhere, including as the opacity index.
        uint rng_id = (push.prim_type == 0u) ? primId + push.num_lines : primId;
        tau_val = rng01(rng_id, ubo.seed);
    }

    if (op < tau_val) {
        discard;
    }

    if (push.prim_type == 0u) {
        // Face: use hardware barycentrics
        vec3 bary = gl_BaryCoordEXT;
        outBaryDepth = vec4(bary.x, bary.y, gl_FragCoord.z, 0.0);
    } else if (push.prim_type == 3u) {
        // Point: single vertex, no interpolation params
        outBaryDepth = vec4(0.0, 0.0, gl_FragCoord.z, 3.0);
    } else {
        // Line (1 = bresen, 2 = quads): (s, t, depth, prim_type)
        outBaryDepth = vec4(v_s, v_t, gl_FragCoord.z, float(push.prim_type));
    }

    // RGB = interpolated vertex color.  Alpha is padded to 1 and ignored by
    // the MSAA filter, which reads 3 channels; per-primitive opacity acts
    // through the `discard` gate above.
    outRGBA = vec4(v_rgb, 1.0);
}

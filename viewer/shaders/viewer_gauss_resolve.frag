// shaders/viewer_gauss_resolve.frag
// Isotropic Gaussian pixel reconstruction filter for the gauss_msaa mode.
//
// The scene is rendered at 2x native resolution into a 1-sample fp16 RGBA
// colour target.  This pass downsamples to native resolution using a 6x6
// sigma=0.5-output-px isotropic Gaussian kernel, evaluated as nine hardware
// bilinear samples.
//
// Bilinear-tap reduction (single pass): the kernel covers a 6x6 block of
// hi-res texels arranged as a 3x3 grid of 2x2 clusters around the
// output-pixel centre.  Because the 2D Gaussian is separable
// (w_ij = wx_i * wy_j), each cluster's weighted sum can be read with one
// bilinear sample placed at the sub-pixel offset whose interpolation
// weights match the Gaussian weights inside that cluster:
//
//     bilinear(alpha, beta) = (1-alpha)*(1-beta)*t_00 + alpha*(1-beta)*t_10
//                    + (1-alpha)*beta*t_01    + alpha*beta*t_11
//                    = (sum_ij w_ij * t_ij) / (Sx * Sy)         (separable kernel)
//
// Unlike the 4x4 variant the middle row/column cluster carries more weight
// than the outer ones (the kernel is no longer flat across clusters), so the
// nine bilinear samples are combined with explicit per-cluster weights.
// Borders rely on the sampler's CLAMP_TO_EDGE address mode.
//
// Reference: Daniel Rakos, "Efficient Gaussian Blur with Linear Sampling"
// (rastergrid.com, 2010) - same trick applied to 1D blur.

#version 450

layout(set=0, binding=0) uniform sampler2D uHiRes;

layout(location=0) out vec4 outColor;

// -- Tap offset (outer clusters) -----------------------------------------
// sigma = 0.5 output-px => sigma = 1.0 hi-res-px.  The outer cluster on one side
// holds texels at hi-res offsets {2.5, 1.5} from the output centre, with
// weights w(2.5) = exp(-3.125), w(1.5) = exp(-1.125).  The bilinear tap is
// placed at offset 2.5 - alpha where
//     alpha = w(1.5) / (w(1.5) + w(2.5)) = 1 / (1 + exp(-2)) = e^2 / (e^2 + 1).
// Both axes use the same magnitude by symmetry.  Middle cluster is at
// offset 0 (equal weights on +/-0.5).
const float OFS_HIRES = 1.6192029;   // 2.5 - e^2 / (e^2 + 1)

// -- Cluster combination weights -----------------------------------------
// 1D normalized cluster sums:
//   r_o = (exp(-3.125) + exp(-1.125)) / S    ~= 0.147308   (outer, each side)
//   r_m = (2*exp(-0.125)) / S                ~= 0.705385   (middle)
//   S   = 2*(exp(-3.125)+exp(-1.125)) + 2*exp(-0.125)
// 2D weights factor as r_i * r_j; the nine values sum to 1 by construction.
const float W_CORNER = 0.0216997;    // r_o^2            (x4 corner taps)
const float W_EDGE   = 0.1039066;    // r_o * r_m       (x4 edge taps)
const float W_CENTER = 0.4975856;    // r_m^2            (x1 center tap)

void main() {
    vec2  hi_size = vec2(textureSize(uHiRes, 0));
    ivec2 out_p   = ivec2(gl_FragCoord.xy);
    // Output-pixel centre in hi-res pixel coordinates.
    vec2  cen_hi  = vec2(2 * out_p) + vec2(1.0);
    vec2  base_uv = cen_hi / hi_size;
    vec2  ofs_uv  = vec2(OFS_HIRES) / hi_size;

    vec3 s_NW = textureLod(uHiRes, base_uv + vec2(-ofs_uv.x, -ofs_uv.y), 0.0).rgb;
    vec3 s_N  = textureLod(uHiRes, base_uv + vec2( 0.0,       -ofs_uv.y), 0.0).rgb;
    vec3 s_NE = textureLod(uHiRes, base_uv + vec2(+ofs_uv.x, -ofs_uv.y), 0.0).rgb;
    vec3 s_W  = textureLod(uHiRes, base_uv + vec2(-ofs_uv.x,  0.0      ), 0.0).rgb;
    vec3 s_C  = textureLod(uHiRes, base_uv,                              0.0).rgb;
    vec3 s_E  = textureLod(uHiRes, base_uv + vec2(+ofs_uv.x,  0.0      ), 0.0).rgb;
    vec3 s_SW = textureLod(uHiRes, base_uv + vec2(-ofs_uv.x, +ofs_uv.y), 0.0).rgb;
    vec3 s_S  = textureLod(uHiRes, base_uv + vec2( 0.0,       +ofs_uv.y), 0.0).rgb;
    vec3 s_SE = textureLod(uHiRes, base_uv + vec2(+ofs_uv.x, +ofs_uv.y), 0.0).rgb;

    vec3 rgb = W_CORNER * (s_NW + s_NE + s_SW + s_SE)
             + W_EDGE   * (s_N  + s_W  + s_E  + s_S )
             + W_CENTER *  s_C;

    outColor = vec4(rgb, 1.0);
}

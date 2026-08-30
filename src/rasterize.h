// src/rasterize.h
#pragma once
#include <cstdint>
#include <memory>
#include <vulkan/vulkan.h>

#include "vk_common.h"

namespace fuzzydr {

// Forward declarations for internal cache structs (defined in rasterize.cpp).
struct ForwardCache;
struct EdgeGradCache;
struct OpacityGradCache;
struct ShaderCache;

// ---------------------------------------------------------------------------
// render_options - a compact uint8[NUM_RENDER_OPTIONS] array of boolean
// rendering flags (always CPU).
//
// Indices (also mirrored as _OPT_* constants in fuzzydr/__init__.py):
static constexpr uint32_t OPT_WHITE_BG         = 0;  // clear rgba to white; else black
static constexpr uint32_t OPT_BRESEN_LINES     = 1;  // 1-px Bresenham LINE_LIST (radius slot unused)
static constexpr uint32_t OPT_CULL_BACKFACE    = 2;  // cull CW faces (keep CCW); lines unaffected
static constexpr uint32_t OPT_AUX_BUFFERS      = 3;  // download prim_id + bary_depth to the caller's buffers
static constexpr uint32_t NUM_RENDER_OPTIONS   = 4;  // number of flags above
// ---------------------------------------------------------------------------

/// Rasterizer - renders triangles, lines, and points into offscreen buffers
/// and provides backward-pass kernels for edge-based RGB and stochastic
/// opacity gradients.  Owns all GPU buffer caches; destroying the object
/// releases them.
///
/// At least one of num_faces, num_lines, or num_points must be > 0.
/// When a primitive class is empty (count == 0), its pointer arguments
/// (faces/face_opacity, lines/line_opacity, or points/point_opacity) may be null.
///
/// Vertex attributes are packed as 7 floats per vertex:
///   vert_attrs[v] = (x, y, z, radius, r, g, b)
/// where (x,y,z) are world-space position, (r,g,b) are vertex color, and the
/// `radius` slot is used only by quad lines (i.e. when the OPT_BRESEN_LINES
/// flag is unset) as a world-space cylinder radius that drives the
/// screen-space line width; it is ignored for Bresenham lines, faces, and
/// points.
/// The world-to-clip transform is performed inside the vertex shader using
/// the viewproj matrix passed as a uniform.
/// Opacity is per-primitive (per-face, per-line, per-point).  It acts through
/// the `discard` gate in the fragment shader and is not written to the RGBA
/// output; the alpha channel is padded to 1.0 and ignored by the MSAA filter,
/// which reads RGB only.
class Rasterizer {
public:
  Rasterizer();
  ~Rasterizer();

  Rasterizer(const Rasterizer&) = delete;
  Rasterizer& operator=(const Rasterizer&) = delete;

  /// Forward rasterization.
  ///
  /// Outputs:
  /// - prim_id_out : uint32[height, width]
  ///     Primitive ID per pixel (gl_PrimitiveID). Background = 0xFFFFFFFF.
  ///     Face and line primitive IDs live in separate namespaces;
  ///     use bary_depth_out.w (prim_type) to disambiguate.
  /// - bary_depth_out : float32[height, width, 4]
  ///     (w0, w1, depth, prim_type) per pixel, where
  ///       For faces (prim_type == 0): w0, w1 are barycentric coords (w2 = 1 - w0 - w1)
  ///       For lines (prim_type in {1, 2}):
  ///           w0 = s (across-line in [-1,1] for quads, 0 for Bresenham),
  ///           w1 = t (along-line in [0,1])
  ///           Vertex interpolation weights: v0 weight = 1-t, v1 weight = t.
  ///       For points (prim_type == 3): w0 = w1 = 0.
  ///       depth      : gl_FragCoord.z in [0, 1]
  ///       prim_type  : 0 = face, 1 = bresen line, 2 = quad line, 3 = point.
  /// - rgba_out : float32[height, width, 4]
  ///     Forward-shaded/interpolated RGBA per pixel.  Alpha is padded to 1.0
  ///     and ignored downstream (the MSAA filter reads RGB only).
  ///
  /// prim_id_out and bary_depth_out may be null to skip their host/CUDA
  /// download; the GPU-side buffers are still written (needed by backward).
  int rasterize(
      uint32_t        num_verts,
      const float*    vert_attrs,     // [num_verts, 7] packed (x,y,z,radius,r,g,b)
      uint32_t        num_faces,
      const uint32_t* faces,          // [num_faces, 3]  (null if num_faces==0, or to reuse cached topology)
      const float*    face_opacity,   // [num_faces]     (non-null when num_faces>0)
      uint32_t        num_lines,
      const uint32_t* lines,          // [num_lines, 2]  (null if num_lines==0, or to reuse cached topology)
      const float*    line_opacity,   // [num_lines]     (non-null when num_lines>0)
      uint32_t        num_points,
      const uint32_t* points,         // [num_points]    (null if num_points==0, or to reuse cached topology)
      const float*    point_opacity,  // [num_points]    (non-null when num_points>0)
      uint32_t        width,
      uint32_t        height,
      const float*    viewproj,       // [4,4] row-major, CPU; converted to column-major internally
      const float*    campos,         // [3] CPU; used by the quad-line VS for radius projection (unused for bresen)
      float           tau,            // opacity threshold in [0,1], or -1 for stochastic masking
      uint32_t        seed,           // RNG seed (used only when tau == -1)
      uint32_t*       prim_id_out,    // [height, width]    (null to skip download; see Outputs)
      float*          bary_depth_out, // [height, width, 4] (null to skip download; see Outputs)
      float*          rgba_out,       // [height, width, 4]
      const uint8_t*  render_options, // [NUM_RENDER_OPTIONS] CPU; see OPT_* flags above
      bool            is_cuda = false); // if true, buffer pointers are CUDA device ptrs (viewproj/campos/render_options stay CPU)

  /// Backprop for edge-based RGB loss using cached forward results.
  /// Produces gradients into packed vertex attributes (positions + RGB).
  /// The compute shader reads prim_type from bary_depth.w to branch between
  /// face (3-vertex barycentric) and line (2-vertex weighted) gradient scatter.
  /// The viewproj matrix from the forward cache is reused to transform
  /// clip-space gradients back to world-space vertex gradients.
  int edge_grad(
      uint32_t     num_verts,
      uint32_t     num_faces,
      uint32_t     num_lines,
      uint32_t     num_points,
      uint32_t     width,
      uint32_t     height,
      const float* grad_out_rgba,    // [height, width, 4] upstream RGBA gradient
      float*       grad_vert_attrs,  // [num_verts, 7] output vertex-attr gradient
      bool         is_cuda = false); // if true, all pointers are CUDA device ptrs

  /// Backward pass for stochastic opacity masking (likelihood-ratio gradient).
  /// Writes gradients into grad_face_opacity / grad_line_opacity / grad_point_opacity.
  /// The face pass mirrors the forward backface-cull mode (per render_options).
  int opacity_grad(
      uint32_t     num_faces,
      uint32_t     num_lines,
      uint32_t     num_points,
      uint32_t     width,
      uint32_t     height,
      const float* error,               // [height, width]
      float*       grad_face_opacity,   // [num_faces]  (null when num_faces==0)
      float*       grad_line_opacity,   // [num_lines]  (null when num_lines==0)
      float*       grad_point_opacity,  // [num_points] (null when num_points==0)
      float        eps,                 // likelihood-ratio denominator epsilon
      bool         is_cuda = false);    // if true, all pointers are CUDA device ptrs

private:
  std::unique_ptr<ForwardCache>      fwd_;
  std::unique_ptr<EdgeGradCache>     edge_;
  std::unique_ptr<OpacityGradCache>  opac_;
  std::unique_ptr<ShaderCache>       shaders_;
};

} // namespace fuzzydr

// src/viewer.h
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace fuzzydr_viewer {

/// Packed scene data passed from Python.
///
/// vert_attrs : float[num_verts * 7]  ->  x, y, z, radius, r, g, b
///              Matches fuzzydr's vertex attribute layout exactly.
///              radius is a world-space half-width, projected to screen
///              pixels by the quad-line style; Bresenham and face draws
///              ignore the slot.  The viewer's constant-width style rewrites
///              this column in place, so the shader always reads it.
///              When sh_coeffs is non-empty the rgb columns are written by
///              the viewer's SH evaluation compute pass and may be left zero.
///
/// sh_coeffs  : float[num_verts * sh_cols]
///              Optional degree-3 SH coefficients, [N,3,16] flattened, so
///              sh_cols is 48.  When non-empty the viewer runs a compute pass
///              each frame before drawing.  The buffer is transposed to
///              [sh_cols, N] on upload for coalesced GPU reads.
///
/// faces      : uint32[num_faces * 3]  -> vertex indices (triangle list, CCW)
/// lines      : uint32[num_lines * 2]  -> vertex index pairs (line list)
///
/// line_strip : uint32[num_strip_indices]  -> the same segments rebaked as
///              LINE_STRIP runs separated by the primitive-restart sentinel
///              UINT32_MAX.  The Bresenham style draws from this, which costs
///              roughly half the vertex-shader invocations of the pair list.
///              The quad styles index `lines` by instance and cannot read a
///              strip, so both layouts are uploaded and each style binds its
///              own.  Leave it empty to draw Bresenham from `lines` instead.
///
/// At least one of num_faces / num_lines must be > 0.
/// When a primitive class is absent (count == 0) its vector may be empty.
struct SceneData {
    std::vector<float>    vert_attrs;
    uint32_t              num_verts = 0;

    std::vector<float>    sh_coeffs;
    uint32_t              sh_cols = 0;

    std::vector<uint32_t> faces;
    uint32_t              num_faces = 0;

    std::vector<uint32_t> lines;
    uint32_t              num_lines = 0;

    std::vector<uint32_t> line_strip;
    uint32_t              num_strip_indices = 0;

    /// True when vert_attrs carries genuine per-vertex radii (checkpoints
    /// saved with bresen_lines=False).  When false the viewer hides the
    /// per-vertex line style, leaving its constant-width style, which
    /// rewrites the radius column from the panel's slider.
    bool                  has_vertex_radius = false;
};

/// Open an interactive Vulkan/GLFW window and render the scene.  Blocks
/// until the window is closed.  SPIR-V shaders are loaded from the same
/// directory as the extension module.
///
/// aa_mode: 0 = hw_msaa (hardware MSAA, 2x/4x adjustable at runtime),
///          1 = gauss_msaa (2x supersampling + 6x6 Gaussian resolve),
///          2 = no AA (hardware path locked to 1 sample).
/// msaa_samples: initial hardware sample count (1, 2 or 4).  Ignored for
/// gauss_msaa and none; clamped down if the device cannot honour it.  The
/// viewer's panel can change it at runtime.
/// background: initial clear colour.  Training renders against either white
/// or black (fuzzydr's `white_bg` flag) and records it as "bg_color" in the
/// checkpoint when it is not white, so callers pass that through here.
/// flip_up: start with the camera's up reference pointing down -Z.  Datasets
/// differ in which way their Z axis points, so some scenes load upside down;
/// this only affects the view matrix, never the geometry.  Toggleable at
/// runtime from the viewer panel.
void launch(const SceneData& scene,
            int width = 1280,
            int height = 720,
            int aa_mode = 0,
            int msaa_samples = 2,
            float bg_r = 1.f, float bg_g = 1.f, float bg_b = 1.f,
            bool flip_up = false,
            const std::string& screenshot_dir = "");

} // namespace fuzzydr_viewer

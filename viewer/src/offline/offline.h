// src/offline/offline.h
//
// Offscreen, scripted-camera rendering path used exclusively by
// fuzzydr_viewer.benchmark(...).  Shares the viewer's SPIR-V shaders (viewer.vert /
// viewer.frag / viewer_sh_eval.comp) but stands up its own Vulkan context
// with no GLFW surface, no swapchain, no ImGui, no vsync.
//
// Scope: line-only.  Faces are NOT supported in this path - every checkpoint
// fed into the benchmark is expected to carry zero faces; the interactive
// viewer (fuzzydr_viewer.launch) handles face rendering separately.
//
// Pipeline it exercises:
//   1. SH compute dispatch (forced every frame, no camera-dirty guard)
//   2. Scene render pass: Bresenham LINE_LIST.
//   3. For aa_mode == gauss_msaa, a fullscreen Gaussian-resolve pass that
//      downsamples the 2x-SSAA scene target to the output resolution.
//
// Timing model: CPU wall-clock around
//     (record command buffer) + (queue submit) + (vkQueueWaitIdle)
// which is "what the renderer can do" with GPU sync but no presentation.
// See fuzzydr_viewer.benchmark in python/fuzzydr_viewer/__init__.py for the public API.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace fuzzydr_viewer {

/// Packed benchmark input - vert/SH/line layout matches SceneData (minus
/// the face index list, which the benchmark path does not support), plus a
/// list of pre-computed camera matrices and timing parameters.
struct BenchmarkInput {
    // -- Geometry (lines only) --------------------------------------------
    std::vector<float>    vert_attrs;   // float[num_verts * 7]
    uint32_t              num_verts = 0;

    std::vector<float>    sh_coeffs;    // float[num_verts * sh_cols]
    uint32_t              sh_cols = 0;

    std::vector<uint32_t> lines;        // uint32 index buffer (size = num_indices)
    uint32_t              num_indices = 0;  // # of indices in `lines`

    // Index-buffer interpretation.  0 = LINE_LIST (every two indices form a
    // segment); 1 = LINE_STRIP with primitive-restart sentinel UINT32_MAX
    // separating maximal walks.  Strip mode pairs naturally with a CPU-side
    // greedy maximal-walk rebake of the original LINE_LIST, halving the
    // vertex-shader load on polyline-soup data.
    uint32_t              line_topology = 0;

    std::string           compute_shader;  // default: viewer_sh_eval.comp.spv

    // -- Cameras (one entry per view) ------------------------------------
    // viewprojs : row-major 4x4, float[num_views * 16]
    // eyes      : float[num_views * 3]  (camera world-space position)
    std::vector<float>    viewprojs;
    std::vector<float>    eyes;
    uint32_t              num_views = 0;

    // -- Frame size ------------------------------------------------------
    uint32_t              width  = 1920;
    uint32_t              height = 1080;

    // -- Warmup / measure counts per view --------------------------------
    uint32_t              warmup  = 20;
    uint32_t              measure = 30;

    // -- Optional: read back one resolved RGBA8 image per view ----------
    // When true, one extra *untimed* frame per view is recorded after the
    // measured loop finishes, and the resolve target is copied to a
    // host-visible staging buffer.  The measured timings are unaffected.
    bool                  capture_screenshots = false;

    // -- Optional: also time a graphics-only loop (no SH dispatch) ------
    // When true, a second warmup + measure loop runs after each view's
    // full-pipeline loop, recording the same command buffer *without* the
    // SH compute dispatch and reusing the evalBuf populated by the
    // preceding loop.  The resulting timings isolate the graphics-pipeline
    // cost (vertex shader + rasterization + AA resolve) from the SH
    // evaluation cost.
    bool                  measure_nosh = false;

    // -- Antialiasing mode ----------------------------------------------
    // All four modes use the same Bresenham line rasterizer; they differ
    // only in the sample count and resolve filter:
    //   0 = hw_msaa_4x : bresenham + 4x hardware MSAA.  Scene renders at
    //                    native (W, H) into an RGBA8 N-sample attachment
    //                    and the driver box-averages samples into the
    //                    1-sample resolveImg.
    //   1 = gauss_msaa : bresenham + 2x supersampling + 6x6 sigma=0.5 isotropic
    //                    Gaussian resolve.  Scene renders at (2W, 2H) into
    //                    a 1-sample fp16 attachment; a fullscreen pass
    //                    downsamples it to (W, H).
    //   2 = none       : 1-sample target at native res.  No AA.
    //   3 = hw_msaa_2x : bresenham + 2x hardware MSAA, otherwise identical
    //                    to hw_msaa_4x.
    uint32_t              aa_mode = 0;
};

/// Result of a benchmark run.
///
///   times       : float[num_views * (warmup + measure)], row-major over
///                 (view, frame).  Wall-clock seconds per frame, warmup
///                 frames first so a caller can eyeball the transient.
///                 *Full pipeline* including the SH compute dispatch.
///   times_nosh  : same shape as ``times``, empty unless
///                 ``input.measure_nosh`` was set.  Wall-clock seconds
///                 per *graphics-only* frame (the SH compute dispatch
///                 is skipped; evalBuf carries over from the preceding
///                 full-pipeline loop so rendered pixels are identical).
///   images      : empty unless input.capture_screenshots was set, in
///                 which case it holds num_views * height * width * 4
///                 bytes of RGBA8 in row-major (view, y, x, channel)
///                 order.
///
/// SH compute is dispatched on every frame of the main ``times`` loop
/// regardless of camera state - no "campos unchanged, skip compute"
/// guard.  For aa_mode == hw_msaa_{2x,4x}, the requested sample count is
/// clamped against device support.
struct BenchmarkOutput {
    std::vector<float>   times;
    std::vector<float>   times_nosh;
    std::vector<uint8_t> images;
};

BenchmarkOutput benchmark(const BenchmarkInput& input);


// ---------------------------------------------------------------------------
// StreamRenderer - persistent-context offscreen renderer.  Static geometry
// (sh_coeffs / faces / lines) is uploaded once; each frame() pushes a fresh
// [N, 7] vert_attrs through a permanently-mapped host-visible staging buffer
// and runs the SH compute + scene render + optional gauss-resolve passes
// in one vkQueueSubmit, optionally reading back the resolved RGBA8 image.
// ---------------------------------------------------------------------------

struct StreamRendererInput {
    std::vector<float>    vert_attrs;     // float[num_verts * 7]  initial
    uint32_t              num_verts = 0;

    std::vector<float>    sh_coeffs;      // float[num_verts * sh_cols]
    uint32_t              sh_cols = 0;

    std::vector<uint32_t> lines;          // uint32 index buffer (size = num_indices)
    uint32_t              num_indices = 0;
    uint32_t              line_topology = 0;  // see BenchmarkInput::line_topology

    std::string           compute_shader; // default: viewer_sh_eval.comp.spv

    uint32_t              width  = 1920;
    uint32_t              height = 1080;
    uint32_t              aa_mode = 0;    // see BenchmarkInput::aa_mode
};

/// One streamed frame.  ``time_seconds`` is wall-clock around the render
/// submit only - the optional readback is a separate untimed submit.
/// ``image`` is empty unless ``capture`` was true; otherwise it holds
/// height*width*4 bytes of RGBA8 in row-major (y, x, channel) order.
struct StreamFrameOutput {
    float                 time_seconds = 0.f;
    std::vector<uint8_t>  image;
};

class StreamRenderer {
public:
    explicit StreamRenderer(const StreamRendererInput& input);
    ~StreamRenderer();

    StreamRenderer(const StreamRenderer&)            = delete;
    StreamRenderer& operator=(const StreamRenderer&) = delete;

    /// Render one frame.  ``vert_attrs`` is num_verts*7 floats (must match
    /// the constructor's num_verts).  ``viewproj`` is row-major 4x4.
    StreamFrameOutput frame(const float* vert_attrs,
                            const float  viewproj[16],
                            const float  eye[3],
                            bool         capture);

    uint32_t width()     const;
    uint32_t height()    const;
    uint32_t num_verts() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace fuzzydr_viewer

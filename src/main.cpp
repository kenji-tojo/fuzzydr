// src/main.cpp
#include <cstdint>
#include <memory>
#include <stdexcept>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "vk_common.h"
#include "rasterize.h"
#include "filter.h"

// -------------------------------
// nanobind
// -------------------------------
namespace nb = nanobind;
using namespace nb::literals;

// Module-level owned instances.
static std::unique_ptr<fuzzydr::Rasterizer> g_rasterizer;
static std::unique_ptr<fuzzydr::Filter>     g_filter;

NB_MODULE(_core, m) {
  m.attr("__version__") = "0.1.0";

  // Explicit lifecycle management for the global context.
  m.def("init",
    [](int gpu_index) {
      fuzzydr::initCtx(gpu_index);
    },
    nb::arg("gpu_index") = -1,
    "Initialize global Vulkan context (idempotent). gpu_index selects the "
    "Vulkan physical device by enumeration index; <0 (default) auto-picks the "
    "first suitable device or the FUZZYDR_DEVICE_INDEX env var if set."
  );

  m.def("shutdown",
    []() {
      // Destroy GPU resources before tearing down the Vulkan context.
      g_rasterizer.reset();
      g_filter.reset();
      fuzzydr::shutdownCtx();
    },
    "Destroy global Vulkan context and release all GPU resources."
  );

  m.def(
    "rasterize",
    [](nb::ndarray<float,    nb::c_contig, nb::shape<-1, 7>>     vert_attrs,      // [num_verts, 7]
       uint32_t num_faces,
       nb::ndarray<uint32_t, nb::c_contig, nb::shape<-1, 3>>     faces,           // [num_faces, 3]
       nb::ndarray<float,    nb::c_contig, nb::shape<-1>>        face_opacity,    // [num_faces]
       uint32_t num_lines,
       nb::ndarray<uint32_t, nb::c_contig, nb::shape<-1, 2>>     lines,           // [num_lines, 2]
       nb::ndarray<float,    nb::c_contig, nb::shape<-1>>        line_opacity,    // [num_lines]
       uint32_t num_points,
       nb::ndarray<uint32_t, nb::c_contig, nb::shape<-1>>        points,          // [num_points]
       nb::ndarray<float,    nb::c_contig, nb::shape<-1>>        point_opacity,   // [num_points]
       nb::ndarray<float,    nb::c_contig, nb::shape<4, 4>>      viewproj,        // [4, 4] row-major (always CPU)
       nb::ndarray<float,    nb::c_contig, nb::shape<3>>         campos,          // [3] (always CPU)
       float tau,
       uint32_t seed,
       nb::ndarray<uint32_t, nb::c_contig, nb::shape<-1, -1>>    prim_id_out,     // [height, width] or empty
       nb::ndarray<float,    nb::c_contig, nb::shape<-1, -1, 4>> bary_depth_out,  // [height, width, 4] or empty
       nb::ndarray<float,    nb::c_contig, nb::shape<-1, -1, 4>> rgba_out,        // [height, width, 4]
       nb::ndarray<uint8_t,  nb::c_contig, nb::shape<4>>         render_options,  // [NUM_RENDER_OPTIONS] (always CPU)
       bool is_cuda /*=false*/
    ) {
      const uint32_t num_verts = (uint32_t)vert_attrs.shape(0);
      const uint32_t height    = (uint32_t)rgba_out.shape(0);
      const uint32_t width     = (uint32_t)rgba_out.shape(1);

      // faces/lines arrays may be empty (size 0) when using cached topology.
      const bool has_faces_data  = (faces.size() > 0);
      const bool has_lines_data  = (lines.size() > 0);
      const bool has_points_data = (points.size() > 0);

      // prim_id_out and bary_depth_out may be empty (size 0) to skip download.
      const bool has_prim_id    = (prim_id_out.size() > 0);
      const bool has_bary_depth = (bary_depth_out.size() > 0);

      if (has_prim_id) {
        if ((uint32_t)prim_id_out.shape(0) != height || (uint32_t)prim_id_out.shape(1) != width) {
          throw std::runtime_error("rasterize: prim_id_out must have same [height, width] as rgba_out or be empty");
        }
      }
      if (has_bary_depth) {
        if ((uint32_t)bary_depth_out.shape(0) != height || (uint32_t)bary_depth_out.shape(1) != width) {
          throw std::runtime_error("rasterize: bary_depth_out must have same [height, width] as rgba_out or be empty");
        }
      }
      if ((uint32_t)face_opacity.shape(0) != num_faces) {
        throw std::runtime_error("rasterize: face_opacity must have shape [num_faces]");
      }
      if ((uint32_t)line_opacity.shape(0) != num_lines) {
        throw std::runtime_error("rasterize: line_opacity must have shape [num_lines]");
      }
      if ((uint32_t)point_opacity.shape(0) != num_points) {
        throw std::runtime_error("rasterize: point_opacity must have shape [num_points]");
      }

      if (!g_rasterizer) g_rasterizer = std::make_unique<fuzzydr::Rasterizer>();

      return g_rasterizer->rasterize(
        num_verts, vert_attrs.data(),
        num_faces, has_faces_data ? faces.data() : nullptr,
        face_opacity.data(),
        num_lines, has_lines_data ? lines.data() : nullptr,
        line_opacity.data(),
        num_points, has_points_data ? points.data() : nullptr,
        point_opacity.data(),
        width, height,
        viewproj.data(), campos.data(),
        tau, seed,
        has_prim_id    ? prim_id_out.data()    : nullptr,
        has_bary_depth ? bary_depth_out.data() : nullptr,
        rgba_out.data(),
        render_options.data(),
        is_cuda
      );
    },
    "vert_attrs"_a, "num_faces"_a, "faces"_a, "face_opacity"_a,
    "num_lines"_a, "lines"_a, "line_opacity"_a,
    "num_points"_a, "points"_a, "point_opacity"_a,
    "viewproj"_a, "campos"_a,
    "tau"_a, "seed"_a,
    "prim_id_out"_a, "bary_depth_out"_a, "rgba_out"_a,
    "render_options"_a, "is_cuda"_a = false,
    "Rasterize triangles, lines, and/or points with packed per-vertex attributes, topology, "
    "and per-primitive opacity. At least one of faces, lines, or points must be non-empty. "
    "Writes primitive IDs, interpolation weights + depth, and RGBA to output buffers. "
    "viewproj is a row-major float32[4,4] view-projection matrix (always CPU pointer). "
    "campos is a float32[3] camera position (always CPU pointer). "
    "render_options is a uint8[4] CPU tensor packing boolean flags: "
    "[0]=white_bg, [1]=bresen_lines (1-px Bresenham LINE_LIST, radius ignored), "
    "[2]=cull_back_faces (discard CW faces; lines unaffected), "
    "[3]=aux_buffers (download prim_id + bary_depth; ignored when 0). "
    "tau < 0 selects stochastic masking with a per-primitive threshold. "
    "If faces (or lines) is empty but num_faces (or num_lines) is > 0, "
    "the previously cached topology buffer is reused. "
    "If prim_id_out or bary_depth_out is empty (size 0), the corresponding "
    "host download is skipped (the GPU buffer is still written for backward passes). "
    "If is_cuda is true, all pointer arguments (except viewproj/campos/render_options) are "
    "treated as CUDA device pointers."
  );

  m.def(
    "edge_grad",
    [](uint32_t num_faces,
       uint32_t num_lines,
       uint32_t num_points,
       nb::ndarray<float, nb::c_contig, nb::shape<-1, -1, 4>> grad_out_rgba,    // [height, width, 4]
       nb::ndarray<float, nb::c_contig, nb::shape<-1, 7>>     grad_vert_attrs,  // [num_verts, 7]
       bool is_cuda /*=false*/
    ) {
      const uint32_t num_verts = (uint32_t)grad_vert_attrs.shape(0);
      const uint32_t height    = (uint32_t)grad_out_rgba.shape(0);
      const uint32_t width     = (uint32_t)grad_out_rgba.shape(1);

      if (!g_rasterizer) throw std::runtime_error("edge_grad: rasterizer not initialized (call rasterize first)");

      return g_rasterizer->edge_grad(
        num_verts, num_faces, num_lines, num_points, width, height,
        grad_out_rgba.data(),
        grad_vert_attrs.data(),
        is_cuda
      );
    },
    "num_faces"_a, "num_lines"_a, "num_points"_a, "grad_out_rgba"_a, "grad_vert_attrs"_a, "is_cuda"_a = false,
    "Edge-based raster backward: accumulates gradients into packed vertex attributes "
    "(world-space positions + RGB) from grad_out_rgba. "
    "The viewproj matrix from the forward cache is used to transform clip-space "
    "gradients back to world-space vertex gradients. "
    "Handles both face (barycentric) and line (2-weight) primitives via prim_type stored in forward cache."
  );

  m.def(
    "opacity_grad",
    [](nb::ndarray<float, nb::c_contig, nb::shape<-1, -1>> error,               // [height, width]
       nb::ndarray<float, nb::c_contig, nb::shape<-1>>     grad_face_opacity,   // [num_faces]
       nb::ndarray<float, nb::c_contig, nb::shape<-1>>     grad_line_opacity,   // [num_lines]
       nb::ndarray<float, nb::c_contig, nb::shape<-1>>     grad_point_opacity,  // [num_points]
       float eps, /*=1e-6f*/
       bool is_cuda /*=false*/
    ) {
      const uint32_t num_faces  = (uint32_t)grad_face_opacity.shape(0);
      const uint32_t num_lines  = (uint32_t)grad_line_opacity.shape(0);
      const uint32_t num_points = (uint32_t)grad_point_opacity.shape(0);
      const uint32_t height    = (uint32_t)error.shape(0);
      const uint32_t width     = (uint32_t)error.shape(1);

      if (!(eps > 0.0f))
        throw std::runtime_error("opacity_grad: eps must be > 0");

      if (!g_rasterizer) throw std::runtime_error("opacity_grad: rasterizer not initialized (call rasterize first)");

      return g_rasterizer->opacity_grad(
        num_faces, num_lines, num_points, width, height,
        error.data(),
        grad_face_opacity.data(),
        grad_line_opacity.data(),
        grad_point_opacity.data(),
        eps,
        is_cuda
      );
    },
    "error"_a, "grad_face_opacity"_a, "grad_line_opacity"_a, "grad_point_opacity"_a,
    "eps"_a = 1e-6f, "is_cuda"_a = false,
    "Likelihood-ratio gradient for stochastic opacity masking. "
    "Writes to grad_face_opacity[:], grad_line_opacity[:], and grad_point_opacity[:]. "
    "The face pass mirrors the forward backface-cull mode."
  );

  // ---- MSAA downsample (forward) ----
  m.def(
    "msaa_downsample_rgba",
    [](nb::ndarray<float, nb::c_contig, nb::shape<-1, -1, 4>> in_img,   // [height, width, 4]
       nb::ndarray<float, nb::c_contig, nb::shape<-1, -1, 3>> out_img,  // [height/2, width/2, 3]
       float sigma,
       bool is_cuda /*=false*/
    ) {
      const uint32_t height = (uint32_t)in_img.shape(0);
      const uint32_t width  = (uint32_t)in_img.shape(1);

      if (width % 2 != 0 || height % 2 != 0) {
        throw std::runtime_error("msaa_downsample_rgba: width and height must be even");
      }
      if ((uint32_t)out_img.shape(0) != height / 2 ||
          (uint32_t)out_img.shape(1) != width / 2) {
        throw std::runtime_error(
          "msaa_downsample_rgba: out_img must have shape [height/2, width/2, 3]");
      }

      if (!g_filter) g_filter = std::make_unique<fuzzydr::Filter>();

      return g_filter->msaa_downsample_rgba(
        width, height, in_img.data(), out_img.data(), sigma, is_cuda
      );
    },
    "in_img"_a, "out_img"_a, "sigma"_a = 0.5f, "is_cuda"_a = false,
    "MSAA pixel reconstruction: isotropic Gaussian weighted downsample.  "
    "Input [H,W,4] (alpha ignored) -> Output [H/2,W/2,3].  "
    "sigma is the Gaussian standard deviation in output pixels (default 0.5)."
  );

  // ---- MSAA downsample (backward) ----
  m.def(
    "msaa_downsample_rgba_backward",
    [](nb::ndarray<float, nb::c_contig, nb::shape<-1, -1, 3>> grad_out,  // [height/2, width/2, 3]
       nb::ndarray<float, nb::c_contig, nb::shape<-1, -1, 4>> grad_in,   // [height, width, 4]
       float sigma,
       bool is_cuda /*=false*/
    ) {
      const uint32_t height = (uint32_t)grad_in.shape(0);
      const uint32_t width  = (uint32_t)grad_in.shape(1);

      if (width % 2 != 0 || height % 2 != 0) {
        throw std::runtime_error(
          "msaa_downsample_rgba_backward: width and height must be even");
      }
      if ((uint32_t)grad_out.shape(0) != height / 2 ||
          (uint32_t)grad_out.shape(1) != width / 2) {
        throw std::runtime_error(
          "msaa_downsample_rgba_backward: grad_out must have shape [height/2, width/2, 3]");
      }

      if (!g_filter) g_filter = std::make_unique<fuzzydr::Filter>();

      return g_filter->msaa_downsample_rgba_backward(
        width, height, grad_out.data(), grad_in.data(), sigma, is_cuda
      );
    },
    "grad_out"_a, "grad_in"_a, "sigma"_a = 0.5f, "is_cuda"_a = false,
    "Backward pass for msaa_downsample_rgba (adjoint of the forward filter). "
    "grad_out [H/2,W/2,3] -> grad_in [H,W,4] (channel 3 written as 0). "
    "sigma must match forward."
  );

}

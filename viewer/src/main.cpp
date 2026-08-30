// src/main.cpp
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>

#include <memory>
#include <stdexcept>

#include "viewer.h"
#include "offline/offline.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_core, m) {
    m.doc() = "fuzzydr_viewer - Vulkan-based interactive scene viewer (C++ core)";

    // ----------------------------------------------------------------------
    // Interactive viewer.  aa_mode:
    //   0 = hw_msaa    - hardware MSAA, 2x/4x toggleable at runtime
    //   1 = gauss_msaa - 2x supersampling + 6x6 Gaussian resolve (4 samples)
    //   2 = none       - hardware path pinned to a single sample
    // ----------------------------------------------------------------------
    m.def("_launch",
          [](nb::bytes vert_bytes,  int num_verts,
             nb::bytes face_bytes,  int num_faces,
             nb::bytes line_bytes,  int num_lines,
             nb::bytes strip_bytes, int num_strip_indices,
             nb::bytes sh_bytes,    int sh_cols,
             int width, int height,
             int aa_mode, int msaa_samples,
             bool has_vertex_radius,
             float bg_r, float bg_g, float bg_b,
             bool flip_up,
             std::string screenshot_dir) {

              if (num_verts <= 0)
                  throw std::runtime_error("vert_attrs must not be empty");
              if (num_faces == 0 && num_lines == 0)
                  throw std::runtime_error(
                      "Scene must contain at least one face or line");
              if (aa_mode < 0 || aa_mode > 2)
                  throw std::runtime_error(
                      "aa_mode must be 0 (hw_msaa_4x), 1 (gauss_msaa) "
                      "or 2 (none)");

              fuzzydr_viewer::SceneData s;
              s.num_verts = uint32_t(num_verts);
              s.num_faces = uint32_t(num_faces);
              s.num_lines = uint32_t(num_lines);
              s.num_strip_indices = uint32_t(num_strip_indices);
              s.has_vertex_radius = has_vertex_radius;

              const auto* vp = reinterpret_cast<const float*>(vert_bytes.c_str());
              s.vert_attrs.assign(vp, vp + num_verts * 7);

              if (sh_cols > 0) {
                  s.sh_cols = uint32_t(sh_cols);
                  const auto* sp = reinterpret_cast<const float*>(sh_bytes.c_str());
                  s.sh_coeffs.assign(sp, sp + num_verts * sh_cols);
              }

              if (num_faces > 0) {
                  const auto* fp = reinterpret_cast<const uint32_t*>(face_bytes.c_str());
                  s.faces.assign(fp, fp + num_faces * 3);
              }
              if (num_lines > 0) {
                  const auto* lp = reinterpret_cast<const uint32_t*>(line_bytes.c_str());
                  s.lines.assign(lp, lp + num_lines * 2);
              }
              if (num_strip_indices > 0) {
                  const auto* sp = reinterpret_cast<const uint32_t*>(strip_bytes.c_str());
                  s.line_strip.assign(sp, sp + num_strip_indices);
              }

              fuzzydr_viewer::launch(s, width, height, aa_mode, msaa_samples,
                                     bg_r, bg_g, bg_b, flip_up, screenshot_dir);
          },
          "vert_bytes"_a, "num_verts"_a,
          "face_bytes"_a, "num_faces"_a,
          "line_bytes"_a, "num_lines"_a,
          "strip_bytes"_a, "num_strip_indices"_a,
          "sh_bytes"_a,   "sh_cols"_a,
          "width"_a  = 1280,
          "height"_a = 720,
          "aa_mode"_a = 0,
          "msaa_samples"_a = 2,
          "has_vertex_radius"_a = false,
          "bg_r"_a = 1.f, "bg_g"_a = 1.f, "bg_b"_a = 1.f,
          "flip_up"_a = false,
          "screenshot_dir"_a = "");

    // ----------------------------------------------------------------------
    // Offscreen scripted-camera benchmark.  Returns a tuple
    //   (times, times_nosh, images_bytes)
    // with times flat [num_views * (warmup + measure)], times_nosh empty
    // unless measure_nosh=True, and images_bytes empty unless
    // capture_screenshots=True.  The Python wrapper reshapes.
    // ----------------------------------------------------------------------
    m.def("_benchmark",
          [](nb::bytes vert_bytes,     int num_verts,
             nb::bytes line_bytes,     int num_indices,
             nb::bytes sh_bytes,       int sh_cols,
             std::string compute_shader,
             nb::bytes viewproj_bytes, int num_views,
             nb::bytes eye_bytes,
             int width, int height,
             int warmup, int measure,
             bool capture_screenshots,
             bool measure_nosh,
             int aa_mode,
             int line_topology) {

              if (num_verts <= 0)
                  throw std::runtime_error("vert_attrs must not be empty");
              if (num_indices <= 0)
                  throw std::runtime_error("benchmark is line-only; num_indices must be > 0");
              if (line_topology < 0 || line_topology > 1)
                  throw std::runtime_error("line_topology must be 0 (LIST) or 1 (STRIP)");
              if (num_views <= 0)
                  throw std::runtime_error("num_views must be > 0");
              if (warmup < 0 || measure <= 0)
                  throw std::runtime_error("warmup >= 0 and measure > 0 required");
              if (width <= 0 || height <= 0)
                  throw std::runtime_error("width / height must be > 0");
              if (aa_mode < 0 || aa_mode > 3)
                  throw std::runtime_error(
                      "aa_mode must be 0..3 "
                      "(hw_msaa_4x / gauss_msaa / bresenham / hw_msaa_2x)");

              fuzzydr_viewer::BenchmarkInput in;
              in.num_verts   = uint32_t(num_verts);
              in.num_indices = uint32_t(num_indices);
              in.line_topology = uint32_t(line_topology);
              in.compute_shader = compute_shader;
              in.num_views = uint32_t(num_views);
              in.width     = uint32_t(width);
              in.height    = uint32_t(height);
              in.warmup    = uint32_t(warmup);
              in.measure   = uint32_t(measure);
              in.capture_screenshots = capture_screenshots;
              in.measure_nosh = measure_nosh;
              in.aa_mode    = uint32_t(aa_mode);

              const auto* vp = reinterpret_cast<const float*>(vert_bytes.c_str());
              in.vert_attrs.assign(vp, vp + size_t(num_verts) * 7);

              if (sh_cols > 0) {
                  in.sh_cols = uint32_t(sh_cols);
                  const auto* sp = reinterpret_cast<const float*>(sh_bytes.c_str());
                  in.sh_coeffs.assign(sp, sp + size_t(num_verts) * sh_cols);
              }

              const auto* lp = reinterpret_cast<const uint32_t*>(line_bytes.c_str());
              in.lines.assign(lp, lp + size_t(num_indices));

              const auto* vjp = reinterpret_cast<const float*>(viewproj_bytes.c_str());
              in.viewprojs.assign(vjp, vjp + size_t(num_views) * 16);
              const auto* ep = reinterpret_cast<const float*>(eye_bytes.c_str());
              in.eyes.assign(ep, ep + size_t(num_views) * 3);

              fuzzydr_viewer::BenchmarkOutput res = fuzzydr_viewer::benchmark(in);

              nb::bytes images_b(
                  reinterpret_cast<const char*>(res.images.data()),
                  res.images.size());
              return nb::make_tuple(std::move(res.times),
                                    std::move(res.times_nosh),
                                    std::move(images_b));
          },
          "vert_bytes"_a,     "num_verts"_a,
          "line_bytes"_a,     "num_indices"_a,
          "sh_bytes"_a,       "sh_cols"_a,
          "compute_shader"_a,
          "viewproj_bytes"_a, "num_views"_a,
          "eye_bytes"_a,
          "width"_a,  "height"_a,
          "warmup"_a, "measure"_a,
          "capture_screenshots"_a,
          "measure_nosh"_a,
          "aa_mode"_a,
          "line_topology"_a);

    // ----------------------------------------------------------------------
    // Persistent-context offscreen renderer.  See fuzzydr_viewer.OffscreenRenderer.
    // ----------------------------------------------------------------------
    nb::class_<fuzzydr_viewer::StreamRenderer>(m, "_OffscreenRenderer")
        .def("__init__",
             [](fuzzydr_viewer::StreamRenderer* self,
                nb::bytes vert_bytes, int num_verts,
                nb::bytes line_bytes, int num_indices,
                nb::bytes sh_bytes,   int sh_cols,
                std::string compute_shader,
                int width, int height,
                int aa_mode,
                int line_topology) {

                 if (num_verts <= 0)
                     throw std::runtime_error("vert_attrs must not be empty");
                 if (num_indices <= 0)
                     throw std::runtime_error(
                         "OffscreenRenderer is line-only; num_indices must be > 0");
                 if (line_topology < 0 || line_topology > 1)
                     throw std::runtime_error("line_topology must be 0 (LIST) or 1 (STRIP)");
                 if (width <= 0 || height <= 0)
                     throw std::runtime_error("width / height must be > 0");
                 if (aa_mode < 0 || aa_mode > 3)
                     throw std::runtime_error(
                         "aa_mode must be 0..3 "
                         "(hw_msaa_4x / gauss_msaa / bresenham / hw_msaa_2x)");

                 fuzzydr_viewer::StreamRendererInput in;
                 in.num_verts     = uint32_t(num_verts);
                 in.num_indices   = uint32_t(num_indices);
                 in.line_topology = uint32_t(line_topology);
                 in.compute_shader = compute_shader;
                 in.width          = uint32_t(width);
                 in.height         = uint32_t(height);
                 in.aa_mode        = uint32_t(aa_mode);

                 const auto* vp = reinterpret_cast<const float*>(vert_bytes.c_str());
                 in.vert_attrs.assign(vp, vp + size_t(num_verts) * 7);

                 if (sh_cols > 0) {
                     in.sh_cols = uint32_t(sh_cols);
                     const auto* sp = reinterpret_cast<const float*>(sh_bytes.c_str());
                     in.sh_coeffs.assign(sp, sp + size_t(num_verts) * sh_cols);
                 }
                 const auto* lp = reinterpret_cast<const uint32_t*>(line_bytes.c_str());
                 in.lines.assign(lp, lp + size_t(num_indices));

                 new (self) fuzzydr_viewer::StreamRenderer(in);
             },
             "vert_bytes"_a, "num_verts"_a,
             "line_bytes"_a, "num_indices"_a,
             "sh_bytes"_a,   "sh_cols"_a,
             "compute_shader"_a = "",
             "width"_a  = 1920,
             "height"_a = 1080,
             "aa_mode"_a = 0,
             "line_topology"_a = 0)
        .def("frame",
             [](fuzzydr_viewer::StreamRenderer& self,
                nb::bytes vert_bytes,
                nb::bytes viewproj_bytes,
                nb::bytes eye_bytes,
                bool      capture) {

                 const size_t expected_va = size_t(self.num_verts()) * 7 * sizeof(float);
                 if (vert_bytes.size() < expected_va)
                     throw std::runtime_error(
                         "frame: vert_bytes shorter than num_verts * 7 floats");
                 if (viewproj_bytes.size() < 16 * sizeof(float))
                     throw std::runtime_error("frame: viewproj must be 16 floats");
                 if (eye_bytes.size() < 3 * sizeof(float))
                     throw std::runtime_error("frame: eye must be 3 floats");

                 const auto* va = reinterpret_cast<const float*>(vert_bytes.c_str());
                 const auto* vp = reinterpret_cast<const float*>(viewproj_bytes.c_str());
                 const auto* ey = reinterpret_cast<const float*>(eye_bytes.c_str());

                 fuzzydr_viewer::StreamFrameOutput out = self.frame(va, vp, ey, capture);

                 nb::bytes img_b(reinterpret_cast<const char*>(out.image.data()),
                                 out.image.size());
                 return nb::make_tuple(out.time_seconds, std::move(img_b));
             },
             "vert_bytes"_a,
             "viewproj_bytes"_a,
             "eye_bytes"_a,
             "capture"_a = false)
        .def_prop_ro("width",     &fuzzydr_viewer::StreamRenderer::width)
        .def_prop_ro("height",    &fuzzydr_viewer::StreamRenderer::height)
        .def_prop_ro("num_verts", &fuzzydr_viewer::StreamRenderer::num_verts);
}

// src/filter.h
#pragma once
#include <cstdint>
#include <memory>

#include "vk_common.h"

namespace fuzzydr {

// Forward declaration for internal cache struct (defined in filter.cpp).
struct FilterCache;

/// Filter - GPU-accelerated image filtering operations.
/// Owns all GPU buffer caches; destroying the object releases them.
class Filter {
public:
  Filter();
  ~Filter();

  Filter(const Filter&) = delete;
  Filter& operator=(const Filter&) = delete;

  /// MSAA pixel reconstruction downsample: isotropic Gaussian weighted
  /// downsample from 2x supersampled input to half resolution.  Reads RGB
  /// from the 4-channel input (alpha ignored) and produces 3-channel RGB.
  /// width and height must be even.
  int msaa_downsample_rgba(
      uint32_t     width,            // input width  (even)
      uint32_t     height,           // input height (even)
      const float* in_rgba,          // [height,   width,   4] input (alpha ignored)
      float*       out_rgb,          // [height/2, width/2, 3] output
      float        sigma = 0.5f,     // Gaussian sigma in output pixels
      bool         is_cuda = false); // if true, pointers are CUDA device ptrs

  /// Backward pass for msaa_downsample_rgba (adjoint of the forward filter).
  /// grad_in_rgba channel 3 (alpha) is written as 0; the 4-channel shape
  /// mirrors the rasterizer's output layout.
  /// sigma must match the forward pass; width and height must be even.
  int msaa_downsample_rgba_backward(
      uint32_t     width,            // original input width  (even)
      uint32_t     height,           // original input height (even)
      const float* grad_out_rgb,     // [height/2, width/2, 3] grad w.r.t. output
      float*       grad_in_rgba,     // [height,   width,   4] grad w.r.t. input
      float        sigma = 0.5f,     // must match forward
      bool         is_cuda = false); // if true, pointers are CUDA device ptrs

private:
  std::unique_ptr<FilterCache> cache_;
};

} // namespace fuzzydr

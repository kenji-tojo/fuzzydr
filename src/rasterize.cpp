// src/rasterize.cpp
#include "rasterize.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(FUZZYDR_WITH_CUDA) && FUZZYDR_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include "vk_common.h"

namespace fuzzydr {

// ===============================
// Internal cache structs (defined here, forward-declared in rasterize.h)
// ===============================

struct ForwardCache {
  bool valid = false;
  uint32_t width = 0, height = 0;
  uint32_t num_verts = 0, num_faces = 0, num_lines = 0, num_points = 0;

  // Render options snapshotted from the forward call - read by opacity_grad()
  // to reproduce the same pipeline (Bresenham vs quad, cull mode).
  uint8_t render_options[NUM_RENDER_OPTIONS] = {};

  // Forward inputs uploaded to GPU buffers and kept for reuse by grads.
  VulkanBuffer vboAttr;       // [num_verts, 7] floats: (x, y, z, radius, r, g, b)
  VulkanBuffer ibo;           // [num_faces, 3] uint32
  VulkanBuffer faceOpacity;   // [num_faces] float
  VulkanBuffer lineIbo;       // [num_lines, 2] uint32
  VulkanBuffer lineOpacity;   // [num_lines] float
  VulkanBuffer pointIbo;      // [num_points] uint32
  VulkanBuffer pointOpacity;  // [num_points] float
  VulkanBuffer ubo;           // sizeof(UBO)

  // Staging + readbacks (reused as storage buffers in backward/grad)
  VulkanBuffer primIdReadback;  // [height, width] uint32 (TRANSFER_DST | STORAGE)
  VulkanBuffer bdReadback;      // [height, width, 4] float (TRANSFER_DST | STORAGE)
  VulkanBuffer rgbaReadback;    // [height, width, 4] float (TRANSFER_DST | STORAGE)

  // Cached render target images (avoid per-frame vkAllocateMemory/vkFreeMemory)
  Image idsImg{}, baryDepthImg{}, rgbaImg{}, depthImg{};
  uint32_t imgWidth = 0, imgHeight = 0;

  void invalidate() {
    valid = false;
    width = height = num_verts = num_faces = num_lines = num_points = 0;
    memset(render_options, 0, sizeof(render_options));
  }

  void requireMatch(uint32_t W, uint32_t H, const char* who) const {
    if (!valid || width != W || height != H) {
      throw std::runtime_error(std::string(who) + ": forward cache inconsistent (missing forward or size mismatch)");
    }
  }

  void requireMesh(uint32_t N, uint32_t M, uint32_t L, uint32_t P, const char* who) const {
    if (!valid || num_verts != N || num_faces != M || num_lines != L || num_points != P) {
      throw std::runtime_error(std::string(who) + ": forward cache inconsistent (mesh size mismatch)");
    }
  }

  void resetAll(VkDevice device) {
    vboAttr.reset(device);
    ibo.reset(device);
    faceOpacity.reset(device);
    lineIbo.reset(device);
    lineOpacity.reset(device);
    pointIbo.reset(device);
    pointOpacity.reset(device);
    ubo.reset(device);
    primIdReadback.reset(device);
    bdReadback.reset(device);
    rgbaReadback.reset(device);
    destroyImage(device, idsImg);
    destroyImage(device, baryDepthImg);
    destroyImage(device, rgbaImg);
    destroyImage(device, depthImg);
    imgWidth = imgHeight = 0;
    invalidate();
  }
};

struct EdgeGradCache {
  VulkanBuffer gOutBuf;      // [height, width, 4] float  (grad RGBA)
  VulkanBuffer gVertBuf;     // [num_verts, 7] float      (grad vert_attrs)

  void resetAll(VkDevice device) {
    gOutBuf.reset(device);
    gVertBuf.reset(device);
  }
};

struct OpacityGradCache {
  VulkanBuffer errBuf;     // [height, width] float
  VulkanBuffer gfopBuf;    // [num_faces] float  (grad_face_opacity)
  VulkanBuffer glopBuf;    // [num_lines] float  (grad_line_opacity)
  VulkanBuffer gpopBuf;    // [num_points] float (grad_point_opacity)

  // Cached dummy render target (avoid per-frame alloc/free)
  Image dummyColor{};
  uint32_t imgWidth = 0, imgHeight = 0;

  void resetAll(VkDevice device) {
    errBuf.reset(device);
    gfopBuf.reset(device);
    glopBuf.reset(device);
    gpopBuf.reset(device);
    destroyImage(device, dummyColor);
    imgWidth = imgHeight = 0;
  }
};

// Cached shader modules: loaded once from disk and reused, avoiding per-frame
// SPIR-V file I/O and vkCreateShaderModule/vkDestroyShaderModule churn.
struct ShaderCache {
  VkShaderModule rasterizeVert  = VK_NULL_HANDLE;  // rasterize.vert.spv
  VkShaderModule rasterizeFrag  = VK_NULL_HANDLE;  // rasterize.frag.spv
  VkShaderModule lineQuadsVert  = VK_NULL_HANDLE;  // rasterize_line_quads.vert.spv
  VkShaderModule lineBresenVert = VK_NULL_HANDLE;  // rasterize_line_bresen.vert.spv
  VkShaderModule pointVert      = VK_NULL_HANDLE;  // rasterize_point.vert.spv
  VkShaderModule edgeGradComp   = VK_NULL_HANDLE;  // edge_grad.comp.spv
  VkShaderModule opacityGradFrag = VK_NULL_HANDLE; // opacity_grad.frag.spv

  void ensureLoaded(VkDevice device) {
    if (rasterizeVert != VK_NULL_HANDLE) return;  // already loaded
    rasterizeVert   = createShaderModule(device, shaderPath("rasterize.vert.spv"));
    rasterizeFrag   = createShaderModule(device, shaderPath("rasterize.frag.spv"));
    lineQuadsVert   = createShaderModule(device, shaderPath("rasterize_line_quads.vert.spv"));
    lineBresenVert  = createShaderModule(device, shaderPath("rasterize_line_bresen.vert.spv"));
    pointVert       = createShaderModule(device, shaderPath("rasterize_point.vert.spv"));
    edgeGradComp    = createShaderModule(device, shaderPath("edge_grad.comp.spv"));
    opacityGradFrag = createShaderModule(device, shaderPath("opacity_grad.frag.spv"));
  }

  void resetAll(VkDevice device) {
    auto destroy = [&](VkShaderModule& m) {
      if (m) { vkDestroyShaderModule(device, m, nullptr); m = VK_NULL_HANDLE; }
    };
    destroy(rasterizeVert);
    destroy(rasterizeFrag);
    destroy(lineQuadsVert);
    destroy(lineBresenVert);
    destroy(pointVert);
    destroy(edgeGradComp);
    destroy(opacityGradFrag);
  }
};

Rasterizer::Rasterizer() = default;

Rasterizer::~Rasterizer() {
  VkContext& ctx = requireCtx();
  VkDevice device = ctx.device;

  if (edge_)    { edge_->resetAll(device);    edge_.reset(); }
  if (opac_)    { opac_->resetAll(device);    opac_.reset(); }
  if (fwd_)     { fwd_->resetAll(device);     fwd_.reset(); }
  if (shaders_) { shaders_->resetAll(device); shaders_.reset(); }
}

// ===============================
// Shared structs
// ===============================

// Vertex attribute: (x,y,z,radius,r,g,b)
struct VertexAttr {
  float x, y, z, radius;
  float r, g, b;
};
static_assert(sizeof(VertexAttr) == sizeof(float) * 7, "VertexAttr must be 7 floats");

// layout(set=0,binding=0) uniform UBO {
//   mat4  viewproj;
//   vec3  campos;
//   float tau;
//   uint  seed;
//   uint  _pad0, _pad1, _pad2;
// } ubo;
struct UBO {
  float    viewproj[16]; // offset 0,  64 bytes (column-major mat4)
  float    campos[3];    // offset 64, 12 bytes
  float    tau;          // offset 76, 4 bytes
  uint32_t seed;         // offset 80, 4 bytes
  uint32_t _pad0;        // offset 84
  uint32_t _pad1;        // offset 88
  uint32_t _pad2;        // offset 92
};
static_assert(sizeof(UBO) == 96, "UBO must be 96 bytes");

// ===============================
// Push constants
// ===============================

// Forward rasterize (vertex + fragment)
// prim_type values used by the push constants AND by the bary_depth.w channel
// read by the edge_grad compute shader:
//   0 = face (triangle)
//   1 = bresen line  (LINE_LIST, 1-px; radius slot unused)
//   2 = quad line    (TRIANGLE_LIST, world-space width from radius)
//   3 = point        (POINT_LIST; radius slot unused)
struct RasterizePushConstants {
  uint32_t width;
  uint32_t height;
  uint32_t prim_type;
  // Face primitive IDs are offset by num_lines to form the RNG counter, so
  // that faces and lines do not share counter values.  Points cannot be mixed
  // with faces or lines, so point IDs need no offset.  See rasterize.frag.
  uint32_t num_lines;
};
static_assert(sizeof(RasterizePushConstants) == 16);

// Edge gradient (compute)
struct EdgeGradPushConstants {
  uint32_t width;
  uint32_t height;
};
static_assert(sizeof(EdgeGradPushConstants) == 8);

// Opacity backward (fragment)
struct OpacityGradPushConstants {
  uint32_t width;
  uint32_t height;
  uint32_t prim_type;        // see RasterizePushConstants for the full enum
  float    eps;
};
static_assert(sizeof(OpacityGradPushConstants) == 16);

// =============================================================================
// Shared helpers used by both the rasterize and opacity_grad passes
// =============================================================================

// Common graphics pipeline state shared by forward and opacity_grad passes.
struct GraphicsPipelineCommonState {
  VkVertexInputBindingDescription      bind{};
  std::array<VkVertexInputAttributeDescription, 2> attrs{};
  VkPipelineVertexInputStateCreateInfo visi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
  VkViewport                           vp{};
  VkRect2D                             sc{};
  VkPipelineViewportStateCreateInfo    vpsi{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
  VkPipelineRasterizationStateCreateInfo rsi{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
  VkPipelineMultisampleStateCreateInfo msi{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};

  void init(uint32_t width, uint32_t height) {
    // Single packed vertex buffer: binding 0, stride 7 floats
    bind.binding   = 0;
    bind.stride    = sizeof(VertexAttr);
    bind.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    // location 0: vec4 pos, location 1: vec3 rgb
    attrs[0].location = 0;
    attrs[0].binding  = 0;
    attrs[0].format   = VK_FORMAT_R32G32B32A32_SFLOAT;
    attrs[0].offset   = offsetof(VertexAttr, x);

    attrs[1].location = 1;
    attrs[1].binding  = 0;
    attrs[1].format   = VK_FORMAT_R32G32B32_SFLOAT;
    attrs[1].offset   = offsetof(VertexAttr, r);

    visi.vertexBindingDescriptionCount   = 1;
    visi.pVertexBindingDescriptions      = &bind;
    visi.vertexAttributeDescriptionCount = (uint32_t)attrs.size();
    visi.pVertexAttributeDescriptions    = attrs.data();

    vp.x        = 0;
    vp.y        = 0;
    vp.width    = (float)width;
    vp.height   = (float)height;
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;

    sc.offset.x      = 0;
    sc.offset.y      = 0;
    sc.extent.width  = width;
    sc.extent.height = height;

    vpsi.viewportCount = 1;
    vpsi.pViewports    = &vp;
    vpsi.scissorCount  = 1;
    vpsi.pScissors     = &sc;

    rsi.polygonMode = VK_POLYGON_MODE_FILL;
    rsi.cullMode    = VK_CULL_MODE_NONE;
    rsi.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rsi.lineWidth   = 1.0f;

    msi.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  }
};

// Create a graphics pipeline with the specified topology.
static VkPipeline createTopologyPipeline(
    VkDevice device,
    VkPrimitiveTopology topology,
    const VkPipelineShaderStageCreateInfo* stages, uint32_t stageCount,
    const GraphicsPipelineCommonState& pipeState,
    const VkPipelineDepthStencilStateCreateInfo& dssi,
    const VkPipelineColorBlendStateCreateInfo& cbsi,
    VkPipelineLayout pipelineLayout,
    VkRenderPass renderPass,
    const char* tag)
{
  VkPipelineInputAssemblyStateCreateInfo iasi{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
  iasi.topology = topology;

  VkGraphicsPipelineCreateInfo gpci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
  gpci.stageCount          = stageCount;
  gpci.pStages             = stages;
  gpci.pVertexInputState   = &pipeState.visi;
  gpci.pInputAssemblyState = &iasi;
  gpci.pViewportState      = &pipeState.vpsi;
  gpci.pRasterizationState = &pipeState.rsi;
  gpci.pMultisampleState   = &pipeState.msi;
  gpci.pDepthStencilState  = &dssi;
  gpci.pColorBlendState    = &cbsi;
  gpci.layout              = pipelineLayout;
  gpci.renderPass          = renderPass;
  gpci.subpass             = 0;

  VkPipeline pipeline = VK_NULL_HANDLE;
  vkCheck(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gpci, nullptr, &pipeline), tag);
  return pipeline;
}

// Helper: fill a VkWriteDescriptorSet for a storage buffer binding.
static VkWriteDescriptorSet makeStorageWrite(
    VkDescriptorSet ds, uint32_t binding, VkDescriptorBufferInfo* bi)
{
  VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  w.dstSet          = ds;
  w.dstBinding      = binding;
  w.descriptorCount = 1;
  w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  w.pBufferInfo     = bi;
  return w;
}

// ===============================
// Forward: rasterize
// ===============================

int Rasterizer::rasterize(
    uint32_t num_verts,
    const float* vert_attrs,          // [num_verts, 7] (x,y,z,radius,r,g,b)
    uint32_t num_faces,
    const uint32_t* faces,            // [num_faces, 3]  (null OK if num_faces==0; also null OK if num_faces>0 when a consistent cached buffer exists)
    const float* face_opacity,        // [num_faces]     (must be non-null when num_faces>0)
    uint32_t num_lines,
    const uint32_t* lines,            // [num_lines, 2]  (null OK if num_lines==0; also null OK if num_lines>0 when a consistent cached buffer exists)
    const float* line_opacity,        // [num_lines]     (must be non-null when num_lines>0)
    uint32_t num_points,
    const uint32_t* points,           // [num_points]    (null OK if num_points==0; also null OK if num_points>0 when a consistent cached buffer exists)
    const float* point_opacity,       // [num_points]    (must be non-null when num_points>0)
    uint32_t width,
    uint32_t height,
    const float* viewproj,            // [4,4] row-major (always CPU pointer)
    const float* campos,              // [3]             (always CPU pointer)
    float tau,                        // threshold ([-1] or [0,1])
    uint32_t seed,                    // rng seed
    uint32_t* prim_id_out,            // [height,width]    (null OK to skip download)
    float* bary_depth_out,            // [height,width,4]  (null OK to skip download)
    float* rgba_out,                  // [height,width,4]
    const uint8_t* render_options,    // [NUM_RENDER_OPTIONS] (always CPU pointer)
    bool is_cuda /*= false*/
) {
  if (!vert_attrs || !rgba_out)
    throw std::runtime_error("rasterize: vert_attrs and rgba_out must be non-null");
  if (!viewproj)
    throw std::runtime_error("rasterize: viewproj must be non-null");
  if (!campos)
    throw std::runtime_error("rasterize: campos must be non-null");
  if (!render_options)
    throw std::runtime_error("rasterize: render_options must be non-null");
  if (num_faces == 0 && num_lines == 0 && num_points == 0)
    throw std::runtime_error("rasterize: at least one of num_faces, num_lines, or num_points must be > 0");
  if (num_faces > 0 && !face_opacity)
    throw std::runtime_error("rasterize: face_opacity must be non-null when num_faces > 0");
  if (num_lines > 0 && !line_opacity)
    throw std::runtime_error("rasterize: line_opacity must be non-null when num_lines > 0");
  if (num_points > 0 && !point_opacity)
    throw std::runtime_error("rasterize: point_opacity must be non-null when num_points > 0");
  if (num_verts == 0)
    throw std::runtime_error("rasterize: empty verts");
  if (width == 0 || height == 0)
    throw std::runtime_error("rasterize: width/height must be > 0");

  // Extract booleans from the render_options array.
  const bool white_bg         = render_options[OPT_WHITE_BG]         != 0;
  const bool bresen_lines     = render_options[OPT_BRESEN_LINES]     != 0;
  const bool cull_backface    = render_options[OPT_CULL_BACKFACE]    != 0;

  // Line prim_type for the push constant: 1 = bresen (LINE_LIST), 2 = quads.
  const uint32_t line_prim_type = bresen_lines ? 1u : 2u;

  VkContext& ctx = requireCtx();
  validateCudaSupport(is_cuda, ctx, "rasterize");

  VkDevice         device         = ctx.device;
  VkPhysicalDevice physicalDevice = ctx.physicalDevice;
  VkQueue          queue          = ctx.queue;
  VkCommandPool    cmdPool        = ctx.cmdPool;

  CudaVkSyncGuard cuda(device, is_cuda);

  // Lazily create global cache
  if (!fwd_) fwd_ = std::make_unique<ForwardCache>();
  if (!shaders_) { shaders_ = std::make_unique<ShaderCache>(); shaders_->ensureLoaded(device); }

  // When topology pointers are null but count > 0, use cached buffers.
  if (num_faces > 0 && !faces) {
    if (!fwd_->valid || fwd_->num_faces != num_faces)
      throw std::runtime_error("rasterize: faces is null with num_faces > 0 but no matching cached face buffer");
  }
  if (num_lines > 0 && !lines) {
    if (!fwd_->valid || fwd_->num_lines != num_lines)
      throw std::runtime_error("rasterize: lines is null with num_lines > 0 but no matching cached line buffer");
  }
  if (num_points > 0 && !points) {
    if (!fwd_->valid || fwd_->num_points != num_points)
      throw std::runtime_error("rasterize: points is null with num_points > 0 but no matching cached point buffer");
  }

  const VkDeviceSize vboAttrBytes  = VkDeviceSize(num_verts) * 7 * sizeof(float);
  const VkDeviceSize iboBytes      = safeSize(VkDeviceSize(num_faces) * 3 * sizeof(uint32_t));
  const VkDeviceSize fopBytes      = safeSize(VkDeviceSize(num_faces) * sizeof(float));
  const VkDeviceSize lineIboBytes  = safeSize(VkDeviceSize(num_lines) * 2 * sizeof(uint32_t));
  const VkDeviceSize lopBytes      = safeSize(VkDeviceSize(num_lines) * sizeof(float));
  const VkDeviceSize pointIboBytes = safeSize(VkDeviceSize(num_points) * sizeof(uint32_t));
  const VkDeviceSize popBytes      = safeSize(VkDeviceSize(num_points) * sizeof(float));

  const VkDeviceSize primBytes    = VkDeviceSize(width) * VkDeviceSize(height) * sizeof(uint32_t);
  const VkDeviceSize bdBytes      = VkDeviceSize(width) * VkDeviceSize(height) * 4 * sizeof(float);
  const VkDeviceSize rgbaBytes    = VkDeviceSize(width) * VkDeviceSize(height) * 4 * sizeof(float);

  const VkMemoryPropertyFlags ioProps =
      is_cuda
        ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        : (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  // Reusable buffers (inputs + reuse in backward)
  fwd_->vboAttr.ensure(device, physicalDevice, vboAttrBytes,
                       VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                       ioProps, is_cuda);
  fwd_->ibo.ensure(device, physicalDevice, iboBytes,
                   VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                   ioProps, is_cuda);
  fwd_->faceOpacity.ensure(device, physicalDevice, fopBytes,
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, ioProps, is_cuda);
  fwd_->lineIbo.ensure(device, physicalDevice, lineIboBytes,
                       VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                       ioProps, is_cuda);
  fwd_->lineOpacity.ensure(device, physicalDevice, lopBytes,
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, ioProps, is_cuda);
  fwd_->pointIbo.ensure(device, physicalDevice, pointIboBytes,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, ioProps, is_cuda);
  fwd_->pointOpacity.ensure(device, physicalDevice, popBytes,
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, ioProps, is_cuda);
  // UBO stays host-visible (always uploaded from the CPU)
  fwd_->ubo.ensure(device, physicalDevice, sizeof(UBO),
                   VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                   /*export_to_cuda=*/false);

  // Readbacks are also STORAGE so backward can reuse without uploading.
  fwd_->primIdReadback.ensure(device, physicalDevice, primBytes,
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              ioProps, is_cuda);
  fwd_->bdReadback.ensure(device, physicalDevice, bdBytes,
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          ioProps, is_cuda);
  fwd_->rgbaReadback.ensure(device, physicalDevice, rgbaBytes,
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            ioProps, is_cuda);

  // Upload pointer inputs
  {
    fwd_->vboAttr.upload(device, vert_attrs, vboAttrBytes, is_cuda, "fwd vboAttr.upload");
    if (num_faces > 0) {
      if (faces)
        fwd_->ibo.upload(device, faces, VkDeviceSize(num_faces) * 3 * sizeof(uint32_t), is_cuda, "fwd ibo.upload");
      fwd_->faceOpacity.upload(device, face_opacity, VkDeviceSize(num_faces) * sizeof(float), is_cuda, "fwd faceOpacity.upload");
    }
    if (num_lines > 0) {
      if (lines)
        fwd_->lineIbo.upload(device, lines, VkDeviceSize(num_lines) * 2 * sizeof(uint32_t), is_cuda, "fwd lineIbo.upload");
      fwd_->lineOpacity.upload(device, line_opacity, VkDeviceSize(num_lines) * sizeof(float), is_cuda, "fwd lineOpacity.upload");
    }
    if (num_points > 0) {
      if (points)
        fwd_->pointIbo.upload(device, points, VkDeviceSize(num_points) * sizeof(uint32_t), is_cuda, "fwd pointIbo.upload");
      fwd_->pointOpacity.upload(device, point_opacity, VkDeviceSize(num_points) * sizeof(float), is_cuda, "fwd pointOpacity.upload");
    }

    UBO u{};
    // Convert viewproj from row-major to column-major for GLSL mat4.
    for (int r = 0; r < 4; ++r)
      for (int c = 0; c < 4; ++c)
        u.viewproj[c * 4 + r] = viewproj[r * 4 + c];
    u.campos[0] = campos[0];
    u.campos[1] = campos[1];
    u.campos[2] = campos[2];
    u.tau = tau;
    u.seed = seed;
    u._pad0 = 0u;
    u._pad1 = 0u;
    u._pad2 = 0u;
    fwd_->ubo.upload(device, &u, sizeof(UBO), /*is_cuda=*/false, "fwd ubo.upload");
  }

  cuda.signalCudaToVk();

  // Record forward cache metadata (used by backward)
  fwd_->valid     = true;
  fwd_->width     = width;
  fwd_->height    = height;
  fwd_->num_verts = num_verts;
  fwd_->num_faces  = num_faces;
  fwd_->num_lines  = num_lines;
  fwd_->num_points = num_points;
  memcpy(fwd_->render_options, render_options, NUM_RENDER_OPTIONS);

  // ----- Per-call Vulkan objects (RAII cleanup via scope guard) -----
  VkRenderPass          renderPass     = VK_NULL_HANDLE;
  VkFramebuffer         framebuffer    = VK_NULL_HANDLE;
  VkPipelineLayout      pipelineLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout setLayout      = VK_NULL_HANDLE;
  VkDescriptorPool      descPool       = VK_NULL_HANDLE;
  VkPipeline            pipelineFace   = VK_NULL_HANDLE;
  VkPipeline            pipelineLine   = VK_NULL_HANDLE;
  VkPipeline            pipelinePoint  = VK_NULL_HANDLE;

  auto cleanup = scopeGuard([&] {
    if (pipelineFace)   vkDestroyPipeline(device, pipelineFace, nullptr);
    if (pipelineLine)   vkDestroyPipeline(device, pipelineLine, nullptr);
    if (pipelinePoint)  vkDestroyPipeline(device, pipelinePoint, nullptr);
    if (framebuffer)    vkDestroyFramebuffer(device, framebuffer, nullptr);
    if (renderPass)     vkDestroyRenderPass(device, renderPass, nullptr);
    if (descPool)       vkDestroyDescriptorPool(device, descPool, nullptr);
    if (pipelineLayout) vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    if (setLayout)      vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
  });

  // ----- Render targets (cached in ForwardCache, recreated only on resolution change) -----
  if (fwd_->imgWidth != width || fwd_->imgHeight != height) {
    destroyImage(device, fwd_->depthImg);
    destroyImage(device, fwd_->rgbaImg);
    destroyImage(device, fwd_->baryDepthImg);
    destroyImage(device, fwd_->idsImg);

    createImage2D(device, physicalDevice, width, height, VK_FORMAT_R32_UINT,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT, fwd_->idsImg);
    createImage2D(device, physicalDevice, width, height, VK_FORMAT_R32G32B32A32_SFLOAT,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT, fwd_->baryDepthImg);
    createImage2D(device, physicalDevice, width, height, VK_FORMAT_R32G32B32A32_SFLOAT,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT, fwd_->rgbaImg);
    createImage2D(device, physicalDevice, width, height, VK_FORMAT_D32_SFLOAT,
                  VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                  VK_IMAGE_ASPECT_DEPTH_BIT, fwd_->depthImg);
    fwd_->imgWidth = width;
    fwd_->imgHeight = height;
  }

  // ----- Render pass + framebuffer -----
  {
    VkAttachmentDescription atts[4]{};
    // primId
    atts[0].format         = fwd_->idsImg.format;
    atts[0].samples        = VK_SAMPLE_COUNT_1_BIT;
    atts[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    atts[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    atts[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    atts[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    atts[0].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    atts[0].finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    // baryDepth
    atts[1] = atts[0];
    atts[1].format = fwd_->baryDepthImg.format;
    // rgba
    atts[2] = atts[0];
    atts[2].format = fwd_->rgbaImg.format;
    // depth
    atts[3] = atts[0];
    atts[3].format      = fwd_->depthImg.format;
    atts[3].storeOp     = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    atts[3].finalLayout  = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRefs[3]{};
    colorRefs[0].attachment = 0;
    colorRefs[0].layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorRefs[1].attachment = 1;
    colorRefs[1].layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorRefs[2].attachment = 2;
    colorRefs[2].layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthRef{};
    depthRef.attachment = 3;
    depthRef.layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount    = 3;
    subpass.pColorAttachments       = colorRefs;
    subpass.pDepthStencilAttachment = &depthRef;

    VkSubpassDependency dep{};
    dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass    = 0;
    dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dep.dstStageMask  = dep.srcStageMask;
    dep.srcAccessMask = 0;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpci.attachmentCount = 4;
    rpci.pAttachments    = atts;
    rpci.subpassCount    = 1;
    rpci.pSubpasses      = &subpass;
    rpci.dependencyCount = 1;
    rpci.pDependencies   = &dep;
    vkCheck(vkCreateRenderPass(device, &rpci, nullptr, &renderPass), "vkCreateRenderPass");

    VkImageView views[4]{fwd_->idsImg.view, fwd_->baryDepthImg.view, fwd_->rgbaImg.view, fwd_->depthImg.view};
    VkFramebufferCreateInfo fbci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fbci.renderPass      = renderPass;
    fbci.attachmentCount = 4;
    fbci.pAttachments    = views;
    fbci.width           = width;
    fbci.height          = height;
    fbci.layers          = 1;
    vkCheck(vkCreateFramebuffer(device, &fbci, nullptr, &framebuffer), "vkCreateFramebuffer");
  }

  // ----- Descriptor set layout + pipeline layout -----
  // binding 0: UBO, binding 1: face_opacity SSBO, binding 2: line_opacity SSBO,
  // binding 3: lines SSBO, binding 4: vert_attrs SSBO,
  // binding 5: point_opacity SSBO, binding 12: points SSBO
  {
    VkDescriptorSetLayoutBinding bindings[7]{};
    bindings[0].binding         = 0;
    bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[1].binding         = 1;
    bindings[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[2].binding         = 2;
    bindings[2].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[3].binding         = 3;
    bindings[3].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[4].binding         = 4;
    bindings[4].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[4].descriptorCount = 1;
    bindings[4].stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[5].binding         = 5;
    bindings[5].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[5].descriptorCount = 1;
    bindings[5].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[6].binding         = 12;
    bindings[6].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[6].descriptorCount = 1;
    bindings[6].stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo dsli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dsli.bindingCount = 7;
    dsli.pBindings    = bindings;
    vkCheck(vkCreateDescriptorSetLayout(device, &dsli, nullptr, &setLayout), "vkCreateDescriptorSetLayout");

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcr.offset     = 0;
    pcr.size       = sizeof(RasterizePushConstants);

    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &setLayout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    vkCheck(vkCreatePipelineLayout(device, &plci, nullptr, &pipelineLayout), "vkCreatePipelineLayout");
  }

  // ----- Descriptor pool + set -----
  VkDescriptorSet descSet = VK_NULL_HANDLE;
  {
    VkDescriptorPoolSize poolSizes[2]{};
    poolSizes[0].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = 1;
    poolSizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = 6;

    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 2;
    dpci.pPoolSizes    = poolSizes;
    vkCheck(vkCreateDescriptorPool(device, &dpci, nullptr, &descPool), "vkCreateDescriptorPool");

    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool     = descPool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &setLayout;
    vkCheck(vkAllocateDescriptorSets(device, &dsai, &descSet), "vkAllocateDescriptorSets");

    VkDescriptorBufferInfo uboInfo{};
    uboInfo.buffer = fwd_->ubo.buf.buf;
    uboInfo.range  = sizeof(UBO);

    VkDescriptorBufferInfo faceInfo{};
    faceInfo.buffer = fwd_->faceOpacity.buf.buf;
    faceInfo.range  = fwd_->faceOpacity.buf.size;

    VkDescriptorBufferInfo lineOpInfo{};
    lineOpInfo.buffer = fwd_->lineOpacity.buf.buf;
    lineOpInfo.range  = fwd_->lineOpacity.buf.size;

    VkDescriptorBufferInfo linesInfo{};
    linesInfo.buffer = fwd_->lineIbo.buf.buf;
    linesInfo.range  = fwd_->lineIbo.buf.size;

    VkDescriptorBufferInfo vattrInfo{};
    vattrInfo.buffer = fwd_->vboAttr.buf.buf;
    vattrInfo.range  = fwd_->vboAttr.buf.size;

    VkDescriptorBufferInfo pointOpInfo{};
    pointOpInfo.buffer = fwd_->pointOpacity.buf.buf;
    pointOpInfo.range  = fwd_->pointOpacity.buf.size;

    VkDescriptorBufferInfo pointsInfo{};
    pointsInfo.buffer = fwd_->pointIbo.buf.buf;
    pointsInfo.range  = fwd_->pointIbo.buf.size;

    VkWriteDescriptorSet writes[7]{};
    // UBO binding (UNIFORM_BUFFER)
    writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet          = descSet;
    writes[0].dstBinding      = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[0].pBufferInfo     = &uboInfo;
    // SSBO bindings
    writes[1] = makeStorageWrite(descSet, 1, &faceInfo);
    writes[2] = makeStorageWrite(descSet, 2, &lineOpInfo);
    writes[3] = makeStorageWrite(descSet, 3, &linesInfo);
    writes[4] = makeStorageWrite(descSet, 4, &vattrInfo);
    writes[5] = makeStorageWrite(descSet, 5, &pointOpInfo);
    writes[6] = makeStorageWrite(descSet, 12, &pointsInfo);

    vkUpdateDescriptorSets(device, 7, writes, 0, nullptr);
  }

  // ----- Shaders + face/line pipelines -----
  {
    // Use cached shader modules (loaded once, never destroyed until shutdown).
    VkShaderModule vs          = shaders_->rasterizeVert;
    VkShaderModule lineVs      = shaders_->lineQuadsVert;
    VkShaderModule lineBresenVs = shaders_->lineBresenVert;
    VkShaderModule fs          = shaders_->rasterizeFrag;

    VkPipelineShaderStageCreateInfo vsStage{};
    vsStage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vsStage.stage  = VK_SHADER_STAGE_VERTEX_BIT;
    vsStage.module = vs;
    vsStage.pName  = "main";

    VkPipelineShaderStageCreateInfo fsStage{};
    fsStage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fsStage.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    fsStage.module = fs;
    fsStage.pName  = "main";

    GraphicsPipelineCommonState pipeState;
    pipeState.init(width, height);

    VkPipelineDepthStencilStateCreateInfo dssi{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    dssi.depthTestEnable  = VK_TRUE;
    dssi.depthWriteEnable = VK_TRUE;
    dssi.depthCompareOp   = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState cbAtts[3]{};
    cbAtts[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT; // primId in R32_UINT
    cbAtts[1].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                               VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT; // baryDepth
    cbAtts[2].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                               VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT; // rgba

    VkPipelineColorBlendStateCreateInfo cbsi{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cbsi.attachmentCount = 3;
    cbsi.pAttachments    = cbAtts;

    // Face pipeline - optionally cull back-faces (CW winding).
    if (num_faces > 0) {
      if (cull_backface)
        pipeState.rsi.cullMode = VK_CULL_MODE_BACK_BIT;
      VkPipelineShaderStageCreateInfo faceStages[2] = {vsStage, fsStage};
      pipelineFace = createTopologyPipeline(device, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                                            faceStages, 2, pipeState, dssi, cbsi, pipelineLayout, renderPass,
                                            "vkCreateGraphicsPipelines(face)");
      pipeState.rsi.cullMode = VK_CULL_MODE_NONE;  // restore for line pipeline
    }

    // Line pipeline - either 1-px Bresenham (LINE_LIST) or instanced quads (TRIANGLE_LIST).
    if (num_lines > 0) {
      auto savedVisi = pipeState.visi;
      VkPipelineVertexInputStateCreateInfo emptyVisi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
      pipeState.visi = emptyVisi;

      if (bresen_lines) {
        VkPipelineShaderStageCreateInfo lineBresenVsStage{};
        lineBresenVsStage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        lineBresenVsStage.stage  = VK_SHADER_STAGE_VERTEX_BIT;
        lineBresenVsStage.module = lineBresenVs;
        lineBresenVsStage.pName  = "main";
        VkPipelineShaderStageCreateInfo lineStages[2] = {lineBresenVsStage, fsStage};
        pipelineLine = createTopologyPipeline(device, VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
                                              lineStages, 2, pipeState, dssi, cbsi, pipelineLayout, renderPass,
                                              "vkCreateGraphicsPipelines(line bresen)");
      } else {
        VkPipelineShaderStageCreateInfo lineVsStage{};
        lineVsStage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        lineVsStage.stage  = VK_SHADER_STAGE_VERTEX_BIT;
        lineVsStage.module = lineVs;
        lineVsStage.pName  = "main";
        VkPipelineShaderStageCreateInfo lineStages[2] = {lineVsStage, fsStage};
        pipelineLine = createTopologyPipeline(device, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                                              lineStages, 2, pipeState, dssi, cbsi, pipelineLayout, renderPass,
                                              "vkCreateGraphicsPipelines(line)");
      }
      pipeState.visi = savedVisi;
    }

    // Point pipeline - POINT_LIST, no VBO input (reads SSBOs like Bresenham lines).
    if (num_points > 0) {
      auto savedVisi = pipeState.visi;
      VkPipelineVertexInputStateCreateInfo emptyVisi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
      pipeState.visi = emptyVisi;

      VkPipelineShaderStageCreateInfo pointVsStage{};
      pointVsStage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      pointVsStage.stage  = VK_SHADER_STAGE_VERTEX_BIT;
      pointVsStage.module = shaders_->pointVert;
      pointVsStage.pName  = "main";
      VkPipelineShaderStageCreateInfo pointStages[2] = {pointVsStage, fsStage};
      pipelinePoint = createTopologyPipeline(device, VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
                                             pointStages, 2, pipeState, dssi, cbsi, pipelineLayout, renderPass,
                                             "vkCreateGraphicsPipelines(point)");
      pipeState.visi = savedVisi;
    }
  }

  // ----- Record commands -----
  VkCommandBuffer cmd = cmdBegin(device, cmdPool);

  // Transition color attachments to renderable
  for (auto* img : {&fwd_->idsImg, &fwd_->baryDepthImg, &fwd_->rgbaImg}) {
    cmdImageBarrier(cmd, img->img,
                    VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT);
  }

  // Begin render pass
  {
    VkClearValue clears[4]{};
    clears[0].color.uint32[0] = 0xFFFFFFFFu;  // primId: invalid
    clears[1].color.float32[2] = 1.0f;        // baryDepth.z = far-plane depth
    if (white_bg) {
      clears[2].color.float32[0] = 1.0f;
      clears[2].color.float32[1] = 1.0f;
      clears[2].color.float32[2] = 1.0f;
    }
    // Background alpha is padded to 1; the MSAA filter reads RGB only.
    clears[2].color.float32[3] = 1.0f;
    clears[3].depthStencil.depth   = 1.0f;
    clears[3].depthStencil.stencil = 0;

    VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rbi.renderPass      = renderPass;
    rbi.framebuffer     = framebuffer;
    rbi.renderArea.offset.x      = 0;
    rbi.renderArea.offset.y      = 0;
    rbi.renderArea.extent.width  = width;
    rbi.renderArea.extent.height = height;
    rbi.clearValueCount = 4;
    rbi.pClearValues    = clears;
    vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
  }

  {
    VkDeviceSize vbOffset = 0;
    VkBuffer vbBuf = fwd_->vboAttr.buf.buf;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vbBuf, &vbOffset);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descSet, 0, nullptr);
  }

  // Face draw
  if (num_faces > 0) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineFace);
    vkCmdBindIndexBuffer(cmd, fwd_->ibo.buf.buf, 0, VK_INDEX_TYPE_UINT32);
    RasterizePushConstants push{width, height, 0u, num_lines};
    vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
    vkCmdDrawIndexed(cmd, num_faces * 3, 1, 0, 0, 0);
  }

  // Line draw - 1-px Bresenham (2 verts/line) or instanced quads (6 verts/line)
  if (num_lines > 0) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLine);
    RasterizePushConstants push{width, height, line_prim_type, num_lines};
    vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
    if (bresen_lines)
      vkCmdDraw(cmd, num_lines * 2, 1, 0, 0);
    else
      vkCmdDraw(cmd, 6, num_lines, 0, 0);
  }

  // Point draw - 1 vertex per point, POINT_LIST
  if (num_points > 0) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelinePoint);
    RasterizePushConstants push{width, height, 3u, num_lines};
    vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
    vkCmdDraw(cmd, num_points, 1, 0, 0);
  }

  vkCmdEndRenderPass(cmd);

  // ----- Readback: transition + copy to readback buffers -----
  for (auto* img : {&fwd_->idsImg, &fwd_->baryDepthImg, &fwd_->rgbaImg}) {
    cmdImageBarrier(cmd, img->img,
                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT);
  }

  {
    VkBufferImageCopy bic{};
    bic.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    bic.imageSubresource.mipLevel       = 0;
    bic.imageSubresource.baseArrayLayer = 0;
    bic.imageSubresource.layerCount     = 1;
    bic.imageExtent.width  = width;
    bic.imageExtent.height = height;
    bic.imageExtent.depth  = 1;

    vkCmdCopyImageToBuffer(cmd, fwd_->idsImg.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           fwd_->primIdReadback.buf.buf, 1, &bic);
    vkCmdCopyImageToBuffer(cmd, fwd_->baryDepthImg.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           fwd_->bdReadback.buf.buf, 1, &bic);
    vkCmdCopyImageToBuffer(cmd, fwd_->rgbaImg.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           fwd_->rgbaReadback.buf.buf, 1, &bic);
  }

  // Make transfer writes visible for readback
  {
    VkAccessFlags        dstAccess = is_cuda ? (VkAccessFlags)0                     : VK_ACCESS_HOST_READ_BIT;
    VkPipelineStageFlags dstStage  = is_cuda ? VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT : VK_PIPELINE_STAGE_HOST_BIT;

    VkBufferMemoryBarrier bmb[3]{};
    VkBuffer      bufs[3]  = {fwd_->primIdReadback.buf.buf, fwd_->bdReadback.buf.buf, fwd_->rgbaReadback.buf.buf};
    VkDeviceSize sizes[3]  = {primBytes, bdBytes, rgbaBytes};
    for (int i = 0; i < 3; ++i) {
      bmb[i].sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
      bmb[i].srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
      bmb[i].dstAccessMask       = dstAccess;
      bmb[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      bmb[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      bmb[i].buffer              = bufs[i];
      bmb[i].size                = sizes[i];
    }
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, dstStage,
                         0, 0, nullptr, 3, bmb, 0, nullptr);
  }

  // ----- Submit + readback -----
  {
    SubmitParams sp{};
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT;
    cuda.fillSubmit(sp, waitStage);
    submitAndWait(device, queue, cmdPool, cmd, sp, "vkCreateFence(rasterize fwd)");
  }

  cuda.waitVkToCuda();

  // prim_id and bary_depth downloads are skipped when the caller passes nullptr
  // (data stays on the GPU buffer for reuse by backward passes).
  if (prim_id_out)
    fwd_->primIdReadback.download(device, prim_id_out, primBytes, is_cuda, "primIdReadback.download");
  if (bary_depth_out)
    fwd_->bdReadback.download(device, bary_depth_out, bdBytes, is_cuda, "bdReadback.download");
  fwd_->rgbaReadback.download(device, rgba_out, rgbaBytes, is_cuda, "rgbaReadback.download");

  cuda.syncStream();

  return 0;
  // scope guard destroys per-call Vulkan objects; CudaVkSyncGuard destroys semaphores
}

// ===============================
// Grad: edge_grad
// ===============================

int Rasterizer::edge_grad(
    uint32_t num_verts,
    uint32_t num_faces,
    uint32_t num_lines,
    uint32_t num_points,
    uint32_t width,
    uint32_t height,
    const float* grad_out_rgba,        // [height,width,4]
    float* grad_vert_attrs,            // [num_verts,7]
    bool is_cuda /*= false*/
) {
  if (!grad_out_rgba || !grad_vert_attrs)
    throw std::runtime_error("edge_grad: null pointer input");
  if (num_faces == 0 && num_lines == 0 && num_points == 0)
    throw std::runtime_error("edge_grad: at least one of num_faces, num_lines, or num_points must be > 0");
  if (num_verts == 0)
    throw std::runtime_error("edge_grad: empty verts");
  if (width == 0 || height == 0)
    throw std::runtime_error("edge_grad: width/height must be > 0");

  VkContext& ctx = requireCtx();
  validateCudaSupport(is_cuda, ctx, "edge_grad");

  VkDevice         device         = ctx.device;
  VkPhysicalDevice physicalDevice = ctx.physicalDevice;
  VkQueue          queue          = ctx.queue;
  VkCommandPool    cmdPool        = ctx.cmdPool;

  CudaVkSyncGuard cuda(device, is_cuda);

  // Forward cache validation
  if (!fwd_) throw std::runtime_error("edge_grad: forward cache missing (run forward first)");
  fwd_->requireMatch(width, height, "edge_grad");
  fwd_->requireMesh(num_verts, num_faces, num_lines, num_points, "edge_grad");

  const VkDeviceSize nPix      = VkDeviceSize(width) * VkDeviceSize(height);
  const VkDeviceSize primBytes = nPix * sizeof(uint32_t);
  const VkDeviceSize baryBytes = nPix * 4 * sizeof(float);
  const VkDeviceSize rgbaBytes = nPix * 4 * sizeof(float);
  const VkDeviceSize faceBytes = safeSize(VkDeviceSize(num_faces) * 3 * sizeof(uint32_t));
  const VkDeviceSize vertBytes = VkDeviceSize(num_verts) * 7 * sizeof(float);
  const VkDeviceSize lineBytes  = safeSize(VkDeviceSize(num_lines) * 2 * sizeof(uint32_t));
  const VkDeviceSize pointBytes = safeSize(VkDeviceSize(num_points) * sizeof(uint32_t));
  const VkDeviceSize gOutBytes  = rgbaBytes;
  const VkDeviceSize gVertBytes = vertBytes;

  fwd_->primIdReadback.requireExact(primBytes, "edge_grad prim_id");
  fwd_->bdReadback.requireExact(baryBytes, "edge_grad bary_depth");
  fwd_->rgbaReadback.requireExact(rgbaBytes, "edge_grad rgba");
  fwd_->ibo.requireExact(faceBytes, "edge_grad faces");
  fwd_->vboAttr.requireExact(vertBytes, "edge_grad vert_attrs");
  fwd_->lineIbo.requireExact(lineBytes, "edge_grad lines");
  fwd_->pointIbo.requireExact(pointBytes, "edge_grad points");

  // Ensure + upload grad buffers
  {
    if (!edge_) edge_ = std::make_unique<EdgeGradCache>();

    const VkMemoryPropertyFlags ioProps =
        is_cuda ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                : (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    edge_->gOutBuf.ensure(device, physicalDevice, gOutBytes,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, ioProps, is_cuda);
    edge_->gVertBuf.ensure(device, physicalDevice, gVertBytes,
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, ioProps, is_cuda);

    edge_->gOutBuf.upload(device, grad_out_rgba, gOutBytes, is_cuda, "edge gOutBuf.upload");
    edge_->gVertBuf.memset(device, 0, gVertBytes, is_cuda, "edge gVertBuf.memset");
  }

  cuda.signalCudaToVk();

  // ----- Per-call Vulkan objects -----
  VkDescriptorSetLayout setLayout      = VK_NULL_HANDLE;
  VkPipelineLayout      pipelineLayout = VK_NULL_HANDLE;
  VkDescriptorPool      descPool       = VK_NULL_HANDLE;
  VkPipeline            pipe           = VK_NULL_HANDLE;

  auto cleanup = scopeGuard([&] {
    if (pipe)           vkDestroyPipeline(device, pipe, nullptr);
    if (descPool)       vkDestroyDescriptorPool(device, descPool, nullptr);
    if (pipelineLayout) vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    if (setLayout)      vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
  });

  // Descriptor layout + pipeline layout
  // 0 primId, 1 baryDepth, 2 faces, 3 vert_attrs, 4 out_rgba, 5 grad_out_rgba, 6 grad_vert_attrs, 7 lines, 8 UBO (viewproj), 9 points
  VkDescriptorSet ds = VK_NULL_HANDLE;
  {
    std::array<VkDescriptorSetLayoutBinding, 10> bindings{};
    for (uint32_t i = 0; i < 8; ++i) {
      bindings[i].binding         = i;
      bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[i].descriptorCount = 1;
      bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    // binding 8: UBO (viewproj from forward cache)
    bindings[8].binding         = 8;
    bindings[8].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[8].descriptorCount = 1;
    bindings[8].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    // binding 9: points SSBO
    bindings[9].binding         = 9;
    bindings[9].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[9].descriptorCount = 1;
    bindings[9].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo dsli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dsli.bindingCount = 10;
    dsli.pBindings    = bindings.data();
    vkCheck(vkCreateDescriptorSetLayout(device, &dsli, nullptr, &setLayout),
            "vkCreateDescriptorSetLayout(edge_grad)");

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset     = 0;
    pcr.size       = sizeof(EdgeGradPushConstants);

    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &setLayout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    vkCheck(vkCreatePipelineLayout(device, &plci, nullptr, &pipelineLayout),
            "vkCreatePipelineLayout(edge_grad)");

    VkDescriptorPoolSize poolSizes[2]{};
    poolSizes[0].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[0].descriptorCount = 9;
    poolSizes[1].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[1].descriptorCount = 1;

    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 2;
    dpci.pPoolSizes    = poolSizes;
    vkCheck(vkCreateDescriptorPool(device, &dpci, nullptr, &descPool),
            "vkCreateDescriptorPool(edge_grad)");

    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool     = descPool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &setLayout;
    vkCheck(vkAllocateDescriptorSets(device, &dsai, &ds), "vkAllocateDescriptorSets(edge_grad)");

    VkDescriptorBufferInfo primInfo{};
    primInfo.buffer = fwd_->primIdReadback.buf.buf;
    primInfo.range  = primBytes;

    VkDescriptorBufferInfo baryInfo{};
    baryInfo.buffer = fwd_->bdReadback.buf.buf;
    baryInfo.range  = baryBytes;

    VkDescriptorBufferInfo faceInfo{};
    faceInfo.buffer = fwd_->ibo.buf.buf;
    faceInfo.range  = faceBytes;

    VkDescriptorBufferInfo vertInfo{};
    vertInfo.buffer = fwd_->vboAttr.buf.buf;
    vertInfo.range  = vertBytes;

    VkDescriptorBufferInfo outInfo{};
    outInfo.buffer = fwd_->rgbaReadback.buf.buf;
    outInfo.range  = rgbaBytes;

    VkDescriptorBufferInfo gOutInfo{};
    gOutInfo.buffer = edge_->gOutBuf.buf.buf;
    gOutInfo.range  = gOutBytes;

    VkDescriptorBufferInfo gVertInfo{};
    gVertInfo.buffer = edge_->gVertBuf.buf.buf;
    gVertInfo.range  = gVertBytes;

    VkDescriptorBufferInfo lineInfo{};
    lineInfo.buffer = fwd_->lineIbo.buf.buf;
    lineInfo.range  = lineBytes;

    VkDescriptorBufferInfo pointInfo{};
    pointInfo.buffer = fwd_->pointIbo.buf.buf;
    pointInfo.range  = pointBytes;

    VkDescriptorBufferInfo* infos[9] = {&primInfo, &baryInfo, &faceInfo, &vertInfo,
                                        &outInfo, &gOutInfo, &gVertInfo, &lineInfo,
                                        &pointInfo};
    std::array<VkWriteDescriptorSet, 10> writes{};
    for (uint32_t i = 0; i < 8; ++i)
      writes[i] = makeStorageWrite(ds, i, infos[i]);

    // binding 8: UBO (viewproj from forward cache)
    VkDescriptorBufferInfo uboInfo{};
    uboInfo.buffer = fwd_->ubo.buf.buf;
    uboInfo.range  = sizeof(UBO);
    writes[8].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[8].dstSet          = ds;
    writes[8].dstBinding      = 8;
    writes[8].descriptorCount = 1;
    writes[8].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[8].pBufferInfo     = &uboInfo;

    // binding 9: points SSBO
    writes[9] = makeStorageWrite(ds, 9, infos[8]);

    vkUpdateDescriptorSets(device, 10, writes.data(), 0, nullptr);
  }

  // Compute pipeline
  {
    VkShaderModule sh = shaders_->edgeGradComp;  // cached

    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = sh;
    stage.pName  = "main";

    VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cp.stage  = stage;
    cp.layout = pipelineLayout;
    vkCheck(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp, nullptr, &pipe),
            "vkCreateComputePipelines(edge_grad)");
  }

  // ----- Record + dispatch -----
  VkCommandBuffer cmd = cmdBegin(device, cmdPool);

  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &ds, 0, nullptr);

  EdgeGradPushConstants push{width, height};
  vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
  vkCmdDispatch(cmd, ceilDiv(width, 16), ceilDiv(height, 16), 1);

  // Make gVert visible to consumer (CPU host read OR CUDA read)
  {
    VkBufferMemoryBarrier bmb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    bmb.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
    bmb.dstAccessMask       = is_cuda ? (VkAccessFlags)0 : VK_ACCESS_HOST_READ_BIT;
    bmb.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bmb.buffer              = edge_->gVertBuf.buf.buf;
    bmb.size                = gVertBytes;

    vkCmdPipelineBarrier(cmd,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      is_cuda ? VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT : VK_PIPELINE_STAGE_HOST_BIT,
      0, 0, nullptr, 1, &bmb, 0, nullptr);
  }

  // ----- Submit + download -----
  {
    SubmitParams sp{};
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    cuda.fillSubmit(sp, waitStage);
    submitAndWait(device, queue, cmdPool, cmd, sp, "vkCreateFence(edge_grad)");
  }

  cuda.waitVkToCuda();
  edge_->gVertBuf.download(device, grad_vert_attrs, gVertBytes, is_cuda, "edge gVertBuf.download");
  cuda.syncStream();

  return 0;
}

// ===============================
// Backward: opacity_grad
// ===============================

int Rasterizer::opacity_grad(
    uint32_t num_faces,
    uint32_t num_lines,
    uint32_t num_points,
    uint32_t width,
    uint32_t height,
    const float* error,               // [height,width]
    float* grad_face_opacity,         // [num_faces]  (may be null if num_faces==0)
    float* grad_line_opacity,         // [num_lines]  (may be null if num_lines==0)
    float* grad_point_opacity,        // [num_points] (may be null if num_points==0)
    float eps,
    bool is_cuda
) {
  if (!error)
    throw std::runtime_error("opacity_grad: null pointer input");
  if (num_faces == 0 && num_lines == 0 && num_points == 0)
    throw std::runtime_error("opacity_grad: at least one of num_faces, num_lines, or num_points must be > 0");
  if (num_faces > 0 && !grad_face_opacity)
    throw std::runtime_error("opacity_grad: grad_face_opacity must be non-null when num_faces > 0");
  if (num_lines > 0 && !grad_line_opacity)
    throw std::runtime_error("opacity_grad: grad_line_opacity must be non-null when num_lines > 0");
  if (num_points > 0 && !grad_point_opacity)
    throw std::runtime_error("opacity_grad: grad_point_opacity must be non-null when num_points > 0");
  if (width == 0 || height == 0)
    throw std::runtime_error("opacity_grad: width/height must be > 0");

  VkContext& ctx = requireCtx();
  validateCudaSupport(is_cuda, ctx, "opacity_grad");

  VkDevice         device         = ctx.device;
  VkPhysicalDevice physicalDevice = ctx.physicalDevice;
  VkQueue          queue          = ctx.queue;
  VkCommandPool    cmdPool        = ctx.cmdPool;

  CudaVkSyncGuard cuda(device, is_cuda);

  // Forward cache validation
  if (!fwd_) throw std::runtime_error("opacity_grad: forward cache missing (run forward first)");
  fwd_->requireMatch(width, height, "opacity_grad");
  // Validate face/line/point counts match the forward cache.
  if (!fwd_->valid || fwd_->num_faces != num_faces || fwd_->num_lines != num_lines || fwd_->num_points != num_points) {
    throw std::runtime_error("opacity_grad: forward cache inconsistent (mesh size mismatch)");
  }

  // Reproduce the pipeline configuration from the forward pass.
  const bool bresen_lines     = fwd_->render_options[OPT_BRESEN_LINES]     != 0;
  const bool cull_backface    = fwd_->render_options[OPT_CULL_BACKFACE]    != 0;

  const uint32_t line_prim_type = bresen_lines ? 1u : 2u;

  const uint32_t num_verts = fwd_->num_verts;

  const VkDeviceSize nPix      = VkDeviceSize(width) * VkDeviceSize(height);
  const VkDeviceSize primBytes = nPix * sizeof(uint32_t);
  const VkDeviceSize bdBytes   = nPix * 4 * sizeof(float);
  const VkDeviceSize errBytes  = nPix * sizeof(float);
  const VkDeviceSize faceBytes = safeSize(VkDeviceSize(num_faces) * 3 * sizeof(uint32_t));
  const VkDeviceSize fopBytes  = safeSize(VkDeviceSize(num_faces) * sizeof(float));
  const VkDeviceSize lineBytes = safeSize(VkDeviceSize(num_lines) * 2 * sizeof(uint32_t));
  const VkDeviceSize lopBytes  = safeSize(VkDeviceSize(num_lines) * sizeof(float));
  const VkDeviceSize attrBytes = VkDeviceSize(num_verts) * 7 * sizeof(float);
  const VkDeviceSize gfopBytes = safeSize(VkDeviceSize(num_faces) * sizeof(float));
  const VkDeviceSize glopBytes  = safeSize(VkDeviceSize(num_lines) * sizeof(float));
  const VkDeviceSize pointBytes = safeSize(VkDeviceSize(num_points) * sizeof(uint32_t));
  const VkDeviceSize popBytes   = safeSize(VkDeviceSize(num_points) * sizeof(float));
  const VkDeviceSize gpopBytes  = safeSize(VkDeviceSize(num_points) * sizeof(float));

  fwd_->primIdReadback.requireExact(primBytes, "opacity_grad prim_id");
  fwd_->bdReadback.requireExact(bdBytes, "opacity_grad bary_depth");
  fwd_->ibo.requireExact(faceBytes, "opacity_grad faces");
  fwd_->faceOpacity.requireExact(fopBytes, "opacity_grad face_opacity");
  fwd_->lineIbo.requireExact(lineBytes, "opacity_grad lines");
  fwd_->lineOpacity.requireExact(lopBytes, "opacity_grad line_opacity");
  fwd_->pointIbo.requireExact(pointBytes, "opacity_grad points");
  fwd_->pointOpacity.requireExact(popBytes, "opacity_grad point_opacity");
  fwd_->vboAttr.requireExact(attrBytes, "opacity_grad vert_attrs");

  // Ensure + upload grad buffers
  {
    if (!opac_) opac_ = std::make_unique<OpacityGradCache>();

    const VkMemoryPropertyFlags ioProps =
        is_cuda ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                : (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    opac_->errBuf.ensure(device, physicalDevice, errBytes,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, ioProps, is_cuda);
    opac_->gfopBuf.ensure(device, physicalDevice, gfopBytes,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, ioProps, is_cuda);
    opac_->glopBuf.ensure(device, physicalDevice, glopBytes,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, ioProps, is_cuda);
    opac_->gpopBuf.ensure(device, physicalDevice, gpopBytes,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, ioProps, is_cuda);

    opac_->errBuf.upload(device, error, errBytes, is_cuda, "opacity_grad errBuf.upload");
    opac_->gfopBuf.memset(device, 0, gfopBytes, is_cuda, "opacity_grad gfopBuf.memset");
    opac_->glopBuf.memset(device, 0, glopBytes, is_cuda, "opacity_grad glopBuf.memset");
    opac_->gpopBuf.memset(device, 0, gpopBytes, is_cuda, "opacity_grad gpopBuf.memset");
  }

  cuda.signalCudaToVk();

  // ----- Per-call Vulkan objects -----
  VkRenderPass          renderPass     = VK_NULL_HANDLE;
  VkFramebuffer         framebuffer    = VK_NULL_HANDLE;
  VkPipelineLayout      pipelineLayout = VK_NULL_HANDLE;
  VkDescriptorSetLayout setLayout      = VK_NULL_HANDLE;
  VkDescriptorPool      descPool       = VK_NULL_HANDLE;
  VkPipeline            pipelineFace      = VK_NULL_HANDLE;
  VkPipeline            pipelineLine      = VK_NULL_HANDLE;
  VkPipeline            pipelinePoint     = VK_NULL_HANDLE;

  auto cleanup = scopeGuard([&] {
    if (pipelineFace)     vkDestroyPipeline(device, pipelineFace,     nullptr);
    if (pipelineLine)     vkDestroyPipeline(device, pipelineLine,     nullptr);
    if (pipelinePoint)    vkDestroyPipeline(device, pipelinePoint,    nullptr);
    if (framebuffer)    vkDestroyFramebuffer(device, framebuffer, nullptr);
    if (renderPass)     vkDestroyRenderPass(device, renderPass, nullptr);
    if (descPool)       vkDestroyDescriptorPool(device, descPool, nullptr);
    if (pipelineLayout) vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    if (setLayout)      vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
  });

  // ----- Dummy render target (cached in OpacityGradCache) + render pass -----
  if (opac_->imgWidth != width || opac_->imgHeight != height) {
    destroyImage(device, opac_->dummyColor);
    VkFormat dummyFormat = VK_FORMAT_R32_UINT;
    createImage2D(device, physicalDevice, width, height, dummyFormat,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_IMAGE_ASPECT_COLOR_BIT, opac_->dummyColor);
    opac_->imgWidth = width;
    opac_->imgHeight = height;
  }
  {
    VkAttachmentDescription colorAtt{};
    colorAtt.format         = opac_->dummyColor.format;
    colorAtt.samples        = VK_SAMPLE_COUNT_1_BIT;
    colorAtt.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAtt.storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAtt.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAtt.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAtt.finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef{};
    colorRef.attachment = 0;
    colorRef.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments    = &colorRef;

    VkSubpassDependency dep{};
    dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass    = 0;
    dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo rpci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpci.attachmentCount = 1;
    rpci.pAttachments    = &colorAtt;
    rpci.subpassCount    = 1;
    rpci.pSubpasses      = &subpass;
    rpci.dependencyCount = 1;
    rpci.pDependencies   = &dep;
    vkCheck(vkCreateRenderPass(device, &rpci, nullptr, &renderPass), "vkCreateRenderPass(opacity_grad)");

    VkImageView views[] = {opac_->dummyColor.view};
    VkFramebufferCreateInfo fbci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fbci.renderPass      = renderPass;
    fbci.attachmentCount = 1;
    fbci.pAttachments    = views;
    fbci.width           = width;
    fbci.height          = height;
    fbci.layers          = 1;
    vkCheck(vkCreateFramebuffer(device, &fbci, nullptr, &framebuffer), "vkCreateFramebuffer(opacity_grad)");
  }

  // ----- Descriptor layout + pipeline layout + pool + set -----
  // 0 UBO,
  // 1 primId, 2 baryDepth,
  // 3 lines (line VS), 4 vert_attrs (line/point VS),
  // 5 face_opacity, 6 grad_face_opacity,
  // 7 error,
  // 8 line_opacity, 9 grad_line_opacity,
  // 10 point_opacity, 11 grad_point_opacity, 12 points (point VS)
  VkDescriptorSet descSet = VK_NULL_HANDLE;
  {
    std::array<VkDescriptorSetLayoutBinding, 13> bindings{};
    // binding 0: UBO
    bindings[0].binding         = 0;
    bindings[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    // bindings 1-12: SSBOs
    for (uint32_t i = 1; i <= 12; ++i) {
      bindings[i].binding         = i;
      bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[i].descriptorCount = 1;
      bindings[i].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    // Vertex-shader-read SSBOs: 3 lines, 4 vert_attrs, 12 points.
    bindings[3].stageFlags  = VK_SHADER_STAGE_VERTEX_BIT;
    bindings[4].stageFlags  = VK_SHADER_STAGE_VERTEX_BIT;
    bindings[12].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo dsli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dsli.bindingCount = 13;
    dsli.pBindings    = bindings.data();
    vkCheck(vkCreateDescriptorSetLayout(device, &dsli, nullptr, &setLayout),
            "vkCreateDescriptorSetLayout(opacity_grad)");

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcr.offset     = 0;
    pcr.size       = sizeof(OpacityGradPushConstants);

    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &setLayout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    vkCheck(vkCreatePipelineLayout(device, &plci, nullptr, &pipelineLayout),
            "vkCreatePipelineLayout(opacity_grad)");

    VkDescriptorPoolSize poolSizes[2]{};
    poolSizes[0].type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = 1;
    poolSizes[1].type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = 12;  // SSBOs: bindings 1-12

    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets       = 1;
    dpci.poolSizeCount = 2;
    dpci.pPoolSizes    = poolSizes;
    vkCheck(vkCreateDescriptorPool(device, &dpci, nullptr, &descPool),
            "vkCreateDescriptorPool(opacity_grad)");

    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool     = descPool;
    dsai.descriptorSetCount = 1;
    dsai.pSetLayouts        = &setLayout;
    vkCheck(vkAllocateDescriptorSets(device, &dsai, &descSet),
            "vkAllocateDescriptorSets(opacity_grad)");

    VkDescriptorBufferInfo uboInfo{};
    uboInfo.buffer = fwd_->ubo.buf.buf;
    uboInfo.range  = sizeof(UBO);

    VkDescriptorBufferInfo primInfo{};
    primInfo.buffer = fwd_->primIdReadback.buf.buf;
    primInfo.range  = primBytes;

    VkDescriptorBufferInfo bdInfo{};
    bdInfo.buffer = fwd_->bdReadback.buf.buf;
    bdInfo.range  = bdBytes;

    VkDescriptorBufferInfo lineInfo{};
    lineInfo.buffer = fwd_->lineIbo.buf.buf;
    lineInfo.range  = lineBytes;

    VkDescriptorBufferInfo vattrInfo{};
    vattrInfo.buffer = fwd_->vboAttr.buf.buf;
    vattrInfo.range  = attrBytes;

    VkDescriptorBufferInfo fopInfo{};
    fopInfo.buffer = fwd_->faceOpacity.buf.buf;
    fopInfo.range  = fopBytes;

    VkDescriptorBufferInfo gfopInfo{};
    gfopInfo.buffer = opac_->gfopBuf.buf.buf;
    gfopInfo.range  = gfopBytes;

    VkDescriptorBufferInfo errInfo{};
    errInfo.buffer = opac_->errBuf.buf.buf;
    errInfo.range  = errBytes;

    VkDescriptorBufferInfo lopInfo{};
    lopInfo.buffer = fwd_->lineOpacity.buf.buf;
    lopInfo.range  = lopBytes;

    VkDescriptorBufferInfo glopInfo{};
    glopInfo.buffer = opac_->glopBuf.buf.buf;
    glopInfo.range  = glopBytes;

    VkDescriptorBufferInfo popInfo{};
    popInfo.buffer = fwd_->pointOpacity.buf.buf;
    popInfo.range  = popBytes;

    VkDescriptorBufferInfo gpopInfo{};
    gpopInfo.buffer = opac_->gpopBuf.buf.buf;
    gpopInfo.range  = gpopBytes;

    VkDescriptorBufferInfo pointIdxInfo{};
    pointIdxInfo.buffer = fwd_->pointIbo.buf.buf;
    pointIdxInfo.range  = pointBytes;

    // binding 0: UBO
    std::array<VkWriteDescriptorSet, 13> writes{};
    writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet          = descSet;
    writes[0].dstBinding      = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[0].pBufferInfo     = &uboInfo;

    // bindings 1-12: SSBOs
    VkDescriptorBufferInfo* infos[12] = {
      &primInfo, &bdInfo, &lineInfo, &vattrInfo, &fopInfo,
      &gfopInfo, &errInfo, &lopInfo, &glopInfo,
      &popInfo, &gpopInfo, &pointIdxInfo
    };
    for (uint32_t i = 0; i < 12; ++i)
      writes[i + 1] = makeStorageWrite(descSet, i + 1, infos[i]);

    vkUpdateDescriptorSets(device, (uint32_t)writes.size(), writes.data(), 0, nullptr);
  }

  // ----- Shaders + face/line pipelines -----
  {
    // Use cached shader modules.
    VkShaderModule vs          = shaders_->rasterizeVert;
    VkShaderModule lineVs      = shaders_->lineQuadsVert;
    VkShaderModule lineBresenVs = shaders_->lineBresenVert;
    VkShaderModule fs          = shaders_->opacityGradFrag;

    VkPipelineShaderStageCreateInfo vsStage{};
    vsStage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vsStage.stage  = VK_SHADER_STAGE_VERTEX_BIT;
    vsStage.module = vs;
    vsStage.pName  = "main";

    VkPipelineShaderStageCreateInfo fsStage{};
    fsStage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fsStage.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    fsStage.module = fs;
    fsStage.pName  = "main";

    GraphicsPipelineCommonState pipeState;
    pipeState.init(width, height);

    VkPipelineDepthStencilStateCreateInfo dssi{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    // depth test OFF for opacity grad pass

    VkPipelineColorBlendAttachmentState cbAtt{};
    cbAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT;

    VkPipelineColorBlendStateCreateInfo cbsi{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cbsi.attachmentCount = 1;
    cbsi.pAttachments    = &cbAtt;

    // Face pipeline - mirror the forward cull mode.
    if (num_faces > 0) {
      if (cull_backface)
        pipeState.rsi.cullMode = VK_CULL_MODE_BACK_BIT;
      VkPipelineShaderStageCreateInfo faceStages[2] = {vsStage, fsStage};
      pipelineFace = createTopologyPipeline(device, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                                            faceStages, 2, pipeState, dssi, cbsi, pipelineLayout, renderPass,
                                            "vkCreateGraphicsPipelines(opacity_grad face)");
      pipeState.rsi.cullMode = VK_CULL_MODE_NONE;
    }

    // Line pipeline - mirror the forward Bresenham vs quad choice.
    if (num_lines > 0) {
      auto savedVisi = pipeState.visi;
      VkPipelineVertexInputStateCreateInfo emptyVisi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
      pipeState.visi = emptyVisi;

      if (bresen_lines) {
        VkPipelineShaderStageCreateInfo lineBresenVsStage{};
        lineBresenVsStage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        lineBresenVsStage.stage  = VK_SHADER_STAGE_VERTEX_BIT;
        lineBresenVsStage.module = lineBresenVs;
        lineBresenVsStage.pName  = "main";
        VkPipelineShaderStageCreateInfo lineStages[2] = {lineBresenVsStage, fsStage};
        pipelineLine = createTopologyPipeline(device, VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
                                              lineStages, 2, pipeState, dssi, cbsi, pipelineLayout, renderPass,
                                              "vkCreateGraphicsPipelines(opacity_grad line bresen)");
      } else {
        VkPipelineShaderStageCreateInfo lineVsStage{};
        lineVsStage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        lineVsStage.stage  = VK_SHADER_STAGE_VERTEX_BIT;
        lineVsStage.module = lineVs;
        lineVsStage.pName  = "main";
        VkPipelineShaderStageCreateInfo lineStages[2] = {lineVsStage, fsStage};
        pipelineLine = createTopologyPipeline(device, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                                              lineStages, 2, pipeState, dssi, cbsi, pipelineLayout, renderPass,
                                              "vkCreateGraphicsPipelines(opacity_grad line)");
      }
      pipeState.visi = savedVisi;
    }

    // Point pipeline - POINT_LIST, no VBO input (reads SSBOs).
    if (num_points > 0) {
      auto savedVisi = pipeState.visi;
      VkPipelineVertexInputStateCreateInfo emptyVisi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
      pipeState.visi = emptyVisi;

      VkPipelineShaderStageCreateInfo pointVsStage{};
      pointVsStage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      pointVsStage.stage  = VK_SHADER_STAGE_VERTEX_BIT;
      pointVsStage.module = shaders_->pointVert;
      pointVsStage.pName  = "main";
      VkPipelineShaderStageCreateInfo pointStages[2] = {pointVsStage, fsStage};
      pipelinePoint = createTopologyPipeline(device, VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
                                             pointStages, 2, pipeState, dssi, cbsi, pipelineLayout, renderPass,
                                             "vkCreateGraphicsPipelines(opacity_grad point)");
      pipeState.visi = savedVisi;
    }
  }

  // ----- Record commands -----
  VkCommandBuffer cmd = cmdBegin(device, cmdPool);

  cmdImageBarrier(cmd, opac_->dummyColor.img,
                  VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                  0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_IMAGE_ASPECT_COLOR_BIT);

  {
    VkClearValue clear{};
    VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rbi.renderPass      = renderPass;
    rbi.framebuffer     = framebuffer;
    rbi.renderArea.offset.x      = 0;
    rbi.renderArea.offset.y      = 0;
    rbi.renderArea.extent.width  = width;
    rbi.renderArea.extent.height = height;
    rbi.clearValueCount = 1;
    rbi.pClearValues    = &clear;
    vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
  }

  {
    VkDeviceSize vbOffset = 0;
    VkBuffer vbBuf = fwd_->vboAttr.buf.buf;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vbBuf, &vbOffset);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descSet, 0, nullptr);
  }

  // Face draw - opacity grad (mirrors forward cull mode)
  if (num_faces > 0) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineFace);
    vkCmdBindIndexBuffer(cmd, fwd_->ibo.buf.buf, 0, VK_INDEX_TYPE_UINT32);
    OpacityGradPushConstants push{width, height, 0u, eps};
    vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
    vkCmdDrawIndexed(cmd, num_faces * 3, 1, 0, 0, 0);
  }

  // Line draw - match the topology used in the forward pass.
  if (num_lines > 0) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLine);
    OpacityGradPushConstants push{width, height, line_prim_type, eps};
    vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
    if (bresen_lines)
      vkCmdDraw(cmd, num_lines * 2, 1, 0, 0);
    else
      vkCmdDraw(cmd, 6, num_lines, 0, 0);
  }

  // Point draw - POINT_LIST
  if (num_points > 0) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelinePoint);
    OpacityGradPushConstants push{width, height, 3u, eps};
    vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
    vkCmdDraw(cmd, num_points, 1, 0, 0);
  }

  vkCmdEndRenderPass(cmd);

  // Make shader writes visible to consumer (CPU host read OR CUDA read)
  {
    VkAccessFlags        dstAccess = is_cuda ? (VkAccessFlags)0                     : VK_ACCESS_HOST_READ_BIT;
    VkPipelineStageFlags dstStage  = is_cuda ? VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT : VK_PIPELINE_STAGE_HOST_BIT;

    const int numBarriers = 3;

    VkBufferMemoryBarrier bmb[3]{};
    VkBuffer      bufs[3]  = {opac_->gfopBuf.buf.buf, opac_->glopBuf.buf.buf, opac_->gpopBuf.buf.buf};
    VkDeviceSize sizes[3]  = {gfopBytes, glopBytes, gpopBytes};
    for (int i = 0; i < numBarriers; ++i) {
      bmb[i].sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
      bmb[i].srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
      bmb[i].dstAccessMask       = dstAccess;
      bmb[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      bmb[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      bmb[i].buffer              = bufs[i];
      bmb[i].size                = sizes[i];
    }
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, dstStage,
                         0, 0, nullptr, numBarriers, bmb, 0, nullptr);
  }

  // ----- Submit + download -----
  {
    SubmitParams sp{};
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    cuda.fillSubmit(sp, waitStage);
    submitAndWait(device, queue, cmdPool, cmd, sp, "vkCreateFence(opacity_grad)");
  }

  cuda.waitVkToCuda();

  if (num_faces > 0)
    opac_->gfopBuf.download(device, grad_face_opacity, VkDeviceSize(num_faces) * sizeof(float),
                            is_cuda, "opacity_grad gfopBuf.download");
  if (num_lines > 0)
    opac_->glopBuf.download(device, grad_line_opacity, VkDeviceSize(num_lines) * sizeof(float),
                            is_cuda, "opacity_grad glopBuf.download");
  if (num_points > 0)
    opac_->gpopBuf.download(device, grad_point_opacity, VkDeviceSize(num_points) * sizeof(float),
                            is_cuda, "opacity_grad gpopBuf.download");

  cuda.syncStream();

  return 0;
}

} // namespace fuzzydr

// src/filter.cpp
#include "filter.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>

#if defined(FUZZYDR_WITH_CUDA) && FUZZYDR_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include "vk_common.h"

namespace fuzzydr {

struct MsaaResolvePushConstants {
  uint32_t out_width;
  uint32_t out_height;
  float    sigma;         // in output pixels
};
static_assert(sizeof(MsaaResolvePushConstants) == 12, "MsaaResolvePushConstants must be 12 bytes");

// Internal cache struct (defined here, forward-declared in filter.h).
//
// `inBuf` holds the full-res RGBA (forward input) or the full-res RGBA
// gradient (backward output).  `poolBuf` holds the half-res RGB (forward
// output) or the half-res RGB gradient (backward input).
struct FilterCache {
  VulkanBuffer inBuf;     // [H*W*4]       float
  VulkanBuffer poolBuf;   // [H/2*W/2*3]   float

  void resetAll(VkDevice device) {
    inBuf.reset(device);
    poolBuf.reset(device);
  }
};

Filter::Filter() = default;

Filter::~Filter() {
  VkContext& ctx = requireCtx();
  VkDevice device = ctx.device;
  if (cache_) {
    cache_->resetAll(device);
    cache_.reset();
  }
}

// =============================================================================
// Helper: set up common Vulkan descriptor set layout, pipeline layout,
// descriptor pool, and descriptor sets for 2-buffer (input/output) compute
// dispatches.  Caller still creates pipelines from shader modules.
// =============================================================================

struct DispatchResources {
  VkDescriptorSetLayout setLayout  = VK_NULL_HANDLE;
  VkPipelineLayout      pipLayout  = VK_NULL_HANDLE;
  VkDescriptorPool      descPool   = VK_NULL_HANDLE;

  void destroy(VkDevice device) {
    if (descPool)  vkDestroyDescriptorPool(device, descPool, nullptr);
    if (pipLayout) vkDestroyPipelineLayout(device, pipLayout, nullptr);
    if (setLayout) vkDestroyDescriptorSetLayout(device, setLayout, nullptr);
    descPool  = VK_NULL_HANDLE;
    pipLayout = VK_NULL_HANDLE;
    setLayout = VK_NULL_HANDLE;
  }
};

// Create layout + pool that can hold `numSets` descriptor sets, each with two
// storage-buffer bindings (input + output).
static DispatchResources createDispatchResources(VkDevice device, uint32_t numSets,
                                                  uint32_t pcSize) {
  DispatchResources res{};

  // Descriptor set layout: binding0 = input, binding1 = output
  std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
  for (uint32_t i = 0; i < 2; ++i) {
    bindings[i].binding            = i;
    bindings[i].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount    = 1;
    bindings[i].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
  }

  VkDescriptorSetLayoutCreateInfo dsli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dsli.bindingCount = (uint32_t)bindings.size();
  dsli.pBindings    = bindings.data();
  vkCheck(vkCreateDescriptorSetLayout(device, &dsli, nullptr, &res.setLayout),
          "vkCreateDescriptorSetLayout(filter)");

  VkPushConstantRange pcr{};
  pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcr.offset     = 0;
  pcr.size       = pcSize;

  VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  plci.setLayoutCount         = 1;
  plci.pSetLayouts            = &res.setLayout;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges    = &pcr;
  vkCheck(vkCreatePipelineLayout(device, &plci, nullptr, &res.pipLayout),
          "vkCreatePipelineLayout(filter)");

  VkDescriptorPoolSize poolSize{};
  poolSize.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  poolSize.descriptorCount = 2 * numSets;

  VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpci.maxSets       = numSets;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes    = &poolSize;
  vkCheck(vkCreateDescriptorPool(device, &dpci, nullptr, &res.descPool),
          "vkCreateDescriptorPool(filter)");

  return res;
}

// Allocate a descriptor set from `res.descPool` and write two buffer bindings.
static VkDescriptorSet allocAndWriteDS(
    VkDevice device,
    const DispatchResources& res,
    VkBuffer inBuf,  VkDeviceSize inSize,
    VkBuffer outBuf, VkDeviceSize outSize)
{
  VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dsai.descriptorPool     = res.descPool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts        = &res.setLayout;

  VkDescriptorSet ds = VK_NULL_HANDLE;
  vkCheck(vkAllocateDescriptorSets(device, &dsai, &ds),
          "vkAllocateDescriptorSets(filter)");

  VkDescriptorBufferInfo inInfo{};
  inInfo.buffer = inBuf;
  inInfo.offset = 0;
  inInfo.range  = inSize;

  VkDescriptorBufferInfo outInfo{};
  outInfo.buffer = outBuf;
  outInfo.offset = 0;
  outInfo.range  = outSize;

  std::array<VkWriteDescriptorSet, 2> writes{};
  writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[0].dstSet          = ds;
  writes[0].dstBinding      = 0;
  writes[0].descriptorCount = 1;
  writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[0].pBufferInfo     = &inInfo;

  writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[1].dstSet          = ds;
  writes[1].dstBinding      = 1;
  writes[1].descriptorCount = 1;
  writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[1].pBufferInfo     = &outInfo;

  vkUpdateDescriptorSets(device, (uint32_t)writes.size(), writes.data(), 0, nullptr);
  return ds;
}

// Create a compute pipeline from a SPIR-V file.
static VkPipeline createComputePipeline(
    VkDevice device, VkPipelineLayout pipLayout, const char* spvName)
{
  VkShaderModule sm = createShaderModule(device, shaderPath(spvName));
  auto destroyShader = scopeGuard([&] { vkDestroyShaderModule(device, sm, nullptr); });

  VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
  stage.module = sm;
  stage.pName  = "main";

  VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  ci.stage  = stage;
  ci.layout = pipLayout;

  VkPipeline pipe = VK_NULL_HANDLE;
  vkCheck(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &ci, nullptr, &pipe),
          "vkCreateComputePipelines(filter)");
  return pipe;
}

// Record a buffer memory barrier between two compute dispatches.
static void recordBufferBarrier(
    VkCommandBuffer cmd,
    VkBuffer buf,
    VkDeviceSize size,
    VkAccessFlags srcAccess,
    VkAccessFlags dstAccess,
    VkPipelineStageFlags srcStage,
    VkPipelineStageFlags dstStage)
{
  VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.buffer        = buf;
  barrier.offset        = 0;
  barrier.size          = size;
  barrier.srcAccessMask = srcAccess;
  barrier.dstAccessMask = dstAccess;
  vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 1, &barrier, 0, nullptr);
}

// =============================================================================
// MSAA resolve  (forward)
//   Isotropic Gaussian weighted downsample: Input [H, W, 4] -> Output [H/2, W/2, 3]
//   (alpha channel of the input is ignored)
// =============================================================================

int Filter::msaa_downsample_rgba(
    uint32_t width,
    uint32_t height,
    const float* in_rgba,
    float* out_rgb,
    float sigma,
    bool is_cuda)
{
  if (!in_rgba || !out_rgb)
    throw std::runtime_error("msaa_downsample_rgba: null pointer");
  if (width == 0 || height == 0)
    throw std::runtime_error("msaa_downsample_rgba: width/height must be > 0");
  if (width % 2 != 0 || height % 2 != 0)
    throw std::runtime_error("msaa_downsample_rgba: width and height must be even");

  VkContext& ctx = requireCtx();
  validateCudaSupport(is_cuda, ctx, "msaa_downsample_rgba");

  VkDevice         device         = ctx.device;
  VkPhysicalDevice physicalDevice = ctx.physicalDevice;
  VkQueue          queue          = ctx.queue;
  VkCommandPool    cmdPool        = ctx.cmdPool;

  CudaVkSyncGuard cuda(device, is_cuda);

  const uint32_t outW = width / 2;
  const uint32_t outH = height / 2;

  const VkDeviceSize fullBytes = VkDeviceSize(width) * VkDeviceSize(height) * 4 * sizeof(float);
  const VkDeviceSize outBytes  = VkDeviceSize(outW)  * VkDeviceSize(outH)   * 3 * sizeof(float);

  {
    if (!cache_) cache_ = std::make_unique<FilterCache>();

    const VkMemoryPropertyFlags ioProps =
        is_cuda ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                : (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    cache_->inBuf.ensure(  device, physicalDevice, fullBytes,
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, ioProps, is_cuda);
    cache_->poolBuf.ensure(device, physicalDevice, outBytes,
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, ioProps, is_cuda);

    cache_->inBuf.upload(device, in_rgba, fullBytes, is_cuda,
                         "msaa_resolve fwd inBuf.upload");
  }

  cuda.signalCudaToVk();

  DispatchResources res{};
  VkPipeline pipe = VK_NULL_HANDLE;

  auto cleanup = scopeGuard([&] {
    if (pipe) vkDestroyPipeline(device, pipe, nullptr);
    res.destroy(device);
  });

  res  = createDispatchResources(device, 1, sizeof(MsaaResolvePushConstants));
  VkDescriptorSet ds = allocAndWriteDS(device, res,
      cache_->inBuf.buf.buf,  fullBytes,
      cache_->poolBuf.buf.buf, outBytes);

  pipe = createComputePipeline(device, res.pipLayout, "msaa_resolve_fwd.comp.spv");

  VkCommandBuffer cmd = cmdBegin(device, cmdPool);

  {
    MsaaResolvePushConstants pc{outW, outH, sigma};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            res.pipLayout, 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, res.pipLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, ceilDiv(outW, 16), ceilDiv(outH, 16), 1);
  }

  recordBufferBarrier(cmd, cache_->poolBuf.buf.buf, outBytes,
                      VK_ACCESS_SHADER_WRITE_BIT,
                      is_cuda ? (VkAccessFlags)0 : VK_ACCESS_HOST_READ_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      is_cuda ? VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT
                              : VK_PIPELINE_STAGE_HOST_BIT);

  {
    SubmitParams sp{};
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    cuda.fillSubmit(sp, waitStage);
    submitAndWait(device, queue, cmdPool, cmd, sp, "vkCreateFence(msaa_resolve_fwd)");
  }

  cuda.waitVkToCuda();
  cache_->poolBuf.download(device, out_rgb, outBytes, is_cuda,
                           "msaa_resolve fwd poolBuf.download");
  cuda.syncStream();

  return 0;
}

// =============================================================================
// MSAA resolve  (backward / adjoint)
//   grad_out [H/2, W/2, 3] -> grad_in [H, W, 4]  (grad_in channel 3 = 0)
// =============================================================================

int Filter::msaa_downsample_rgba_backward(
    uint32_t width,
    uint32_t height,
    const float* grad_out_rgb,
    float* grad_in_rgba,
    float sigma,
    bool is_cuda)
{
  if (!grad_out_rgb || !grad_in_rgba)
    throw std::runtime_error("msaa_downsample_rgba_backward: null pointer");
  if (width == 0 || height == 0)
    throw std::runtime_error("msaa_downsample_rgba_backward: width/height must be > 0");
  if (width % 2 != 0 || height % 2 != 0)
    throw std::runtime_error("msaa_downsample_rgba_backward: width and height must be even");

  VkContext& ctx = requireCtx();
  validateCudaSupport(is_cuda, ctx, "msaa_downsample_rgba_backward");

  VkDevice         device         = ctx.device;
  VkPhysicalDevice physicalDevice = ctx.physicalDevice;
  VkQueue          queue          = ctx.queue;
  VkCommandPool    cmdPool        = ctx.cmdPool;

  CudaVkSyncGuard cuda(device, is_cuda);

  const uint32_t halfW = width / 2;
  const uint32_t halfH = height / 2;

  const VkDeviceSize fullBytes = VkDeviceSize(width) * VkDeviceSize(height) * 4 * sizeof(float);
  const VkDeviceSize halfBytes = VkDeviceSize(halfW) * VkDeviceSize(halfH)  * 3 * sizeof(float);

  {
    if (!cache_) cache_ = std::make_unique<FilterCache>();

    const VkMemoryPropertyFlags ioProps =
        is_cuda ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                : (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    cache_->poolBuf.ensure(device, physicalDevice, halfBytes,
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, ioProps, is_cuda);
    cache_->inBuf.ensure(  device, physicalDevice, fullBytes,
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, ioProps, is_cuda);

    cache_->poolBuf.upload(device, grad_out_rgb, halfBytes, is_cuda,
                           "msaa_resolve bwd poolBuf.upload");
  }

  cuda.signalCudaToVk();

  DispatchResources res{};
  VkPipeline pipe = VK_NULL_HANDLE;

  auto cleanup = scopeGuard([&] {
    if (pipe) vkDestroyPipeline(device, pipe, nullptr);
    res.destroy(device);
  });

  res  = createDispatchResources(device, 1, sizeof(MsaaResolvePushConstants));
  VkDescriptorSet ds = allocAndWriteDS(device, res,
      cache_->poolBuf.buf.buf, halfBytes,
      cache_->inBuf.buf.buf,   fullBytes);

  pipe = createComputePipeline(device, res.pipLayout, "msaa_resolve_bwd.comp.spv");

  VkCommandBuffer cmd = cmdBegin(device, cmdPool);

  {
    // Push constants carry the *output* (half-res) dimensions + sigma,
    // matching the forward shader struct.
    MsaaResolvePushConstants pc{halfW, halfH, sigma};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            res.pipLayout, 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cmd, res.pipLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    // Dispatch over the full-res grid (each thread = one high-res pixel).
    vkCmdDispatch(cmd, ceilDiv(width, 16), ceilDiv(height, 16), 1);
  }

  recordBufferBarrier(cmd, cache_->inBuf.buf.buf, fullBytes,
                      VK_ACCESS_SHADER_WRITE_BIT,
                      is_cuda ? (VkAccessFlags)0 : VK_ACCESS_HOST_READ_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      is_cuda ? VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT
                              : VK_PIPELINE_STAGE_HOST_BIT);

  {
    SubmitParams sp{};
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    cuda.fillSubmit(sp, waitStage);
    submitAndWait(device, queue, cmdPool, cmd, sp, "vkCreateFence(msaa_resolve_bwd)");
  }

  cuda.waitVkToCuda();
  cache_->inBuf.download(device, grad_in_rgba, fullBytes, is_cuda,
                         "msaa_resolve bwd inBuf.download");
  cuda.syncStream();

  return 0;
}

} // namespace fuzzydr

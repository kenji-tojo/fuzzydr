// src/vk_common.h
#pragma once

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
  #ifndef NOMINMAX
  #define NOMINMAX
  #endif
  #include <windows.h>
  #include <vulkan/vulkan_win32.h>
  #include <string>
#else
  #include <dlfcn.h>
  #include <libgen.h>
  #include <limits.h>
  #include <unistd.h> // close()
#endif

#if defined(FUZZYDR_WITH_CUDA) && FUZZYDR_WITH_CUDA
  #include <cuda.h>
  #include <cuda_runtime.h>
#endif

// ---- RAII scope guard ----
// Runs a cleanup lambda on scope exit.

template<typename F>
class ScopeGuard {
  F fn_;
  bool active_ = true;
public:
  explicit ScopeGuard(F&& f) : fn_(std::move(f)) {}
  ~ScopeGuard() { if (active_) fn_(); }
  void dismiss() { active_ = false; }
  ScopeGuard(const ScopeGuard&) = delete;
  ScopeGuard& operator=(const ScopeGuard&) = delete;
};

template<typename F>
[[nodiscard]] ScopeGuard<F> scopeGuard(F&& f) { return ScopeGuard<F>(std::forward<F>(f)); }

// ---- Common Vulkan helpers ----

static inline void vkCheck(VkResult r, const char* what) {
  if (r != VK_SUCCESS) {
    throw std::runtime_error(std::string("Vulkan error: ") + what + " (code " + std::to_string((int)r) + ")");
  }
}

static inline std::vector<uint8_t> readFileBytes(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Failed to open file: " + path);
  f.seekg(0, std::ios::end);
  auto pos = f.tellg();
  if (pos == std::ifstream::pos_type(-1))
    throw std::runtime_error("Failed to determine file size (non-seekable?): " + path);
  size_t size = (size_t)pos;
  f.seekg(0, std::ios::beg);
  std::vector<uint8_t> data(size);
  f.read((char*)data.data(), (std::streamsize)size);
  return data;
}

static inline uint32_t findMemoryTypeIndex(VkPhysicalDevice phys, uint32_t typeBits, VkMemoryPropertyFlags props) {
  VkPhysicalDeviceMemoryProperties memProps{};
  vkGetPhysicalDeviceMemoryProperties(phys, &memProps);
  for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
    if ((typeBits & (1u << i)) && ((memProps.memoryTypes[i].propertyFlags & props) == props)) {
      return i;
    }
  }
  throw std::runtime_error("No suitable memory type found.");
}

struct Buffer {
  VkBuffer buf = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  VkDeviceSize size = 0;
};

struct Image {
  VkImage img = VK_NULL_HANDLE;
  VkDeviceMemory mem = VK_NULL_HANDLE;
  VkImageView view = VK_NULL_HANDLE;
  VkFormat format{};
  uint32_t width = 0, height = 0;
};

static inline void createBuffer(VkDevice dev, VkPhysicalDevice phys, VkDeviceSize size,
                                VkBufferUsageFlags usage, VkMemoryPropertyFlags memProps,
                                Buffer& out) {
  out.size = size;

  VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  ci.size = size;
  ci.usage = usage;
  ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  vkCheck(vkCreateBuffer(dev, &ci, nullptr, &out.buf), "vkCreateBuffer");

  VkMemoryRequirements req{};
  vkGetBufferMemoryRequirements(dev, out.buf, &req);

  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = findMemoryTypeIndex(phys, req.memoryTypeBits, memProps);
  vkCheck(vkAllocateMemory(dev, &ai, nullptr, &out.mem), "vkAllocateMemory(buffer)");
  vkCheck(vkBindBufferMemory(dev, out.buf, out.mem, 0), "vkBindBufferMemory");
}

static inline void destroyBuffer(VkDevice dev, Buffer& b) {
  if (b.buf) vkDestroyBuffer(dev, b.buf, nullptr);
  if (b.mem) vkFreeMemory(dev, b.mem, nullptr);
  b = {};
}

static inline void createImage2D(VkDevice dev, VkPhysicalDevice phys, uint32_t w, uint32_t h,
                                 VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect,
                                 Image& out) {
  out.width = w;
  out.height = h;
  out.format = format;

  VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ci.imageType = VK_IMAGE_TYPE_2D;
  ci.format = format;
  ci.extent = {w, h, 1};
  ci.mipLevels = 1;
  ci.arrayLayers = 1;
  ci.samples = VK_SAMPLE_COUNT_1_BIT;
  ci.tiling = VK_IMAGE_TILING_OPTIMAL;
  ci.usage = usage;
  ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  vkCheck(vkCreateImage(dev, &ci, nullptr, &out.img), "vkCreateImage");

  VkMemoryRequirements req{};
  vkGetImageMemoryRequirements(dev, out.img, &req);

  VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  ai.allocationSize = req.size;
  ai.memoryTypeIndex = findMemoryTypeIndex(phys, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  vkCheck(vkAllocateMemory(dev, &ai, nullptr, &out.mem), "vkAllocateMemory(image)");
  vkCheck(vkBindImageMemory(dev, out.img, out.mem, 0), "vkBindImageMemory");

  VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  vi.image = out.img;
  vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
  vi.format = format;
  vi.subresourceRange.aspectMask = aspect;
  vi.subresourceRange.baseMipLevel = 0;
  vi.subresourceRange.levelCount = 1;
  vi.subresourceRange.baseArrayLayer = 0;
  vi.subresourceRange.layerCount = 1;

  vkCheck(vkCreateImageView(dev, &vi, nullptr, &out.view), "vkCreateImageView");
}

static inline void destroyImage(VkDevice dev, Image& im) {
  if (im.view) vkDestroyImageView(dev, im.view, nullptr);
  if (im.img) vkDestroyImage(dev, im.img, nullptr);
  if (im.mem) vkFreeMemory(dev, im.mem, nullptr);
  im = {};
}

// ---- Command buffer helpers ----

struct SubmitParams {
  // Optional GPU->GPU sync
  const VkSemaphore* waitSems = nullptr;
  const VkPipelineStageFlags* waitStages = nullptr;
  uint32_t waitCount = 0;

  const VkSemaphore* signalSems = nullptr;
  uint32_t signalCount = 0;

  // Optional fence. If null, no fence is used.
  VkFence fence = VK_NULL_HANDLE;

  // If true and fence != null, the fence is waited on and destroyed after submit.
  // If fence == null, this flag is ignored.
  bool waitFence = false;

  // If true, always wait for queue idle after submit (debug only).
  bool waitQueueIdle = false;
};

// Allocate + begin a command buffer.
static inline VkCommandBuffer cmdBegin(VkDevice dev, VkCommandPool pool,
                                      VkCommandBufferUsageFlags flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT) {
  VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ai.commandPool = pool;
  ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = 1;

  VkCommandBuffer cmd = VK_NULL_HANDLE;
  vkCheck(vkAllocateCommandBuffers(dev, &ai, &cmd), "vkAllocateCommandBuffers");

  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = flags;
  vkCheck(vkBeginCommandBuffer(cmd, &bi), "vkBeginCommandBuffer");
  return cmd;
}

// End + submit a command buffer (does not free cmd unless freeCmd=true).
static inline void cmdEndSubmit(VkDevice dev, VkQueue q, VkCommandPool pool, VkCommandBuffer cmd,
                                const SubmitParams& sp,
                                bool freeCmd = true) {
  vkCheck(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");

  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.waitSemaphoreCount = sp.waitCount;
  si.pWaitSemaphores = sp.waitSems;
  si.pWaitDstStageMask = sp.waitStages;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  si.signalSemaphoreCount = sp.signalCount;
  si.pSignalSemaphores = sp.signalSems;

  vkCheck(vkQueueSubmit(q, 1, &si, sp.fence), "vkQueueSubmit");

  if (sp.waitFence && sp.fence) {
    vkCheck(vkWaitForFences(dev, 1, &sp.fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
  }
  if (sp.waitQueueIdle) {
    vkCheck(vkQueueWaitIdle(q), "vkQueueWaitIdle");
  }

  if (freeCmd) {
    vkFreeCommandBuffers(dev, pool, 1, &cmd);
  }
}

// End + submit + fence-wait in one call.  Creates and destroys a temporary fence.
static inline void submitAndWait(VkDevice dev, VkQueue q, VkCommandPool pool,
                                 VkCommandBuffer cmd, SubmitParams& sp, const char* tag) {
  VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  VkFence fence = VK_NULL_HANDLE;
  vkCheck(vkCreateFence(dev, &fci, nullptr, &fence), tag);
  sp.fence = fence;
  sp.waitFence = true;
  cmdEndSubmit(dev, q, pool, cmd, sp, /*freeCmd=*/true);
  vkDestroyFence(dev, fence, nullptr);
}

static inline void cmdImageBarrier(VkCommandBuffer cmd, VkImage img,
                                   VkImageLayout oldLayout, VkImageLayout newLayout,
                                   VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage,
                                   VkAccessFlags srcAccess, VkAccessFlags dstAccess,
                                   VkImageAspectFlags aspectMask) {
  VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  b.oldLayout = oldLayout;
  b.newLayout = newLayout;
  b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.image = img;
  b.subresourceRange.aspectMask = aspectMask;
  b.subresourceRange.baseMipLevel = 0;
  b.subresourceRange.levelCount = 1;
  b.subresourceRange.baseArrayLayer = 0;
  b.subresourceRange.layerCount = 1;
  b.srcAccessMask = srcAccess;
  b.dstAccessMask = dstAccess;

  vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

static inline VkShaderModule createShaderModule(VkDevice dev, const std::string& path) {
  auto bytes = readFileBytes(path);
  VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  ci.codeSize = bytes.size();
  ci.pCode = reinterpret_cast<const uint32_t*>(bytes.data());
  VkShaderModule mod = VK_NULL_HANDLE;
  vkCheck(vkCreateShaderModule(dev, &ci, nullptr, &mod), "vkCreateShaderModule");
  return mod;
}

static inline bool hasDeviceExtension(VkPhysicalDevice pd, const char* name) {
  uint32_t count = 0;
  vkEnumerateDeviceExtensionProperties(pd, nullptr, &count, nullptr);
  std::vector<VkExtensionProperties> exts(count);
  vkEnumerateDeviceExtensionProperties(pd, nullptr, &count, exts.data());
  for (auto& e : exts) {
    if (std::strcmp(e.extensionName, name) == 0) return true;
  }
  return false;
}

// ---- Module/shader path helpers ----
// NOTE: these are `inline` (not `static inline`) so that the linker
// deduplicates them across translation units.  On POSIX this ensures
// dladdr(&moduleDir, ...) always resolves to the same shared-library
// address regardless of which TU the caller is in.
inline std::string moduleDir() {
#ifdef _WIN32
  // Get the path to the module (DLL) that contains this function.
  HMODULE hm = nullptr;
  if (!GetModuleHandleExA(
          GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
          reinterpret_cast<LPCSTR>(&moduleDir), &hm)) {
    throw std::runtime_error("GetModuleHandleExA failed: cannot locate module path");
  }

  wchar_t wpath[MAX_PATH];
  DWORD len = GetModuleFileNameW(hm, wpath, MAX_PATH);
  if (len == 0 || len == MAX_PATH) {
    throw std::runtime_error("GetModuleFileNameW failed: cannot locate module path");
  }

  std::wstring ws(wpath, len);
  // Strip filename -> directory
  size_t pos = ws.find_last_of(L"\\/");
  std::wstring wdir = (pos == std::wstring::npos) ? L"." : ws.substr(0, pos);

  // Convert to UTF-8 std::string
  int needed = WideCharToMultiByte(CP_UTF8, 0, wdir.c_str(), (int)wdir.size(),
                                   nullptr, 0, nullptr, nullptr);
  if (needed <= 0) throw std::runtime_error("WideCharToMultiByte failed");
  std::string dir(needed, '\0');
  WideCharToMultiByte(CP_UTF8, 0, wdir.c_str(), (int)wdir.size(),
                      dir.data(), needed, nullptr, nullptr);
  return dir;
#else
  Dl_info info{};
  if (dladdr((void*)&moduleDir, &info) == 0 || !info.dli_fname) {
    throw std::runtime_error("dladdr failed: cannot locate module path");
  }

  char path[PATH_MAX];
  std::strncpy(path, info.dli_fname, sizeof(path));
  path[sizeof(path) - 1] = '\0';

  char* dir = ::dirname(path);
  if (!dir) throw std::runtime_error("dirname failed");
  return std::string(dir);
#endif
}

inline std::string shaderPath(const char* filename) {
#ifdef _WIN32
  return moduleDir() + "\\" + filename;
#else
  return moduleDir() + "/" + filename;
#endif
}

namespace fuzzydr {

// ---- Shared arithmetic helpers ----

/// Integer ceiling division: ceilDiv(a, b) = ceil(a / b).
inline uint32_t ceilDiv(uint32_t a, uint32_t b) { return (a + b - 1u) / b; }

/// Minimum buffer size for dummy descriptor bindings when a primitive class is empty.
inline constexpr VkDeviceSize kMinBufSize = 4;

/// Returns at least kMinBufSize so that zero-element SSBO bindings remain valid.
inline VkDeviceSize safeSize(VkDeviceSize s) { return (s > kMinBufSize) ? s : kMinBufSize; }

// ---- Vulkan context ----

class VkContext {
public:
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;

  // Fields used by the rasterize path:
  VkQueue queue = VK_NULL_HANDLE;
  uint32_t graphicsQF = UINT32_MAX;
  VkCommandPool cmdPool = VK_NULL_HANDLE;

  bool extMemEnabled = false;
  bool extSemEnabled = false;

  // gpuIndex selects the Vulkan physical device by enumeration index.
  // <0 (default) picks the first suitable device.
  explicit VkContext(int gpuIndex = -1);
  ~VkContext();

  VkContext(const VkContext&) = delete;
  VkContext& operator=(const VkContext&) = delete;

  void requireExternalMemory(const char* what) const {
    if (!extMemEnabled) {
      throw std::runtime_error(std::string(what) +
        ": VkContext has no external-memory support enabled (cannot do Vulkan<->CUDA interop)");
    }
  }

  void requireExternalSemaphore(const char* what) const {
    if (!extSemEnabled) {
      throw std::runtime_error(std::string(what) +
        ": VkContext has no external-semaphore support enabled (cannot do Vulkan<->CUDA sync interop)");
    }
  }
};

// Helper: common CUDA requirement checks used by all entry points.
static inline void validateCudaSupport(
    [[maybe_unused]] bool is_cuda,
    [[maybe_unused]] VkContext& ctx,
    [[maybe_unused]] const char* who)
{
#if !(defined(FUZZYDR_WITH_CUDA) && FUZZYDR_WITH_CUDA)
  if (is_cuda)
    throw std::runtime_error(std::string(who) + ": is_cuda=true but FUZZYDR_WITH_CUDA=0");
#else
  if (is_cuda) {
    ctx.requireExternalMemory(who);
    ctx.requireExternalSemaphore(who);
  }
#endif
}

void initCtx(int gpuIndex = -1);
void shutdownCtx();
VkContext& requireCtx();

struct VulkanBuffer {
  Buffer buf{};
  VkDeviceSize bytes = 0;       // logical size last requested via ensure()
  VkDeviceSize capacity = 0;    // actual allocated VkBuffer size (>= bytes); see ensure()
  VkBufferUsageFlags usage = 0;
  VkMemoryPropertyFlags props = 0;

  // True if this buffer was created as exportable external memory and imported into CUDA.
  bool exportToCuda = false;

#if defined(FUZZYDR_WITH_CUDA) && FUZZYDR_WITH_CUDA
  // CUDA interop state (valid only if exportToCuda==true)
  VkDeviceSize allocSize = 0;        // Vk allocation size (may be >= bytes)
  CUexternalMemory extMem = nullptr;
  CUdeviceptr cudaPtr = 0;

  #if defined(_WIN32)
    HANDLE handle = nullptr;
  #else
    int fd = -1;
  #endif
#endif

  bool initialized() const {
    return buf.buf != VK_NULL_HANDLE && buf.mem != VK_NULL_HANDLE && bytes > 0;
  }

  // Destroys Vulkan + CUDA interop resources (safe to call repeatedly).
  void reset(VkDevice device);

  // export_to_cuda defaults false.
  void ensure(VkDevice device,
              VkPhysicalDevice physicalDevice,
              VkDeviceSize requiredBytes,
              VkBufferUsageFlags requiredUsage,
              VkMemoryPropertyFlags requiredProps,
              bool export_to_cuda = false);

  void requireExact(VkDeviceSize requiredBytes, const char* what) const;

  // If is_cuda==false: src/dst are HOST pointers, copied by mapping VkMemory.
  // If is_cuda==true:  src/dst are DEVICE pointers, copied with cudaMemcpy on cudaPtr.
  void upload(VkDevice device, const void* src, VkDeviceSize nbytes, bool is_cuda, const char* what) const;
  void download(VkDevice device, void* dst, VkDeviceSize nbytes, bool is_cuda, const char* what) const;
  void memset(VkDevice device, int value, VkDeviceSize nbytes, bool is_cuda, const char* what) const;
};

#if defined(FUZZYDR_WITH_CUDA) && FUZZYDR_WITH_CUDA

// A single pair of semaphores for CUDA<->Vulkan binary sync.
struct CudaVkSemaphore {
  VkSemaphore vkSem = VK_NULL_HANDLE;

#ifdef _WIN32
  HANDLE handle = nullptr; // CloseHandle (NOT owned by CUDA)
#else
  int fd = -1; // owned by CUDA after successful import
#endif

  cudaExternalSemaphore_t cudaSem = nullptr;

  void reset(VkDevice dev) {
    if (cudaSem) { cudaDestroyExternalSemaphore(cudaSem); cudaSem = nullptr; }

#ifdef _WIN32
    if (handle) { CloseHandle(handle); handle = nullptr; }
#else
    // For OPAQUE_FD, ownership is transferred to CUDA on successful import.
    // So fd is set to -1 after import and must NOT be closed here.
    fd = -1;
#endif

    if (vkSem) { vkDestroySemaphore(dev, vkSem, nullptr); vkSem = VK_NULL_HANDLE; }
  }
};

static inline void createCudaVkSemaphore(VkDevice dev, CudaVkSemaphore& out) {
#ifdef _WIN32
  auto pfnGetSemaphoreWin32HandleKHR =
      (PFN_vkGetSemaphoreWin32HandleKHR)vkGetDeviceProcAddr(dev, "vkGetSemaphoreWin32HandleKHR");
  if (!pfnGetSemaphoreWin32HandleKHR) {
    throw std::runtime_error("vkGetSemaphoreWin32HandleKHR not found (external semaphore not enabled?)");
  }

  VkExportSemaphoreCreateInfo ex{VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
  ex.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

  VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  sci.pNext = &ex;
  vkCheck(vkCreateSemaphore(dev, &sci, nullptr, &out.vkSem), "vkCreateSemaphore(external)");

  VkSemaphoreGetWin32HandleInfoKHR hi{VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR};
  hi.semaphore = out.vkSem;
  hi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

  HANDLE h = nullptr;
  vkCheck(pfnGetSemaphoreWin32HandleKHR(dev, &hi, &h), "vkGetSemaphoreWin32HandleKHR");
  if (!h) throw std::runtime_error("vkGetSemaphoreWin32HandleKHR returned null handle");
  out.handle = h;

  cudaExternalSemaphoreHandleDesc sh{};
  sh.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
  sh.handle.win32.handle = out.handle;

  auto e = cudaImportExternalSemaphore(&out.cudaSem, &sh);
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("cudaImportExternalSemaphore failed: ") + cudaGetErrorString(e));
  }

#else
  auto pfnGetSemaphoreFdKHR =
      (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(dev, "vkGetSemaphoreFdKHR");
  if (!pfnGetSemaphoreFdKHR) {
    throw std::runtime_error("vkGetSemaphoreFdKHR not found (external semaphore not enabled?)");
  }

  VkExportSemaphoreCreateInfo ex{VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
  ex.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

  VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  sci.pNext = &ex;
  vkCheck(vkCreateSemaphore(dev, &sci, nullptr, &out.vkSem), "vkCreateSemaphore(external)");

  VkSemaphoreGetFdInfoKHR hi{VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
  hi.semaphore = out.vkSem;
  hi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

  int fd = -1;
  vkCheck(pfnGetSemaphoreFdKHR(dev, &hi, &fd), "vkGetSemaphoreFdKHR");
  if (fd < 0) throw std::runtime_error("vkGetSemaphoreFdKHR returned invalid fd");
  out.fd = fd;

  cudaExternalSemaphoreHandleDesc sh{};
  sh.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
  sh.handle.fd = out.fd;

  // For OPAQUE_FD, ownership of the fd transfers to CUDA on successful import.
  auto e = cudaImportExternalSemaphore(&out.cudaSem, &sh);
  if (e != cudaSuccess) {
    // import failed, so the fd is still owned here
    ::close(out.fd);
    out.fd = -1;
    throw std::runtime_error(std::string("cudaImportExternalSemaphore failed: ") + cudaGetErrorString(e));
  }

  // success: CUDA owns the fd now
  out.fd = -1;
#endif
}

// Helpers: no-op when is_cuda=false.
static inline void cudaSignalIf(cudaStream_t stream, bool is_cuda, const CudaVkSemaphore& s) {
  if (!is_cuda) return;
  cudaExternalSemaphoreSignalParams p{};
  auto e = cudaSignalExternalSemaphoresAsync(&s.cudaSem, &p, 1, stream);
  if (e != cudaSuccess) throw std::runtime_error(std::string("cudaSignalExternalSemaphoresAsync: ") + cudaGetErrorString(e));
}

static inline void cudaWaitIf(cudaStream_t stream, bool is_cuda, const CudaVkSemaphore& s) {
  if (!is_cuda) return;
  cudaExternalSemaphoreWaitParams p{};
  auto e = cudaWaitExternalSemaphoresAsync(&s.cudaSem, &p, 1, stream);
  if (e != cudaSuccess) throw std::runtime_error(std::string("cudaWaitExternalSemaphoresAsync: ") + cudaGetErrorString(e));
}

#endif // FUZZYDR_WITH_CUDA

// =============================================================================
// CudaVkSyncGuard - RAII wrapper for per-call CUDA<->Vulkan semaphore pair
// =============================================================================
// Usage:
//   CudaVkSyncGuard cuda(device, is_cuda);
//   cuda.signalCudaToVk();
//   ... record Vulkan commands ...
//   cuda.fillSubmit(sp, waitStage);
//   submitAndWait(...);
//   cuda.waitVkToCuda();
//   ... download ...
//   cuda.syncStream();
// Semaphores are destroyed automatically when the scope exits.

struct CudaVkSyncGuard {
#if defined(FUZZYDR_WITH_CUDA) && FUZZYDR_WITH_CUDA
  cudaStream_t stream = nullptr;
  CudaVkSemaphore semCudaToVk{};
  CudaVkSemaphore semVkToCuda{};
#endif
  VkDevice device_ = VK_NULL_HANDLE;
  bool active_ = false;

  CudaVkSyncGuard() = default;

  CudaVkSyncGuard(VkDevice dev, bool is_cuda) : device_(dev), active_(is_cuda) {
#if defined(FUZZYDR_WITH_CUDA) && FUZZYDR_WITH_CUDA
    if (active_) {
      cudaFree(nullptr); // ensure CUDA runtime initialized
      createCudaVkSemaphore(device_, semCudaToVk);
      createCudaVkSemaphore(device_, semVkToCuda);
    }
#endif
  }

  ~CudaVkSyncGuard() {
#if defined(FUZZYDR_WITH_CUDA) && FUZZYDR_WITH_CUDA
    if (active_) {
      semCudaToVk.reset(device_);
      semVkToCuda.reset(device_);
    }
#endif
  }

  CudaVkSyncGuard(const CudaVkSyncGuard&) = delete;
  CudaVkSyncGuard& operator=(const CudaVkSyncGuard&) = delete;

  /// CUDA signals that its writes are done; Vulkan can now read external memory.
  void signalCudaToVk() {
#if defined(FUZZYDR_WITH_CUDA) && FUZZYDR_WITH_CUDA
    if (active_) cudaSignalIf(stream, true, semCudaToVk);
#endif
  }

  /// After Vulkan submit, make CUDA wait until Vulkan work is done.
  void waitVkToCuda() {
#if defined(FUZZYDR_WITH_CUDA) && FUZZYDR_WITH_CUDA
    if (active_) cudaWaitIf(stream, true, semVkToCuda);
#endif
  }

  /// Synchronize the CUDA stream after downloads.
  void syncStream() {
#if defined(FUZZYDR_WITH_CUDA) && FUZZYDR_WITH_CUDA
    if (active_) cudaStreamSynchronize(stream);
#endif
  }

  /// Fill SubmitParams with wait/signal semaphores for Vulkan<->CUDA sync.
  /// `waitStage` must outlive the submit call.
  void fillSubmit(SubmitParams& sp, VkPipelineStageFlags& waitStage) {
#if defined(FUZZYDR_WITH_CUDA) && FUZZYDR_WITH_CUDA
    if (active_) {
      sp.waitSems    = &semCudaToVk.vkSem;
      sp.waitStages  = &waitStage;
      sp.waitCount   = 1;
      sp.signalSems  = &semVkToCuda.vkSem;
      sp.signalCount = 1;
    }
#endif
  }
};

} // namespace fuzzydr

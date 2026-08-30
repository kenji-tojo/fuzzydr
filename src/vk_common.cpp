// src/vk_common.cpp
#include "vk_common.h"

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

#if defined(FUZZYDR_WITH_CUDA) && FUZZYDR_WITH_CUDA
  #include <iostream>
  #if defined(_WIN32)
    #include <vulkan/vulkan_win32.h>
  #else
    #include <unistd.h>
  #endif
#endif

namespace fuzzydr {

static std::unique_ptr<VkContext> g_ctx;

static inline void pushUnique(std::vector<const char*>& v, const char* s) {
  for (auto* x : v) if (std::strcmp(x, s) == 0) return;
  v.push_back(s);
}

static inline uint32_t findQueueFamilyWithFlags(VkPhysicalDevice pd, VkQueueFlags wantFlags) {
  uint32_t qfCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfCount, nullptr);
  std::vector<VkQueueFamilyProperties> qfs(qfCount);
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfCount, qfs.data());

  for (uint32_t i = 0; i < qfCount; ++i) {
    if ((qfs[i].queueFlags & wantFlags) == wantFlags) return i;
  }
  return UINT32_MAX;
}

void initCtx(int gpuIndex) {
  if (!g_ctx) {
    // Explicit arg wins; otherwise fall back to the FUZZYDR_DEVICE_INDEX env
    // var. <0 = auto.
    if (gpuIndex < 0) {
      if (const char* e = std::getenv("FUZZYDR_DEVICE_INDEX")) {
        if (e[0] != '\0') gpuIndex = std::atoi(e);
      }
    }
    g_ctx = std::make_unique<VkContext>(gpuIndex);
  }
}

void shutdownCtx() {
  g_ctx.reset();
}

VkContext& requireCtx() {
  if (!g_ctx) {
    throw std::runtime_error("Vulkan context is not initialized. Call _core.init() first.");
  }
  return *g_ctx;
}

VkContext::VkContext(int gpuIndex) {
  // ---------------- Instance ----------------
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "FuzzyDR";
  app.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  app.pEngineName = "none";
  app.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  app.apiVersion = VK_API_VERSION_1_1;

  std::vector<const char*> instanceExts;
  instanceExts.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
#ifdef __APPLE__
  instanceExts.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#endif

  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  ici.enabledExtensionCount = (uint32_t)instanceExts.size();
  ici.ppEnabledExtensionNames = instanceExts.data();
#ifdef __APPLE__
  ici.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

  vkCheck(vkCreateInstance(&ici, nullptr, &instance), "vkCreateInstance");

  // ---------------- Pick physical device ----------------
  uint32_t physCount = 0;
  vkCheck(vkEnumeratePhysicalDevices(instance, &physCount, nullptr), "vkEnumeratePhysicalDevices(count)");
  if (physCount == 0) throw std::runtime_error("No Vulkan physical devices found.");

  std::vector<VkPhysicalDevice> phys(physCount);
  vkCheck(vkEnumeratePhysicalDevices(instance, &physCount, phys.data()), "vkEnumeratePhysicalDevices(list)");

  // Requirements for the rasterize path:
  const char* baryExt = VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME;
  const char* atomicFloatExt = VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME;

  // Note: CUDA interop external-memory/semaphore extensions are optional and are
  // enabled (if present) during device creation below; they are not required for
  // physical device selection.

  // A device is suitable if it has a graphics queue and the rasterize-required
  // extensions. Returns the graphics queue family index, or UINT32_MAX if not.
  auto suitableQueue = [&](VkPhysicalDevice pd) -> uint32_t {
    uint32_t gq = findQueueFamilyWithFlags(pd, VK_QUEUE_GRAPHICS_BIT);
    if (gq == UINT32_MAX) return UINT32_MAX;
    if (!hasDeviceExtension(pd, baryExt)) return UINT32_MAX;
    if (!hasDeviceExtension(pd, atomicFloatExt)) return UINT32_MAX;
    return gq;
  };

  if (gpuIndex >= 0) {
    // Explicit selection: pick the physical device at the requested Vulkan
    // enumeration index (multi-GPU interop must match the CUDA device).
    if ((uint32_t)gpuIndex >= physCount)
      throw std::runtime_error("Requested Vulkan GPU index is out of range "
                               "(fewer physical devices than requested).");
    uint32_t gq = suitableQueue(phys[gpuIndex]);
    if (gq == UINT32_MAX)
      throw std::runtime_error("Requested Vulkan GPU index is not suitable "
                               "(missing required graphics queue/exts).");
    physicalDevice = phys[gpuIndex];
    graphicsQF = gq;
  } else {
    // Auto selection: first suitable device.
    for (auto pd : phys) {
      uint32_t gq = suitableQueue(pd);
      if (gq == UINT32_MAX) continue;
      physicalDevice = pd;
      graphicsQF = gq;
      break;
    }
  }

  if (!physicalDevice || graphicsQF == UINT32_MAX)
    throw std::runtime_error("No suitable Vulkan physical device found (missing required graphics queue/exts).");

  // ---------------- Feature checks ----------------
  VkPhysicalDeviceFeatures2 feats2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};

  VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR baryFeats{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};

  VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFeats{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};

  feats2.pNext = &baryFeats;
  baryFeats.pNext = &atomicFeats;

  vkGetPhysicalDeviceFeatures2(physicalDevice, &feats2);

  if (!baryFeats.fragmentShaderBarycentric) {
    throw std::runtime_error("fragmentShaderBarycentric feature is not available.");
  }
  if (!atomicFeats.shaderBufferFloat32Atomics || !atomicFeats.shaderBufferFloat32AtomicAdd) {
    throw std::runtime_error(
        "VK_EXT_shader_atomic_float missing required features: "
        "shaderBufferFloat32Atomics and shaderBufferFloat32AtomicAdd.");
  }

  // Ensure graphics queue family also supports compute.
  {
    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &qfCount, qfs.data());

    if ((qfs[graphicsQF].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) {
      throw std::runtime_error("Selected graphics queue family does not support compute. "
                               "This build uses a single queue, so a compute-capable graphics queue is required.");
    }
  }

  // ---------------- Device creation ----------------
  float qprio = 1.0f;

  VkDeviceQueueCreateInfo qciG{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qciG.queueFamilyIndex = graphicsQF;
  qciG.queueCount = 1;
  qciG.pQueuePriorities = &qprio;

  std::vector<const char*> devExts;
  pushUnique(devExts, baryExt);
  pushUnique(devExts, atomicFloatExt);

#ifdef __APPLE__
  const char* portabilitySubset = "VK_KHR_portability_subset";
  if (hasDeviceExtension(physicalDevice, portabilitySubset)) pushUnique(devExts, portabilitySubset);
#endif

#if defined(FUZZYDR_WITH_CUDA) && FUZZYDR_WITH_CUDA
  // Enable external memory extensions if available (used by the CUDA interop path).
  const bool hasExtMem =
      hasDeviceExtension(physicalDevice, VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME) &&
#if defined(_WIN32)
      hasDeviceExtension(physicalDevice, VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
#else
      hasDeviceExtension(physicalDevice, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
#endif

  if (hasExtMem) {
    pushUnique(devExts, VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
#if defined(_WIN32)
    pushUnique(devExts, VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
#else
    pushUnique(devExts, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
#endif
    extMemEnabled = true;
  } else {
    extMemEnabled = false; // CUDA interop paths detect this and throw cleanly
  }
#endif

#if defined(FUZZYDR_WITH_CUDA) && FUZZYDR_WITH_CUDA
  const bool hasExtSem =
      hasDeviceExtension(physicalDevice, VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME) &&
#if defined(_WIN32)
      hasDeviceExtension(physicalDevice, VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
#else
      hasDeviceExtension(physicalDevice, VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
#endif

  if (hasExtSem) {
    pushUnique(devExts, VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
#if defined(_WIN32)
    pushUnique(devExts, VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
#else
    pushUnique(devExts, VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
#endif
    extSemEnabled = true;
  } else {
    extSemEnabled = false; // CUDA interop paths detect this and throw cleanly
  }
#endif

  VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR baryFeatsEnable{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};
  baryFeatsEnable.fragmentShaderBarycentric = VK_TRUE;

  VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFeatsEnable{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
  atomicFeatsEnable.shaderBufferFloat32Atomics = VK_TRUE;
  atomicFeatsEnable.shaderBufferFloat32AtomicAdd = VK_TRUE;

  // Chain: atomicFeatsEnable -> baryFeatsEnable
  atomicFeatsEnable.pNext = &baryFeatsEnable;

  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.pNext = &atomicFeatsEnable;
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qciG;
  dci.enabledExtensionCount = (uint32_t)devExts.size();
  dci.ppEnabledExtensionNames = devExts.data();

  vkCheck(vkCreateDevice(physicalDevice, &dci, nullptr, &device), "vkCreateDevice");

#if defined(FUZZYDR_WITH_CUDA) && FUZZYDR_WITH_CUDA && defined(_WIN32)
  if (extMemEnabled) {
    auto p = (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR");
    if (!p) extMemEnabled = false;
  }
  if (extSemEnabled) {
    auto p = (PFN_vkGetSemaphoreWin32HandleKHR)vkGetDeviceProcAddr(device, "vkGetSemaphoreWin32HandleKHR");
    if (!p) extSemEnabled = false;
  }
#elif defined(FUZZYDR_WITH_CUDA) && FUZZYDR_WITH_CUDA
  // Mirror the Windows guard: verify that the FD-based interop function
  // pointers actually resolve before claiming interop is available.
  if (extMemEnabled) {
    auto p = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR");
    if (!p) extMemEnabled = false;
  }
  if (extSemEnabled) {
    auto p = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(device, "vkGetSemaphoreFdKHR");
    if (!p) extSemEnabled = false;
  }
#endif

  // queues
  vkGetDeviceQueue(device, graphicsQF, 0, &queue);

  // ---------------- Command pools ----------------
  VkCommandPoolCreateInfo pciG{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pciG.queueFamilyIndex = graphicsQF;
  pciG.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  vkCheck(vkCreateCommandPool(device, &pciG, nullptr, &cmdPool), "vkCreateCommandPool(graphics)");
}

VkContext::~VkContext() {
  if (device) {
    if (cmdPool) {
      vkDestroyCommandPool(device, cmdPool, nullptr);
      cmdPool = VK_NULL_HANDLE;
    }

    vkDestroyDevice(device, nullptr);
    device = VK_NULL_HANDLE;
  }

  if (instance) {
    vkDestroyInstance(instance, nullptr);
    instance = VK_NULL_HANDLE;
  }

  physicalDevice = VK_NULL_HANDLE;
  queue = VK_NULL_HANDLE;
  graphicsQF = UINT32_MAX;
  extMemEnabled = false;
  extSemEnabled = false;
}

#if defined(FUZZYDR_WITH_CUDA) && FUZZYDR_WITH_CUDA

#if defined(_WIN32)
  static constexpr VkExternalMemoryHandleTypeFlagBits kVkExtHandle =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
  static constexpr CUexternalMemoryHandleType kCuExtHandle =
      CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32;
#else
  static constexpr VkExternalMemoryHandleTypeFlagBits kVkExtHandle =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
  static constexpr CUexternalMemoryHandleType kCuExtHandle =
      CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
#endif

static inline void cuCheck(CUresult r, const char* what) {
  if (r != CUDA_SUCCESS) {
    const char* s = nullptr;
    cuGetErrorString(r, &s);
    throw std::runtime_error(std::string("CUDA driver error (") + what + "): " + (s ? s : "(unknown)"));
  }
}
static inline void rtCheck(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA runtime error (") + what + "): " + cudaGetErrorString(e));
  }
}

static inline void ensureCudaInitOnce() {
  static bool inited = false;
  if (!inited) {
    cuCheck(cuInit(0), "cuInit");
    rtCheck(cudaFree(nullptr), "cudaFree(nullptr)");
    inited = true;
  }
}

static void destroyCudaInterop(VulkanBuffer& b) {
  if (!b.exportToCuda) return;

  if (b.extMem) {
    cuCheck(cuDestroyExternalMemory(b.extMem), "cuDestroyExternalMemory");
    b.extMem = nullptr;
  }
  b.cudaPtr = 0;
  b.allocSize = 0;

#if defined(_WIN32)
  if (b.handle) {
    CloseHandle(b.handle);
    b.handle = nullptr;
  }
#else
  if (b.fd >= 0) {
    close(b.fd);
    b.fd = -1;
  }
#endif

  b.exportToCuda = false;
}

static void createCudaExportableBuffer(VkDevice dev,
                                       VkPhysicalDevice phys,
                                       VkDeviceSize bytes,
                                       VkBufferUsageFlags usage,
                                       VulkanBuffer& out) {
  ensureCudaInitOnce();

  // Buffer with external memory create-info
  VkExternalMemoryBufferCreateInfo extBuf{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
  extBuf.handleTypes = kVkExtHandle;

  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.pNext = &extBuf;
  bci.size = bytes;
  bci.usage = usage;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  vkCheck(vkCreateBuffer(dev, &bci, nullptr, &out.buf.buf), "vkCreateBuffer(ext)");

  out.buf.size = bytes;   // IMPORTANT: needed for descriptor ranges (CPU path sets this in createBuffer)

  VkMemoryRequirements req{};
  vkGetBufferMemoryRequirements(dev, out.buf.buf, &req);
  out.allocSize = req.size;

  // Exportable allocation
  VkExportMemoryAllocateInfo exportInfo{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
  exportInfo.handleTypes = kVkExtHandle;

  VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mai.pNext = &exportInfo;
  mai.allocationSize = req.size;
  mai.memoryTypeIndex = findMemoryTypeIndex(phys, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  vkCheck(vkAllocateMemory(dev, &mai, nullptr, &out.buf.mem), "vkAllocateMemory(ext)");
  vkCheck(vkBindBufferMemory(dev, out.buf.buf, out.buf.mem, 0), "vkBindBufferMemory(ext)");

  // Export handle from VkDeviceMemory
#if defined(_WIN32)
  auto vkGetMemoryWin32HandleKHR =
      (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(dev, "vkGetMemoryWin32HandleKHR");
  if (!vkGetMemoryWin32HandleKHR) throw std::runtime_error("vkGetMemoryWin32HandleKHR not found");

  VkMemoryGetWin32HandleInfoKHR hi{VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
  hi.memory = out.buf.mem;
  hi.handleType = kVkExtHandle;
  vkCheck(vkGetMemoryWin32HandleKHR(dev, &hi, &out.handle), "vkGetMemoryWin32HandleKHR");
  if (!out.handle) throw std::runtime_error("vkGetMemoryWin32HandleKHR returned null handle");
#else
  auto vkGetMemoryFdKHR =
      (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(dev, "vkGetMemoryFdKHR");
  if (!vkGetMemoryFdKHR) throw std::runtime_error("vkGetMemoryFdKHR not found");

  VkMemoryGetFdInfoKHR fi{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
  fi.memory = out.buf.mem;
  fi.handleType = kVkExtHandle;
  vkCheck(vkGetMemoryFdKHR(dev, &fi, &out.fd), "vkGetMemoryFdKHR");
  if (out.fd < 0) throw std::runtime_error("vkGetMemoryFdKHR returned invalid fd");

  // Guard: close the fd on any exception before CUDA takes ownership.
  auto fdGuard = scopeGuard([&]{ if (out.fd >= 0) { ::close(out.fd); out.fd = -1; } });
#endif

  // Import into CUDA
  CUDA_EXTERNAL_MEMORY_HANDLE_DESC cdesc{};
  cdesc.type = kCuExtHandle;
  cdesc.size = (unsigned long long)out.allocSize;

#if defined(_WIN32)
  cdesc.handle.win32.handle = out.handle;
  cdesc.handle.win32.name = nullptr;
#else
  cdesc.handle.fd = out.fd;
#endif

  cuCheck(cuImportExternalMemory(&out.extMem, &cdesc), "cuImportExternalMemory");

#ifndef _WIN32
  // For OPAQUE_FD, ownership transfers to CUDA upon successful import.
  out.fd = -1;
  fdGuard.dismiss();
#endif

  CUDA_EXTERNAL_MEMORY_BUFFER_DESC bdesc{};
  bdesc.offset = 0;
  bdesc.size = (unsigned long long)bytes;

  cuCheck(cuExternalMemoryGetMappedBuffer(&out.cudaPtr, out.extMem, &bdesc),
          "cuExternalMemoryGetMappedBuffer");
  if (out.cudaPtr == 0) throw std::runtime_error("cuExternalMemoryGetMappedBuffer returned null ptr");

  out.exportToCuda = true;
}

#endif // FUZZYDR_WITH_CUDA

void VulkanBuffer::reset(VkDevice device) {
#if defined(FUZZYDR_WITH_CUDA) && FUZZYDR_WITH_CUDA
  destroyCudaInterop(*this);
#endif

  if (device != VK_NULL_HANDLE) {
    destroyBuffer(device, buf);
  } else {
    buf = {};
  }

  bytes = 0;
  capacity = 0;
  usage = 0;
  props = 0;
}

void VulkanBuffer::ensure(VkDevice device,
                          VkPhysicalDevice physicalDevice,
                          VkDeviceSize requiredBytes,
                          VkBufferUsageFlags requiredUsage,
                          VkMemoryPropertyFlags requiredProps,
                          bool export_to_cuda) {
  // Fast path: the existing allocation is large enough and the flags match,
  // so only the logical size changes; no Vulkan free/alloc is needed.
  if (initialized() &&
      capacity >= requiredBytes &&
      usage == requiredUsage &&
      props == requiredProps &&
      exportToCuda == export_to_cuda) {
    bytes = requiredBytes;
    return;
  }

  reset(device);

  // Geometric growth (1.5x) with a 64 KiB floor, so small size changes are
  // absorbed without reallocating.
  VkDeviceSize newCapacity = requiredBytes + (requiredBytes >> 1);
  const VkDeviceSize minHeadroom = VkDeviceSize{64} * 1024;
  if (newCapacity < requiredBytes + minHeadroom) {
    newCapacity = requiredBytes + minHeadroom;
  }

  bytes = requiredBytes;
  capacity = newCapacity;
  usage = requiredUsage;
  props = requiredProps;

  if (!export_to_cuda) {
    createBuffer(device, physicalDevice, newCapacity, requiredUsage, requiredProps, buf);
    exportToCuda = false;
    return;
  }

#if !(defined(FUZZYDR_WITH_CUDA) && FUZZYDR_WITH_CUDA)
  throw std::runtime_error("VulkanBuffer::ensure: export_to_cuda requested but FUZZYDR_WITH_CUDA=0");
#else
  // Requires a VkContext created with external memory enabled.  That cannot be
  // checked here, so the call site must call ctx.requireExternalMemory().
  createCudaExportableBuffer(device, physicalDevice, newCapacity, requiredUsage, *this);
#endif
}

void VulkanBuffer::requireExact(VkDeviceSize requiredBytes, const char* what) const {
  if (!initialized() || bytes != requiredBytes) {
    throw std::runtime_error(std::string(what) + ": vk buffer inconsistent (uninitialized or size mismatch)");
  }
}

void VulkanBuffer::upload(VkDevice device, const void* src, VkDeviceSize nbytes, bool is_cuda, const char* what) const {
  if (!src) throw std::runtime_error(std::string(what) + ": null src");
  if (!initialized() || nbytes > bytes) throw std::runtime_error(std::string(what) + ": upload out of range");

  if (!is_cuda) {
    void* p = nullptr;
    vkCheck(vkMapMemory(device, buf.mem, 0, nbytes, 0, &p), what);
    std::memcpy(p, src, (size_t)nbytes);
    vkUnmapMemory(device, buf.mem);
    return;
  }

#if !(defined(FUZZYDR_WITH_CUDA) && FUZZYDR_WITH_CUDA)
  throw std::runtime_error(std::string(what) + ": CUDA upload requested but FUZZYDR_WITH_CUDA=0");
#else
  if (!exportToCuda || cudaPtr == 0) {
    throw std::runtime_error(std::string(what) + ": CUDA upload requested but buffer is not exportable/imported");
  }
  rtCheck(cudaMemcpy((void*)cudaPtr, src, (size_t)nbytes, cudaMemcpyDeviceToDevice), "cudaMemcpy D2D (upload)");
#endif
}

void VulkanBuffer::download(VkDevice device, void* dst, VkDeviceSize nbytes, bool is_cuda, const char* what) const {
  if (!dst) throw std::runtime_error(std::string(what) + ": null dst");
  if (!initialized() || nbytes > bytes) throw std::runtime_error(std::string(what) + ": download out of range");

  if (!is_cuda) {
    void* p = nullptr;
    vkCheck(vkMapMemory(device, buf.mem, 0, nbytes, 0, &p), what);
    std::memcpy(dst, p, (size_t)nbytes);
    vkUnmapMemory(device, buf.mem);
    return;
  }

#if !(defined(FUZZYDR_WITH_CUDA) && FUZZYDR_WITH_CUDA)
  throw std::runtime_error(std::string(what) + ": CUDA download requested but FUZZYDR_WITH_CUDA=0");
#else
  if (!exportToCuda || cudaPtr == 0) {
    throw std::runtime_error(std::string(what) + ": CUDA download requested but buffer is not exportable/imported");
  }
  rtCheck(cudaMemcpy(dst, (const void*)cudaPtr, (size_t)nbytes, cudaMemcpyDeviceToDevice), "cudaMemcpy D2D (download)");
#endif
}

void VulkanBuffer::memset(VkDevice device, int value, VkDeviceSize nbytes, bool is_cuda, const char* what) const {
  if (!initialized() || nbytes > bytes) throw std::runtime_error(std::string(what) + ": memset out of range");

  if (!is_cuda) {
    void* p = nullptr;
    vkCheck(vkMapMemory(device, buf.mem, 0, nbytes, 0, &p), what);
    std::memset(p, value, (size_t)nbytes);
    vkUnmapMemory(device, buf.mem);
    return;
  }

#if !(defined(FUZZYDR_WITH_CUDA) && FUZZYDR_WITH_CUDA)
  throw std::runtime_error(std::string(what) + ": CUDA memset requested but FUZZYDR_WITH_CUDA=0");
#else
  if (!exportToCuda || cudaPtr == 0) {
    throw std::runtime_error(std::string(what) + ": CUDA memset requested but buffer is not exportable/imported");
  }
  rtCheck(cudaMemset((void*)cudaPtr, value, (size_t)nbytes), "cudaMemset");
#endif
}

} // namespace fuzzydr

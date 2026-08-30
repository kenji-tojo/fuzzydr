// src/offline/offline.cpp
//
// Offscreen Vulkan implementations of fuzzydr_viewer.benchmark(...) and
// fuzzydr_viewer.OffscreenRenderer (StreamRenderer at the C++ layer).  Shares the
// viewer's SPIR-V but builds a minimal Vulkan stack on its own: no GLFW
// surface, no swapchain, no ImGui, no double-buffering.  SH compute is
// dispatched every frame with no camera-dirty guard so reported timings
// include the full inference pipeline.
//
// AA modes, selected via aa_mode:
//   0 = hw_msaa_4x - bresenham LINE_LIST + 4x hardware MSAA at native (W, H).
//   1 = gauss_msaa - bresenham LINE_LIST at (2W, 2H) into a 1-sample fp16
//                    target; a fullscreen 6x6 sigma=0.5 Gaussian pass downsamples
//                    to (W, H).
//   2 = none       - LINE_LIST at native (W, H), 1-sample (no AA).
//   3 = hw_msaa_2x - bresenham LINE_LIST + 2x hardware MSAA at native (W, H).
//
// Timing per frame = wall-clock around record-cb -> queue-submit -> wait.
//
// StreamRenderer composes a BenchmarkImpl: same init/cleanup, plus a
// permanently-mapped host-visible vertex-staging buffer.  Each frame() memcpys
// updated vert_attrs into the staging buffer; recordFrame's streaming hook
// prepends a vkCmdCopyBuffer + barrier so the new contents are visible to
// SH compute and the vertex shader.

#include "offline.h"

#include <vulkan/vulkan.h>

#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#  ifndef NOMINMAX
#  define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <dlfcn.h>
#  include <libgen.h>
#  include <limits.h>
#endif

namespace fuzzydr_viewer {
namespace {

// ---------------------------------------------------------------------------
// Helpers - deliberately duplicated from viewer.cpp so the benchmark TU is
// self-contained.  They are all trivial, so the duplication cost is low.
// ---------------------------------------------------------------------------
#define VK_CHECK(x)                                                          \
    do {                                                                     \
        VkResult _r = (x);                                                   \
        if (_r != VK_SUCCESS)                                                \
            throw std::runtime_error(std::string(__FILE__) + ":" +           \
                                     std::to_string(__LINE__) +              \
                                     "  VkResult=" + std::to_string(_r));    \
    } while (0)

std::string moduleDir() {
#ifdef _WIN32
    HMODULE hm = nullptr;
    if (!GetModuleHandleExA(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCSTR>(&moduleDir), &hm))
        throw std::runtime_error("GetModuleHandleExA failed");

    wchar_t wpath[MAX_PATH];
    DWORD len = GetModuleFileNameW(hm, wpath, MAX_PATH);
    if (len == 0 || len == MAX_PATH)
        throw std::runtime_error("GetModuleFileNameW failed");

    std::wstring ws(wpath, len);
    size_t pos = ws.find_last_of(L"\\/");
    std::wstring wdir = (pos == std::wstring::npos) ? L"." : ws.substr(0, pos);

    int needed = WideCharToMultiByte(CP_UTF8, 0, wdir.c_str(), (int)wdir.size(),
                                     nullptr, 0, nullptr, nullptr);
    if (needed <= 0) throw std::runtime_error("WideCharToMultiByte failed");
    std::string dir(needed, '\0');
    WideCharToMultiByte(CP_UTF8, 0, wdir.c_str(), (int)wdir.size(),
                        dir.data(), needed, nullptr, nullptr);
    return dir;
#else
    Dl_info info{};
    if (dladdr((void*)&moduleDir, &info) == 0 || !info.dli_fname)
        throw std::runtime_error("dladdr failed");
    char path[PATH_MAX];
    std::strncpy(path, info.dli_fname, sizeof(path));
    path[sizeof(path) - 1] = '\0';
    char* dir = ::dirname(path);
    if (!dir) throw std::runtime_error("dirname failed");
    return std::string(dir);
#endif
}

std::string shaderPath(const char* filename) {
#ifdef _WIN32
    return moduleDir() + "\\" + filename;
#else
    return moduleDir() + "/" + filename;
#endif
}

std::vector<uint32_t> readSpv(const std::string& path) {
    std::ifstream f(path, std::ios::ate | std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open SPIR-V: " + path);
    size_t bytes = static_cast<size_t>(f.tellg());
    if (bytes % 4 != 0)
        throw std::runtime_error("SPIR-V not 4-byte aligned: " + path);
    std::vector<uint32_t> buf(bytes / 4);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(buf.data()), bytes);
    return buf;
}

VkShaderModule makeShader(VkDevice dev, const std::vector<uint32_t>& spv) {
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = spv.size() * 4;
    ci.pCode    = spv.data();
    VkShaderModule m;
    VK_CHECK(vkCreateShaderModule(dev, &ci, nullptr, &m));
    return m;
}

uint32_t findMemType(VkPhysicalDevice pd,
                     uint32_t typeBits,
                     VkMemoryPropertyFlags flags) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((typeBits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & flags) == flags)
            return i;
    throw std::runtime_error("No suitable memory type");
}

// ---------------------------------------------------------------------------
// Push constant layouts.  The graphics block is the first 84 bytes of the
// viewer's layout - viewer.vert declares exactly these and reads only
// viewproj, so the same SPIR-V serves both paths.  The SH compute block is
// 16 bytes, identical to the viewer's.
// ---------------------------------------------------------------------------
struct GfxPushConst {
    float    viewproj[16];
    float    campos[3];
    uint32_t width;
    uint32_t height;
};
static_assert(sizeof(GfxPushConst) == 84, "GfxPushConst must be 84 bytes");

struct CompPushConst {
    float    campos[3];
    uint32_t nVerts;
};
static_assert(sizeof(CompPushConst) == 16, "CompPushConst must be 16 bytes");

// Offscreen attachment formats.  RGBA8 matches the viewer's swapchain so
// benchmark screenshots are apples-to-apples with presented frames.
// GAUSS_HIRES_FMT is the 2x-supersampled scene target under gauss_msaa;
// fp16 keeps bandwidth manageable while preserving enough precision in
// [0, 1] for the Gaussian resolve to converge cleanly.  Both formats are
// mandatory Vulkan core support as colour attachment + sampled (NEAREST).
constexpr VkFormat COLOR_FMT       = VK_FORMAT_R8G8B8A8_UNORM;
constexpr VkFormat GAUSS_HIRES_FMT = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr VkFormat DEPTH_FMT       = VK_FORMAT_D32_SFLOAT;

// ---------------------------------------------------------------------------
// BenchmarkImpl
// ---------------------------------------------------------------------------
struct BenchmarkImpl {
    VkInstance       inst  = VK_NULL_HANDLE;
    VkPhysicalDevice pDev  = VK_NULL_HANDLE;
    VkDevice         dev   = VK_NULL_HANDLE;
    uint32_t         qFam  = 0;
    VkQueue          queue = VK_NULL_HANDLE;

    // AA config - resolved in run() from BenchmarkInput::aa_mode.
    //   gaussMsaa : aa_mode == 1   (2x supersample + Gaussian resolve)
    //   samples   : 4x for hw_msaa_4x, 2x for hw_msaa_2x, 1x otherwise
    uint32_t              aaMode    = 0;
    bool                  gaussMsaa = false;
    VkSampleCountFlagBits samples   = VK_SAMPLE_COUNT_4_BIT;

    // Dimensions.
    uint32_t width  = 0;   // output resolution (screenshot / resolveImg size)
    uint32_t height = 0;
    uint32_t sceneW = 0;   // scene-pass resolution (2x width  under gauss_msaa)
    uint32_t sceneH = 0;   // scene-pass resolution (2x height under gauss_msaa)

    // Scene colour attachment.  N-sample MSAA target in hw_msaa mode, or a
    // 1-sample 2x-res target in gauss_msaa mode.
    VkImage        sceneColorImg  = VK_NULL_HANDLE;
    VkDeviceMemory sceneColorMem  = VK_NULL_HANDLE;
    VkImageView    sceneColorView = VK_NULL_HANDLE;
    // Depth at the same dims + sample count as sceneColor.
    VkImage        sceneDepthImg  = VK_NULL_HANDLE;
    VkDeviceMemory sceneDepthMem  = VK_NULL_HANDLE;
    VkImageView    sceneDepthView = VK_NULL_HANDLE;
    // 1-sample RGBA8 output at native (width, height).  Target of either the
    // HW MSAA resolve attachment (hw_msaa) or the Gaussian-resolve pass
    // (gauss_msaa).  Read back for screenshots.
    VkImage        resolveImg     = VK_NULL_HANDLE;
    VkDeviceMemory resolveMem     = VK_NULL_HANDLE;
    VkImageView    resolveView    = VK_NULL_HANDLE;

    VkRenderPass   sceneRp        = VK_NULL_HANDLE;
    VkFramebuffer  sceneFb        = VK_NULL_HANDLE;

    // Gaussian-resolve pass (populated only when gaussMsaa).
    VkSampler             gaussSampler     = VK_NULL_HANDLE;
    VkRenderPass          gaussRp          = VK_NULL_HANDLE;
    VkFramebuffer         gaussFb          = VK_NULL_HANDLE;
    VkDescriptorSetLayout gaussDescLayout  = VK_NULL_HANDLE;
    VkPipelineLayout      gaussPipeLayout  = VK_NULL_HANDLE;
    VkDescriptorPool      gaussDescPool    = VK_NULL_HANDLE;
    VkDescriptorSet       gaussDescSet     = VK_NULL_HANDLE;
    VkPipeline            gaussPipeline    = VK_NULL_HANDLE;

    // Graphics line pipeline - uses viewer.vert / viewer.frag.  Topology
    // (LINE_LIST or LINE_STRIP + primitive restart) is fixed by
    // ``lineTopology`` at pipeline-build time; the bench keeps just one.
    VkDescriptorSetLayout gfxDescLayout = VK_NULL_HANDLE;  // empty, for PC parity
    VkPipelineLayout      gfxPipeLayout = VK_NULL_HANDLE;
    VkPipeline            linePipeline  = VK_NULL_HANDLE;

    // SH compute.
    VkDescriptorSetLayout compDescLayout = VK_NULL_HANDLE;
    VkPipelineLayout      compPipeLayout = VK_NULL_HANDLE;
    VkPipeline            compPipeline   = VK_NULL_HANDLE;
    VkDescriptorPool      compDescPool   = VK_NULL_HANDLE;
    VkDescriptorSet       compDescSet    = VK_NULL_HANDLE;

    // Geometry buffers (lines-only - no face support in the benchmark path).
    // ``nIndices`` is the number of uint32 indices in ``lBuf``; under LINE_LIST
    // it equals 2x the number of segments, under LINE_STRIP it includes the
    // primitive-restart sentinels.  ``lineTopology`` selects between the
    // two pipelines built in initGfxPipelines().
    uint32_t       nVerts = 0, nIndices = 0;
    uint32_t       lineTopology = 0;
    VkBuffer       vBuf = VK_NULL_HANDLE;    VkDeviceMemory vMem    = VK_NULL_HANDLE;
    VkBuffer       shBuf = VK_NULL_HANDLE;   VkDeviceMemory shMem   = VK_NULL_HANDLE;
    VkBuffer       evalBuf = VK_NULL_HANDLE; VkDeviceMemory evalMem = VK_NULL_HANDLE;
    VkBuffer       lBuf = VK_NULL_HANDLE;    VkDeviceMemory lMem    = VK_NULL_HANDLE;

    // Command buffer + fence.
    VkCommandPool   cmdPool = VK_NULL_HANDLE;
    VkCommandBuffer cmdBuf  = VK_NULL_HANDLE;
    VkFence         fence   = VK_NULL_HANDLE;

    // Host-visible staging buffer for screenshot readback.
    VkBuffer       screenshotBuf = VK_NULL_HANDLE;
    VkDeviceMemory screenshotMem = VK_NULL_HANDLE;

    // Streaming hook.  When non-null, recordFrame() prepends a copy from
    // streamStgBuf into vBuf + a barrier.  Owned by StreamRenderer::Impl,
    // not by BenchmarkImpl::cleanup.
    VkBuffer       streamStgBuf = VK_NULL_HANDLE;

    BenchmarkOutput run(const BenchmarkInput& in);

    void initInstance();
    void pickPhysDevice();
    void initDevice();
    void clampSamples();
    void initImages();
    void initSceneRp();
    void initSceneFb();
    void initGaussResolve();   // no-op unless gaussMsaa
    void initGfxPipelines();
    void initComputePipeline(const std::string& spvName);
    void initComputeDescriptors();
    void uploadGeom(const BenchmarkInput& in);
    void initCmds();

    void mkBuf(VkDeviceSize sz, VkBufferUsageFlags usage,
               VkMemoryPropertyFlags memFlags,
               VkBuffer& buf, VkDeviceMemory& mem);
    void stageUpload(const void* data, VkDeviceSize sz,
                     VkBufferUsageFlags dstUsage,
                     VkBuffer& buf, VkDeviceMemory& mem);

    void recordFrame(const float viewproj[16], const float eye[3],
                     bool skip_sh);
    void captureScreenshot(const float viewproj[16], const float eye[3],
                           uint8_t* dst);

    // Read resolveImg into ``dst`` (must be width*height*4 bytes).  Caller
    // must have just rendered into resolveImg (its layout is
    // COLOR_ATTACHMENT_OPTIMAL); this transitions to TRANSFER_SRC_OPTIMAL,
    // copies, and submits - resolveImg ends in TRANSFER_SRC_OPTIMAL.
    void copyResolveToHost(uint8_t* dst);

    void cleanup();
};

// ---------------------------------------------------------------------------
// Vulkan init
// ---------------------------------------------------------------------------
void BenchmarkImpl::initInstance() {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "fuzzydr_viewer-benchmark";
    app.apiVersion       = VK_API_VERSION_1_2;

    std::vector<const char*> exts;
    VkInstanceCreateFlags instFlags = 0;
#ifdef __APPLE__
    exts.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    instFlags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.flags                   = instFlags;
    ci.pApplicationInfo        = &app;
    ci.enabledExtensionCount   = uint32_t(exts.size());
    ci.ppEnabledExtensionNames = exts.empty() ? nullptr : exts.data();
    VK_CHECK(vkCreateInstance(&ci, nullptr, &inst));
}

void BenchmarkImpl::pickPhysDevice() {
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    if (n == 0) throw std::runtime_error("No Vulkan-capable GPU found");
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(inst, &n, devs.data());

    VkPhysicalDevice fallback = VK_NULL_HANDLE;
    uint32_t fallbackQ = 0;
    for (auto d : devs) {
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(d, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qfp(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(d, &qn, qfp.data());
        for (uint32_t i = 0; i < qn; i++) {
            if (!(qfp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(d, &props);
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                pDev = d;  qFam = i;  return;
            }
            if (fallback == VK_NULL_HANDLE) { fallback = d; fallbackQ = i; }
            break;
        }
    }
    if (fallback == VK_NULL_HANDLE)
        throw std::runtime_error("No suitable GPU queue family found");
    pDev = fallback;  qFam = fallbackQ;
}

void BenchmarkImpl::initDevice() {
    float prio = 1.f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qFam;
    qci.queueCount       = 1;
    qci.pQueuePriorities = &prio;

    std::vector<const char*> devExts;
#ifdef __APPLE__
    devExts.push_back("VK_KHR_portability_subset");
#endif

    VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    ci.queueCreateInfoCount    = 1;
    ci.pQueueCreateInfos       = &qci;
    ci.enabledExtensionCount   = uint32_t(devExts.size());
    ci.ppEnabledExtensionNames = devExts.empty() ? nullptr : devExts.data();
    VK_CHECK(vkCreateDevice(pDev, &ci, nullptr, &dev));
    vkGetDeviceQueue(dev, qFam, 0, &queue);
}

// hw_msaa_4x asks for 4x and hw_msaa_2x for 2x, clamped to what the
// colour+depth combo supports.  gauss_msaa and none use 1 sample (run() has
// already initialised ``samples``, so the loop below is a no-op for them).
void BenchmarkImpl::clampSamples() {
    if (samples == VK_SAMPLE_COUNT_1_BIT) return;

    auto queryCounts = [&](VkFormat fmt, VkImageUsageFlags usage) {
        VkImageFormatProperties fp{};
        if (vkGetPhysicalDeviceImageFormatProperties(
                pDev, fmt, VK_IMAGE_TYPE_2D,
                VK_IMAGE_TILING_OPTIMAL, usage, 0, &fp) != VK_SUCCESS)
            return static_cast<VkSampleCountFlags>(VK_SAMPLE_COUNT_1_BIT);
        return fp.sampleCounts;
    };
    VkSampleCountFlags counts =
        queryCounts(COLOR_FMT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) &
        queryCounts(DEPTH_FMT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);

    for (auto c : {VK_SAMPLE_COUNT_4_BIT,
                   VK_SAMPLE_COUNT_2_BIT,
                   VK_SAMPLE_COUNT_1_BIT}) {
        if (c <= samples && (counts & c)) { samples = c; return; }
    }
    samples = VK_SAMPLE_COUNT_1_BIT;
}

static void mkImage(VkDevice dev, VkPhysicalDevice pDev,
                    uint32_t w, uint32_t h,
                    VkFormat fmt, VkSampleCountFlagBits samples,
                    VkImageUsageFlags usage, VkImageAspectFlags aspect,
                    VkImage& outImg, VkDeviceMemory& outMem, VkImageView& outView) {
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType     = VK_IMAGE_TYPE_2D;
    ci.format        = fmt;
    ci.extent        = {w, h, 1};
    ci.mipLevels     = 1;
    ci.arrayLayers   = 1;
    ci.samples       = samples;
    ci.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ci.usage         = usage;
    ci.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK(vkCreateImage(dev, &ci, nullptr, &outImg));

    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(dev, outImg, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize  = mr.size;
    ai.memoryTypeIndex = findMemType(pDev, mr.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(dev, &ai, nullptr, &outMem));
    vkBindImageMemory(dev, outImg, outMem, 0);

    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image            = outImg;
    vci.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    vci.format           = fmt;
    vci.subresourceRange = {aspect, 0, 1, 0, 1};
    VK_CHECK(vkCreateImageView(dev, &vci, nullptr, &outView));
}

void BenchmarkImpl::initImages() {
    const bool msaa = (samples != VK_SAMPLE_COUNT_1_BIT);

    // Scene colour.  MSAA attachments are TRANSIENT so tilers keep them in
    // tile memory.  gauss_msaa's 2x-res colour is SAMPLED (so the resolve
    // pass can read it) and uses GAUSS_HIRES_FMT (fp16) for headroom; u8
    // here would quantise the supersampled buffer below the resolve filter.
    const VkFormat sceneFmt = gaussMsaa ? GAUSS_HIRES_FMT : COLOR_FMT;
    VkImageUsageFlags sceneUsage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
        (msaa      ? VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT : 0) |
        (gaussMsaa ? VK_IMAGE_USAGE_SAMPLED_BIT              : 0);
    mkImage(dev, pDev, sceneW, sceneH, sceneFmt, samples,
            sceneUsage, VK_IMAGE_ASPECT_COLOR_BIT,
            sceneColorImg, sceneColorMem, sceneColorView);

    // Depth at the same dims + sample count as scene colour.
    mkImage(dev, pDev, sceneW, sceneH, DEPTH_FMT, samples,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            VK_IMAGE_ASPECT_DEPTH_BIT,
            sceneDepthImg, sceneDepthMem, sceneDepthView);

    // 1-sample RGBA8 output at (width, height).  TRANSFER_SRC for screenshot
    // readback; populated by either HW MSAA resolve, direct scene render
    // (hw_msaa 1-sample fallback), or the Gaussian-resolve pass.
    mkImage(dev, pDev, width, height, COLOR_FMT, VK_SAMPLE_COUNT_1_BIT,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            resolveImg, resolveMem, resolveView);
}

// Scene render pass.
//   hw_msaa w/ MSAA : [sceneColor@Nx, sceneDepth@Nx, resolveImg@1x]; HW
//                     resolve attaches at index 2.
//   hw_msaa 1-sample: [resolveImg, sceneDepth].  No sceneColor used; the
//                     scene target IS the resolve output.
//   gauss_msaa      : [sceneColor@1x@2xres, sceneDepth@1x@2xres].  Final
//                     layout = SHADER_READ so the Gaussian pass samples it.
//
// hw_msaa 1-sample reuses the sceneColor allocation for depth only; the
// colour target is routed to resolveImg by rebuilding the framebuffer.
void BenchmarkImpl::initSceneRp() {
    const bool msaa = (samples != VK_SAMPLE_COUNT_1_BIT);

    // Subpass deps: external->subpass (wait for prior scene/resolve reads)
    // and subpass->external (publish writes to the downstream pass: HW
    // MSAA consumer, gauss-resolve sampler, or screenshot transfer).
    std::array<VkSubpassDependency, 2> deps{};
    deps[0].srcSubpass    = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass    = 0;
    deps[0].srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[0].dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    deps[1].srcSubpass    = 0;
    deps[1].dstSubpass    = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[1].dstStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                            VK_PIPELINE_STAGE_TRANSFER_BIT;
    deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                            VK_ACCESS_TRANSFER_READ_BIT;

    const VkImageLayout sceneColorFinal = gaussMsaa
        ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription depthAtt{};
    depthAtt.format         = DEPTH_FMT;
    depthAtt.samples        = samples;
    depthAtt.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAtt.storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAtt.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAtt.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAtt.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    if (msaa) {
        // MSAA colour -> HW resolve -> resolveImg.
        std::array<VkAttachmentDescription, 3> att{};
        att[0].format         = COLOR_FMT;
        att[0].samples        = samples;
        att[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        att[0].storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        att[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        att[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        att[0].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        att[0].finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        att[1]                = depthAtt;
        att[2].format         = COLOR_FMT;
        att[2].samples        = VK_SAMPLE_COUNT_1_BIT;
        att[2].loadOp         = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        att[2].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        att[2].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        att[2].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        att[2].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        att[2].finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkAttachmentReference depRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkAttachmentReference resRef{2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkSubpassDescription sub{};
        sub.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sub.colorAttachmentCount    = 1;
        sub.pColorAttachments       = &colRef;
        sub.pResolveAttachments     = &resRef;
        sub.pDepthStencilAttachment = &depRef;

        VkRenderPassCreateInfo ci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        ci.attachmentCount = uint32_t(att.size());
        ci.pAttachments    = att.data();
        ci.subpassCount    = 1;
        ci.pSubpasses      = &sub;
        ci.dependencyCount = uint32_t(deps.size());
        ci.pDependencies   = deps.data();
        VK_CHECK(vkCreateRenderPass(dev, &ci, nullptr, &sceneRp));
        return;
    }

    // 1-sample path: hw_msaa fallback writes directly into resolveImg;
    // gauss_msaa writes into sceneColor (fp16, then sampled by the resolve).
    std::array<VkAttachmentDescription, 2> att{};
    att[0].format         = gaussMsaa ? GAUSS_HIRES_FMT : COLOR_FMT;
    att[0].samples        = VK_SAMPLE_COUNT_1_BIT;
    att[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    att[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[0].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    att[0].finalLayout    = sceneColorFinal;
    att[1]                = depthAtt;

    VkAttachmentReference colRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription sub{};
    sub.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount    = 1;
    sub.pColorAttachments       = &colRef;
    sub.pDepthStencilAttachment = &depRef;

    VkRenderPassCreateInfo ci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    ci.attachmentCount = uint32_t(att.size());
    ci.pAttachments    = att.data();
    ci.subpassCount    = 1;
    ci.pSubpasses      = &sub;
    ci.dependencyCount = uint32_t(deps.size());
    ci.pDependencies   = deps.data();
    VK_CHECK(vkCreateRenderPass(dev, &ci, nullptr, &sceneRp));
}

void BenchmarkImpl::initSceneFb() {
    const bool msaa = (samples != VK_SAMPLE_COUNT_1_BIT);
    VkFramebufferCreateInfo ci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    ci.renderPass = sceneRp;
    ci.width      = sceneW;
    ci.height     = sceneH;
    ci.layers     = 1;

    // hw_msaa fallback (1 sample) uses resolveView as the colour target;
    // gauss_msaa uses sceneColorView (2x-res).
    VkImageView atts3[3] = {sceneColorView, sceneDepthView, resolveView};
    VkImageView atts2[2] = {gaussMsaa ? sceneColorView : resolveView,
                            sceneDepthView};
    if (msaa) {
        ci.attachmentCount = 3;
        ci.pAttachments    = atts3;
    } else {
        ci.attachmentCount = 2;
        ci.pAttachments    = atts2;
    }
    VK_CHECK(vkCreateFramebuffer(dev, &ci, nullptr, &sceneFb));
}

// ---------------------------------------------------------------------------
// Gaussian-resolve pass - fullscreen tri samples the 2x-res sceneColor and
// writes into resolveImg at (width, height).  Filter is the 6x6 sigma=0.5
// kernel implemented in shaders/viewer_gauss_resolve.frag.
// ---------------------------------------------------------------------------
void BenchmarkImpl::initGaussResolve() {
    if (!gaussMsaa) return;

    // Sampler - LINEAR for the resolve frag's bilinear-tap reduction.
    {
        VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sci.magFilter    = VK_FILTER_LINEAR;
        sci.minFilter    = VK_FILTER_LINEAR;
        sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VK_CHECK(vkCreateSampler(dev, &sci, nullptr, &gaussSampler));
    }

    {
        VkDescriptorSetLayoutBinding b{};
        b.binding         = 0;
        b.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b.descriptorCount = 1;
        b.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo dlci{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dlci.bindingCount = 1;
        dlci.pBindings    = &b;
        VK_CHECK(vkCreateDescriptorSetLayout(dev, &dlci, nullptr, &gaussDescLayout));

        VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        lci.setLayoutCount = 1;
        lci.pSetLayouts    = &gaussDescLayout;
        VK_CHECK(vkCreatePipelineLayout(dev, &lci, nullptr, &gaussPipeLayout));

        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
        VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pci.maxSets       = 1;
        pci.poolSizeCount = 1;
        pci.pPoolSizes    = &ps;
        VK_CHECK(vkCreateDescriptorPool(dev, &pci, nullptr, &gaussDescPool));

        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool     = gaussDescPool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &gaussDescLayout;
        VK_CHECK(vkAllocateDescriptorSets(dev, &ai, &gaussDescSet));

        VkDescriptorImageInfo ii{};
        ii.sampler     = gaussSampler;
        ii.imageView   = sceneColorView;
        ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet          = gaussDescSet;
        w.dstBinding      = 0;
        w.descriptorCount = 1;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo      = &ii;
        vkUpdateDescriptorSets(dev, 1, &w, 0, nullptr);
    }

    {
        VkAttachmentDescription att{};
        att.format         = COLOR_FMT;
        att.samples        = VK_SAMPLE_COUNT_1_BIT;
        att.loadOp         = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        att.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        att.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        att.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        att.finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkSubpassDescription sub{};
        sub.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sub.colorAttachmentCount = 1;
        sub.pColorAttachments    = &colRef;

        VkSubpassDependency dep{};
        dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
        dep.dstSubpass    = 0;
        dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        VkRenderPassCreateInfo ci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        ci.attachmentCount = 1;
        ci.pAttachments    = &att;
        ci.subpassCount    = 1;
        ci.pSubpasses      = &sub;
        ci.dependencyCount = 1;
        ci.pDependencies   = &dep;
        VK_CHECK(vkCreateRenderPass(dev, &ci, nullptr, &gaussRp));
    }

    {
        VkFramebufferCreateInfo ci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        ci.renderPass      = gaussRp;
        ci.attachmentCount = 1;
        ci.pAttachments    = &resolveView;
        ci.width           = width;
        ci.height          = height;
        ci.layers          = 1;
        VK_CHECK(vkCreateFramebuffer(dev, &ci, nullptr, &gaussFb));
    }

    {
        VkShaderModule vs = makeShader(dev, readSpv(shaderPath("viewer_gauss_resolve.vert.spv")));
        VkShaderModule fs = makeShader(dev, readSpv(shaderPath("viewer_gauss_resolve.frag.spv")));

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vs;  stages[0].pName = "main";
        stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fs;  stages[1].pName = "main";

        VkPipelineVertexInputStateCreateInfo vi{
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo vp{
            VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount = 1;  vp.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{
            VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode    = VK_CULL_MODE_NONE;
        rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth   = 1.f;
        VkPipelineMultisampleStateCreateInfo ms{
            VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{
            VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        VkPipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                             VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo cb{
            VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = 1;  cb.pAttachments = &cba;
        VkDynamicState dynSt[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dyn{
            VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dyn.dynamicStateCount = 2;  dyn.pDynamicStates = dynSt;

        VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        ci.stageCount          = 2;
        ci.pStages             = stages;
        ci.pVertexInputState   = &vi;
        ci.pInputAssemblyState = &ia;
        ci.pViewportState      = &vp;
        ci.pRasterizationState = &rs;
        ci.pMultisampleState   = &ms;
        ci.pDepthStencilState  = &ds;
        ci.pColorBlendState    = &cb;
        ci.pDynamicState       = &dyn;
        ci.layout              = gaussPipeLayout;
        ci.renderPass          = gaussRp;
        VK_CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr,
                                           &gaussPipeline));
        vkDestroyShaderModule(dev, vs, nullptr);
        vkDestroyShaderModule(dev, fs, nullptr);
    }
}

// ---------------------------------------------------------------------------
// Graphics pipelines (face + bresen) - use viewer.vert / viewer.frag.
// ---------------------------------------------------------------------------
void BenchmarkImpl::initGfxPipelines() {
    VkDescriptorSetLayoutCreateInfo dlci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    VK_CHECK(vkCreateDescriptorSetLayout(dev, &dlci, nullptr, &gfxDescLayout));

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pcr.offset     = 0;
    pcr.size       = sizeof(GfxPushConst);

    VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    lci.setLayoutCount         = 1;
    lci.pSetLayouts            = &gfxDescLayout;
    lci.pushConstantRangeCount = 1;
    lci.pPushConstantRanges    = &pcr;
    VK_CHECK(vkCreatePipelineLayout(dev, &lci, nullptr, &gfxPipeLayout));

    VkShaderModule vsMod = makeShader(dev, readSpv(shaderPath("viewer.vert.spv")));
    VkShaderModule fsMod = makeShader(dev, readSpv(shaderPath("viewer.frag.spv")));

    auto buildVboPipeline = [&](VkPrimitiveTopology topo,
                                bool                primitiveRestart) -> VkPipeline {
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vsMod;  stages[0].pName = "main";
        stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fsMod;  stages[1].pName = "main";

        // Vertex layout: stride 28, {pos (loc 0, @0), rgb (loc 1, @16)}.
        // The radius slot at offset 12 is unused by viewer.vert.
        VkVertexInputBindingDescription bind{0, 28, VK_VERTEX_INPUT_RATE_VERTEX};
        VkVertexInputAttributeDescription attrs[2]{};
        attrs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT,  0};
        attrs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, 16};

        VkPipelineVertexInputStateCreateInfo vi{
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vi.vertexBindingDescriptionCount   = 1;
        vi.pVertexBindingDescriptions      = &bind;
        vi.vertexAttributeDescriptionCount = 2;
        vi.pVertexAttributeDescriptions    = attrs;

        VkPipelineInputAssemblyStateCreateInfo ia{
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology               = topo;
        ia.primitiveRestartEnable = primitiveRestart ? VK_TRUE : VK_FALSE;

        VkPipelineViewportStateCreateInfo vp{
            VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount = 1;
        vp.scissorCount  = 1;

        VkPipelineRasterizationStateCreateInfo rs{
            VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode    = VK_CULL_MODE_NONE;
        rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth   = 1.f;

        VkPipelineMultisampleStateCreateInfo ms{
            VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = samples;

        VkPipelineDepthStencilStateCreateInfo ds{
            VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds.depthTestEnable  = VK_TRUE;
        ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp   = VK_COMPARE_OP_LESS;

        VkPipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                             VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo cb{
            VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        cb.attachmentCount = 1;
        cb.pAttachments    = &cba;

        VkDynamicState dynSt[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dyn{
            VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dyn.dynamicStateCount = 2;
        dyn.pDynamicStates    = dynSt;

        VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        ci.stageCount          = 2;
        ci.pStages             = stages;
        ci.pVertexInputState   = &vi;
        ci.pInputAssemblyState = &ia;
        ci.pViewportState      = &vp;
        ci.pRasterizationState = &rs;
        ci.pMultisampleState   = &ms;
        ci.pDepthStencilState  = &ds;
        ci.pColorBlendState    = &cb;
        ci.pDynamicState       = &dyn;
        ci.layout              = gfxPipeLayout;
        ci.renderPass          = sceneRp;

        VkPipeline pipe;
        VK_CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr, &pipe));
        return pipe;
    };

    if (lineTopology == 1) {
        // LINE_STRIP + primitive restart: UINT32_MAX in the index buffer
        // (the standard sentinel for VK_INDEX_TYPE_UINT32) breaks the strip
        // between maximal walks.
        linePipeline = buildVboPipeline(VK_PRIMITIVE_TOPOLOGY_LINE_STRIP, true);
    } else {
        linePipeline = buildVboPipeline(VK_PRIMITIVE_TOPOLOGY_LINE_LIST, false);
    }

    vkDestroyShaderModule(dev, vsMod, nullptr);
    vkDestroyShaderModule(dev, fsMod, nullptr);
}

// ---------------------------------------------------------------------------
// SH compute pipeline - shared binding layout with the viewer.
// ---------------------------------------------------------------------------
void BenchmarkImpl::initComputePipeline(const std::string& spvName) {
    std::array<VkDescriptorSetLayoutBinding, 3> binds{};
    for (uint32_t i = 0; i < 3; i++) {
        binds[i].binding         = i;
        binds[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dlci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dlci.bindingCount = uint32_t(binds.size());
    dlci.pBindings    = binds.data();
    VK_CHECK(vkCreateDescriptorSetLayout(dev, &dlci, nullptr, &compDescLayout));

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset     = 0;
    pcr.size       = sizeof(CompPushConst);

    VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    lci.setLayoutCount         = 1;
    lci.pSetLayouts            = &compDescLayout;
    lci.pushConstantRangeCount = 1;
    lci.pPushConstantRanges    = &pcr;
    VK_CHECK(vkCreatePipelineLayout(dev, &lci, nullptr, &compPipeLayout));

    VkShaderModule cs = makeShader(dev, readSpv(shaderPath(spvName.c_str())));
    VkPipelineShaderStageCreateInfo stage{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = cs;
    stage.pName  = "main";

    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage  = stage;
    ci.layout = compPipeLayout;
    VK_CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr, &compPipeline));
    vkDestroyShaderModule(dev, cs, nullptr);
}

void BenchmarkImpl::initComputeDescriptors() {
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pci.maxSets       = 1;
    pci.poolSizeCount = 1;
    pci.pPoolSizes    = &ps;
    VK_CHECK(vkCreateDescriptorPool(dev, &pci, nullptr, &compDescPool));

    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool     = compDescPool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts        = &compDescLayout;
    VK_CHECK(vkAllocateDescriptorSets(dev, &ai, &compDescSet));

    VkDescriptorBufferInfo bufs[3]{};
    bufs[0].buffer = vBuf;     bufs[0].range = VK_WHOLE_SIZE;
    bufs[1].buffer = shBuf;    bufs[1].range = VK_WHOLE_SIZE;
    bufs[2].buffer = evalBuf;  bufs[2].range = VK_WHOLE_SIZE;

    std::array<VkWriteDescriptorSet, 3> writes{};
    for (uint32_t i = 0; i < 3; i++) {
        writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet          = compDescSet;
        writes[i].dstBinding      = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &bufs[i];
    }
    vkUpdateDescriptorSets(dev, uint32_t(writes.size()), writes.data(), 0, nullptr);
}

// ---------------------------------------------------------------------------
// Buffers
// ---------------------------------------------------------------------------
void BenchmarkImpl::mkBuf(VkDeviceSize sz, VkBufferUsageFlags usage,
                           VkMemoryPropertyFlags memFlags,
                           VkBuffer& buf, VkDeviceMemory& mem) {
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size        = sz;
    bci.usage       = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(dev, &bci, nullptr, &buf));

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev, buf, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize  = mr.size;
    ai.memoryTypeIndex = findMemType(pDev, mr.memoryTypeBits, memFlags);
    VK_CHECK(vkAllocateMemory(dev, &ai, nullptr, &mem));
    vkBindBufferMemory(dev, buf, mem, 0);
}

void BenchmarkImpl::stageUpload(const void* data, VkDeviceSize sz,
                                 VkBufferUsageFlags dstUsage,
                                 VkBuffer& buf, VkDeviceMemory& mem) {
    VkBuffer       stg  = VK_NULL_HANDLE;
    VkDeviceMemory stgM = VK_NULL_HANDLE;
    mkBuf(sz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          stg, stgM);
    void* p = nullptr;
    VK_CHECK(vkMapMemory(dev, stgM, 0, sz, 0, &p));
    std::memcpy(p, data, size_t(sz));
    vkUnmapMemory(dev, stgM);

    mkBuf(sz, dstUsage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
          buf, mem);

    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool        = cmdPool;
    cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer tmp;
    VK_CHECK(vkAllocateCommandBuffers(dev, &cai, &tmp));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(tmp, &bi));
    VkBufferCopy cp{0, 0, sz};
    vkCmdCopyBuffer(tmp, stg, buf, 1, &cp);
    VK_CHECK(vkEndCommandBuffer(tmp));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &tmp;
    VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue));
    vkFreeCommandBuffers(dev, cmdPool, 1, &tmp);
    vkDestroyBuffer(dev, stg, nullptr);
    vkFreeMemory(dev, stgM, nullptr);
}

void BenchmarkImpl::uploadGeom(const BenchmarkInput& in) {
    nVerts   = in.num_verts;
    nIndices = in.num_indices;

    {
        VkDeviceSize sz = VkDeviceSize(nVerts) * 7 * sizeof(float);
        stageUpload(in.vert_attrs.data(), sz,
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    vBuf, vMem);
    }

    if (in.sh_cols > 0) {
        VkDeviceSize sz = VkDeviceSize(nVerts) * in.sh_cols * sizeof(float);
        std::vector<float> transposed(size_t(nVerts) * in.sh_cols);
        const float* src = in.sh_coeffs.data();
        float*       dst = transposed.data();
        for (uint32_t vi = 0; vi < nVerts; vi++)
            for (uint32_t c = 0; c < in.sh_cols; c++)
                dst[size_t(c) * nVerts + vi] = src[size_t(vi) * in.sh_cols + c];
        stageUpload(transposed.data(), sz,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    shBuf, shMem);

        mkBuf(VkDeviceSize(nVerts) * 7 * sizeof(float),
              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
              VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
              evalBuf, evalMem);
    }

    {
        VkDeviceSize sz = VkDeviceSize(nIndices) * sizeof(uint32_t);
        stageUpload(in.lines.data(), sz, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                    lBuf, lMem);
    }
}

void BenchmarkImpl::initCmds() {
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = qFam;
    VK_CHECK(vkCreateCommandPool(dev, &pci, nullptr, &cmdPool));

    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool        = cmdPool;
    cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(dev, &cai, &cmdBuf));

    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_CHECK(vkCreateFence(dev, &fci, nullptr, &fence));
}

// ---------------------------------------------------------------------------
// Frame recording
// ---------------------------------------------------------------------------
void BenchmarkImpl::recordFrame(const float viewproj[16], const float eye[3],
                                 bool skip_sh) {
    VK_CHECK(vkResetCommandBuffer(cmdBuf, 0));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VK_CHECK(vkBeginCommandBuffer(cmdBuf, &bi));

    // Streaming hook: copy host-mapped staging into vBuf and gate the
    // downstream stages on the transfer.  vBuf is read by SH compute as a
    // storage buffer (sh_cols > 0) and/or by the vertex shader as a vertex
    // attribute (sh_cols == 0); the dst mask covers both.
    if (streamStgBuf != VK_NULL_HANDLE) {
        VkBufferCopy cp{};
        cp.srcOffset = 0;
        cp.dstOffset = 0;
        cp.size      = VkDeviceSize(nVerts) * 7 * sizeof(float);
        vkCmdCopyBuffer(cmdBuf, streamStgBuf, vBuf, 1, &cp);

        VkBufferMemoryBarrier mbar{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        mbar.srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT;
        mbar.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT |
                                   VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
        mbar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        mbar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        mbar.buffer              = vBuf;
        mbar.offset              = 0;
        mbar.size                = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmdBuf,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                             VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                             0, 0, nullptr, 1, &mbar, 0, nullptr);
    }

    // SH compute dispatch + buffer barrier so the scene pipeline sees its writes.
    const bool sceneHasSh = (shBuf != VK_NULL_HANDLE);
    if (sceneHasSh && !skip_sh) {
        vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, compPipeline);
        vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                                compPipeLayout, 0, 1, &compDescSet, 0, nullptr);
        CompPushConst cpc{};
        cpc.campos[0] = eye[0];
        cpc.campos[1] = eye[1];
        cpc.campos[2] = eye[2];
        cpc.nVerts    = nVerts;
        vkCmdPushConstants(cmdBuf, compPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(cpc), &cpc);
        uint32_t groups = (nVerts + 255u) / 256u;
        vkCmdDispatch(cmdBuf, groups, 1, 1);

        VkBufferMemoryBarrier bar{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        bar.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
        bar.dstAccessMask       = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT |
                                  VK_ACCESS_SHADER_READ_BIT;
        bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bar.buffer              = evalBuf;
        bar.offset              = 0;
        bar.size                = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cmdBuf,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_VERTEX_INPUT_BIT |
                             VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                             0, 0, nullptr, 1, &bar, 0, nullptr);
    }

    // -- Scene pass -----------------------------------------------------
    const bool msaa = (samples != VK_SAMPLE_COUNT_1_BIT);
    VkClearValue clears[3] = {};
    clears[0].color        = {{1.f, 1.f, 1.f, 1.f}};
    clears[1].depthStencil = {1.f, 0};

    VkRenderPassBeginInfo rpbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rpbi.renderPass        = sceneRp;
    rpbi.framebuffer       = sceneFb;
    rpbi.renderArea.extent = {sceneW, sceneH};
    rpbi.clearValueCount   = msaa ? 3u : 2u;
    rpbi.pClearValues      = clears;
    vkCmdBeginRenderPass(cmdBuf, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport vport{0.f, 0.f, float(sceneW), float(sceneH), 0.f, 1.f};
    VkRect2D scissor{{0, 0}, {sceneW, sceneH}};
    vkCmdSetViewport(cmdBuf, 0, 1, &vport);
    vkCmdSetScissor (cmdBuf, 0, 1, &scissor);

    GfxPushConst gpc{};
    std::memcpy(gpc.viewproj, viewproj, sizeof(gpc.viewproj));
    gpc.campos[0] = eye[0];
    gpc.campos[1] = eye[1];
    gpc.campos[2] = eye[2];
    gpc.width  = sceneW;
    gpc.height = sceneH;

    VkBuffer drawVbo = sceneHasSh ? evalBuf : vBuf;
    VkDeviceSize off = 0;

    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, linePipeline);
    vkCmdPushConstants(cmdBuf, gfxPipeLayout, VK_SHADER_STAGE_VERTEX_BIT,
                       0, sizeof(gpc), &gpc);
    vkCmdBindVertexBuffers(cmdBuf, 0, 1, &drawVbo, &off);
    vkCmdBindIndexBuffer  (cmdBuf, lBuf, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmdBuf, nIndices, 1, 0, 0, 0);
    vkCmdEndRenderPass(cmdBuf);

    // -- Gauss resolve pass (gauss_msaa only) ---------------------------
    if (gaussMsaa) {
        VkRenderPassBeginInfo gb{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        gb.renderPass        = gaussRp;
        gb.framebuffer       = gaussFb;
        gb.renderArea.extent = {width, height};
        gb.clearValueCount   = 0;
        vkCmdBeginRenderPass(cmdBuf, &gb, VK_SUBPASS_CONTENTS_INLINE);
        VkViewport outVp{0.f, 0.f, float(width), float(height), 0.f, 1.f};
        VkRect2D   outSc{{0, 0}, {width, height}};
        vkCmdSetViewport(cmdBuf, 0, 1, &outVp);
        vkCmdSetScissor (cmdBuf, 0, 1, &outSc);
        vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, gaussPipeline);
        vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                gaussPipeLayout, 0, 1, &gaussDescSet, 0, nullptr);
        vkCmdDraw(cmdBuf, 3, 1, 0, 0);
        vkCmdEndRenderPass(cmdBuf);
    }

    VK_CHECK(vkEndCommandBuffer(cmdBuf));
}

// ---------------------------------------------------------------------------
// run - top-level benchmark loop.
// ---------------------------------------------------------------------------
BenchmarkOutput BenchmarkImpl::run(const BenchmarkInput& in) {
    if (in.num_verts == 0)
        throw std::runtime_error("benchmark: num_verts must be > 0");
    if (in.num_indices == 0)
        throw std::runtime_error("benchmark: num_indices must be > 0 (line-only path)");
    if (in.line_topology > 1)
        throw std::runtime_error("benchmark: line_topology must be 0 (LIST) or 1 (STRIP)");
    if (in.num_views == 0)
        throw std::runtime_error("benchmark: num_views must be > 0");

    width        = in.width;
    height       = in.height;
    aaMode       = in.aa_mode;
    gaussMsaa    = (aaMode == 1);
    lineTopology = in.line_topology;
    // Sample count per mode; clampSamples() narrows further if the device
    // or attachment combo can't honor it.
    switch (aaMode) {
        case 0:  samples = VK_SAMPLE_COUNT_4_BIT;  break;  // hw_msaa_4x
        case 3:  samples = VK_SAMPLE_COUNT_2_BIT;  break;  // hw_msaa_2x
        default: samples = VK_SAMPLE_COUNT_1_BIT;  break;  // gauss_msaa / bresenham
    }
    sceneW = gaussMsaa ? width  * 2u : width;
    sceneH = gaussMsaa ? height * 2u : height;

    initInstance();
    pickPhysDevice();
    initDevice();
    clampSamples();
    initImages();
    initSceneRp();
    initSceneFb();

    initCmds();
    uploadGeom(in);

    if (in.sh_cols > 0) {
        std::string spvName = in.compute_shader.empty()
                                ? "viewer_sh_eval.comp.spv"
                                : in.compute_shader;
        initComputePipeline(spvName);
        initComputeDescriptors();
    }

    initGfxPipelines();
    initGaussResolve();  // no-op unless gaussMsaa

    const uint32_t total_frames = in.warmup + in.measure;
    BenchmarkOutput out;
    out.times.assign(size_t(in.num_views) * total_frames, 0.f);
    if (in.measure_nosh)
        out.times_nosh.assign(size_t(in.num_views) * total_frames, 0.f);
    if (in.capture_screenshots)
        out.images.assign(size_t(in.num_views) *
                          size_t(in.height) * size_t(in.width) * 4, 0);

    auto submitAndWait = [&]() {
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cmdBuf;
        VK_CHECK(vkQueueSubmit(queue, 1, &si, fence));
        VK_CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(dev, 1, &fence));
    };

    for (uint32_t v = 0; v < in.num_views; v++) {
        const float* vp  = in.viewprojs.data() + size_t(v) * 16;
        const float* eye = in.eyes.data()      + size_t(v) * 3;

        for (uint32_t i = 0; i < total_frames; i++) {
            auto t0 = std::chrono::steady_clock::now();
            recordFrame(vp, eye, /*skip_sh=*/false);
            submitAndWait();
            auto t1 = std::chrono::steady_clock::now();
            out.times[size_t(v) * total_frames + i] =
                std::chrono::duration<float>(t1 - t0).count();
        }

        if (in.measure_nosh) {
            for (uint32_t i = 0; i < total_frames; i++) {
                auto t0 = std::chrono::steady_clock::now();
                recordFrame(vp, eye, /*skip_sh=*/true);
                submitAndWait();
                auto t1 = std::chrono::steady_clock::now();
                out.times_nosh[size_t(v) * total_frames + i] =
                    std::chrono::duration<float>(t1 - t0).count();
            }
        }

        if (in.capture_screenshots) {
            uint8_t* dst = out.images.data() +
                size_t(v) * size_t(in.height) * size_t(in.width) * 4;
            captureScreenshot(vp, eye, dst);
        }
    }

    cleanup();
    return out;
}

// ---------------------------------------------------------------------------
// copyResolveToHost - one untimed submit: barrier resolveImg from
// COLOR_ATTACHMENT_OPTIMAL -> TRANSFER_SRC_OPTIMAL, vkCmdCopyImageToBuffer
// into a lazy-allocated host-visible staging buffer, then memcpy to ``dst``.
// resolveImg ends in TRANSFER_SRC_OPTIMAL - fine because the next render
// pass uses initialLayout=UNDEFINED.
// ---------------------------------------------------------------------------
void BenchmarkImpl::copyResolveToHost(uint8_t* dst) {
    const VkDeviceSize pixelBytes = VkDeviceSize(width) * height * 4;

    if (screenshotBuf == VK_NULL_HANDLE) {
        mkBuf(pixelBytes,
              VK_BUFFER_USAGE_TRANSFER_DST_BIT,
              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
              screenshotBuf, screenshotMem);
    }

    VK_CHECK(vkResetCommandBuffer(cmdBuf, 0));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VK_CHECK(vkBeginCommandBuffer(cmdBuf, &bi));

    VkImageMemoryBarrier ibar{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    ibar.srcAccessMask       = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    ibar.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;
    ibar.oldLayout           = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ibar.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    ibar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ibar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    ibar.image               = resolveImg;
    ibar.subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmdBuf,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &ibar);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent      = {width, height, 1};
    vkCmdCopyImageToBuffer(cmdBuf, resolveImg,
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           screenshotBuf, 1, &region);

    VK_CHECK(vkEndCommandBuffer(cmdBuf));
    {
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cmdBuf;
        VK_CHECK(vkQueueSubmit(queue, 1, &si, fence));
        VK_CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(dev, 1, &fence));
    }

    void* p = nullptr;
    VK_CHECK(vkMapMemory(dev, screenshotMem, 0, pixelBytes, 0, &p));
    std::memcpy(dst, p, size_t(pixelBytes));
    vkUnmapMemory(dev, screenshotMem);
}

// ---------------------------------------------------------------------------
// captureScreenshot - records + submits one *untimed* render frame, then
// copies resolveImg into ``dst`` via copyResolveToHost.
// ---------------------------------------------------------------------------
void BenchmarkImpl::captureScreenshot(const float viewproj[16], const float eye[3],
                                       uint8_t* dst) {
    // Submit 1: render into resolveImg.  Sharing cmdBuf with the timed loop
    // is safe because screenshots are recorded outside the measured region.
    recordFrame(viewproj, eye, /*skip_sh=*/false);
    {
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &cmdBuf;
        VK_CHECK(vkQueueSubmit(queue, 1, &si, fence));
        VK_CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(dev, 1, &fence));
    }

    // Submit 2: transition resolveImg + copy to host-visible staging buffer.
    copyResolveToHost(dst);
}

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------
void BenchmarkImpl::cleanup() {
    if (dev) vkDeviceWaitIdle(dev);

    auto destroyBuf = [&](VkBuffer& b, VkDeviceMemory& m) {
        if (b) { vkDestroyBuffer(dev, b, nullptr); b = VK_NULL_HANDLE; }
        if (m) { vkFreeMemory   (dev, m, nullptr); m = VK_NULL_HANDLE; }
    };
    destroyBuf(vBuf,    vMem);
    destroyBuf(shBuf,   shMem);
    destroyBuf(evalBuf, evalMem);
    destroyBuf(lBuf,    lMem);
    destroyBuf(screenshotBuf, screenshotMem);

    if (fence)   { vkDestroyFence(dev, fence, nullptr); fence = VK_NULL_HANDLE; }
    if (cmdPool) { vkDestroyCommandPool(dev, cmdPool, nullptr); cmdPool = VK_NULL_HANDLE; }

    if (compDescPool)   { vkDestroyDescriptorPool    (dev, compDescPool,   nullptr); compDescPool   = VK_NULL_HANDLE; }
    if (compPipeline)   { vkDestroyPipeline          (dev, compPipeline,   nullptr); compPipeline   = VK_NULL_HANDLE; }
    if (compPipeLayout) { vkDestroyPipelineLayout    (dev, compPipeLayout, nullptr); compPipeLayout = VK_NULL_HANDLE; }
    if (compDescLayout) { vkDestroyDescriptorSetLayout(dev, compDescLayout, nullptr); compDescLayout = VK_NULL_HANDLE; }

    if (linePipeline) { vkDestroyPipeline(dev, linePipeline, nullptr); linePipeline = VK_NULL_HANDLE; }
    if (gfxPipeLayout)  { vkDestroyPipelineLayout(dev, gfxPipeLayout, nullptr); gfxPipeLayout = VK_NULL_HANDLE; }
    if (gfxDescLayout)  { vkDestroyDescriptorSetLayout(dev, gfxDescLayout, nullptr); gfxDescLayout = VK_NULL_HANDLE; }

    if (gaussPipeline)    { vkDestroyPipeline          (dev, gaussPipeline,    nullptr); gaussPipeline    = VK_NULL_HANDLE; }
    if (gaussDescPool)    { vkDestroyDescriptorPool    (dev, gaussDescPool,    nullptr); gaussDescPool    = VK_NULL_HANDLE; }
    if (gaussPipeLayout)  { vkDestroyPipelineLayout    (dev, gaussPipeLayout,  nullptr); gaussPipeLayout  = VK_NULL_HANDLE; }
    if (gaussDescLayout)  { vkDestroyDescriptorSetLayout(dev, gaussDescLayout, nullptr); gaussDescLayout  = VK_NULL_HANDLE; }
    if (gaussFb)          { vkDestroyFramebuffer       (dev, gaussFb,          nullptr); gaussFb          = VK_NULL_HANDLE; }
    if (gaussRp)          { vkDestroyRenderPass        (dev, gaussRp,          nullptr); gaussRp          = VK_NULL_HANDLE; }
    if (gaussSampler)     { vkDestroySampler           (dev, gaussSampler,     nullptr); gaussSampler     = VK_NULL_HANDLE; }

    if (sceneFb) { vkDestroyFramebuffer(dev, sceneFb, nullptr); sceneFb = VK_NULL_HANDLE; }
    if (sceneRp) { vkDestroyRenderPass (dev, sceneRp, nullptr); sceneRp = VK_NULL_HANDLE; }

    auto destroyImg = [&](VkImage& img, VkDeviceMemory& mem, VkImageView& view) {
        if (view) { vkDestroyImageView(dev, view, nullptr); view = VK_NULL_HANDLE; }
        if (img)  { vkDestroyImage    (dev, img,  nullptr); img  = VK_NULL_HANDLE; }
        if (mem)  { vkFreeMemory      (dev, mem,  nullptr); mem  = VK_NULL_HANDLE; }
    };
    destroyImg(sceneColorImg, sceneColorMem, sceneColorView);
    destroyImg(sceneDepthImg, sceneDepthMem, sceneDepthView);
    destroyImg(resolveImg,    resolveMem,    resolveView);

    if (dev)  { vkDestroyDevice(dev, nullptr);     dev  = VK_NULL_HANDLE; }
    if (inst) { vkDestroyInstance(inst, nullptr);  inst = VK_NULL_HANDLE; }
}

} // anonymous namespace

BenchmarkOutput benchmark(const BenchmarkInput& in) {
    BenchmarkImpl impl;
    return impl.run(in);
}

// ---------------------------------------------------------------------------
// StreamRenderer
// ---------------------------------------------------------------------------
struct StreamRenderer::Impl {
    BenchmarkImpl bench;

    // Permanently-mapped HOST_COHERENT staging buffer sized to one [N, 7]
    // vert_attrs frame.  bench.streamStgBuf points at streamStgBuf so that
    // BenchmarkImpl::recordFrame prepends the staging->vBuf copy + barrier.
    VkBuffer       streamStgBuf    = VK_NULL_HANDLE;
    VkDeviceMemory streamStgMem    = VK_NULL_HANDLE;
    void*          streamStgMapped = nullptr;
    VkDeviceSize   streamStgBytes  = 0;

    void destroy();
};

void StreamRenderer::Impl::destroy() {
    if (bench.dev) vkDeviceWaitIdle(bench.dev);

    // Detach the streaming hook before bench.cleanup() so a stray recordFrame
    // can't reference the about-to-be-destroyed staging buffer.
    bench.streamStgBuf = VK_NULL_HANDLE;

    if (streamStgMapped && streamStgMem) {
        vkUnmapMemory(bench.dev, streamStgMem);
        streamStgMapped = nullptr;
    }
    if (streamStgBuf) {
        vkDestroyBuffer(bench.dev, streamStgBuf, nullptr);
        streamStgBuf = VK_NULL_HANDLE;
    }
    if (streamStgMem) {
        vkFreeMemory(bench.dev, streamStgMem, nullptr);
        streamStgMem = VK_NULL_HANDLE;
    }

    bench.cleanup();
}

StreamRenderer::StreamRenderer(const StreamRendererInput& in)
    : impl_(std::make_unique<Impl>())
{
    if (in.num_verts == 0)
        throw std::runtime_error("StreamRenderer: num_verts must be > 0");
    if (in.num_indices == 0)
        throw std::runtime_error("StreamRenderer: num_indices must be > 0 (line-only)");
    if (in.line_topology > 1)
        throw std::runtime_error("StreamRenderer: line_topology must be 0 (LIST) or 1 (STRIP)");
    if (in.width == 0 || in.height == 0)
        throw std::runtime_error("StreamRenderer: width/height must be > 0");
    if (in.aa_mode > 3)
        throw std::runtime_error(
            "StreamRenderer: aa_mode must be 0..3 "
            "(hw_msaa_4x / gauss_msaa / bresenham / hw_msaa_2x)");
    const size_t expected_attrs = size_t(in.num_verts) * 7;
    if (in.vert_attrs.size() < expected_attrs)
        throw std::runtime_error(
            "StreamRenderer: vert_attrs.size() < num_verts * 7");

    BenchmarkImpl& b = impl_->bench;

    b.width        = in.width;
    b.height       = in.height;
    b.aaMode       = in.aa_mode;
    b.gaussMsaa    = (b.aaMode == 1);
    b.lineTopology = in.line_topology;
    switch (b.aaMode) {
        case 0:  b.samples = VK_SAMPLE_COUNT_4_BIT;  break;  // hw_msaa_4x
        case 3:  b.samples = VK_SAMPLE_COUNT_2_BIT;  break;  // hw_msaa_2x
        default: b.samples = VK_SAMPLE_COUNT_1_BIT;  break;  // gauss / bresenham
    }
    b.sceneW = b.gaussMsaa ? b.width  * 2u : b.width;
    b.sceneH = b.gaussMsaa ? b.height * 2u : b.height;

    b.initInstance();
    b.pickPhysDevice();
    b.initDevice();
    b.clampSamples();
    b.initImages();
    b.initSceneRp();
    b.initSceneFb();
    b.initCmds();

    // uploadGeom reads only the geometry fields of BenchmarkInput, so a
    // shimmed temporary is safe and avoids forking a near-identical upload.
    BenchmarkInput tmp;
    tmp.num_verts     = in.num_verts;
    tmp.num_indices   = in.num_indices;
    tmp.line_topology = in.line_topology;
    tmp.sh_cols        = in.sh_cols;
    tmp.compute_shader = in.compute_shader;
    tmp.vert_attrs     = in.vert_attrs;
    tmp.sh_coeffs      = in.sh_coeffs;
    tmp.lines          = in.lines;
    b.uploadGeom(tmp);

    if (in.sh_cols > 0) {
        const std::string spvName = in.compute_shader.empty()
                                      ? "viewer_sh_eval.comp.spv"
                                      : in.compute_shader;
        b.initComputePipeline(spvName);
        b.initComputeDescriptors();
    }
    b.initGfxPipelines();
    b.initGaussResolve();   // no-op unless gaussMsaa

    impl_->streamStgBytes = VkDeviceSize(in.num_verts) * 7 * sizeof(float);
    b.mkBuf(impl_->streamStgBytes,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            impl_->streamStgBuf, impl_->streamStgMem);
    VK_CHECK(vkMapMemory(b.dev, impl_->streamStgMem, 0,
                         impl_->streamStgBytes, 0, &impl_->streamStgMapped));
    b.streamStgBuf = impl_->streamStgBuf;
}

StreamRenderer::~StreamRenderer() {
    if (impl_) impl_->destroy();
}

StreamFrameOutput StreamRenderer::frame(const float* vert_attrs,
                                         const float  viewproj[16],
                                         const float  eye[3],
                                         bool         capture)
{
    BenchmarkImpl& b = impl_->bench;

    // HOST_COHERENT memory: the subsequent vkQueueSubmit observes the writes
    // without a manual vkFlushMappedMemoryRanges.
    std::memcpy(impl_->streamStgMapped, vert_attrs, size_t(impl_->streamStgBytes));

    auto t0 = std::chrono::steady_clock::now();
    b.recordFrame(viewproj, eye, /*skip_sh=*/false);
    {
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &b.cmdBuf;
        VK_CHECK(vkQueueSubmit(b.queue, 1, &si, b.fence));
        VK_CHECK(vkWaitForFences(b.dev, 1, &b.fence, VK_TRUE, UINT64_MAX));
        VK_CHECK(vkResetFences(b.dev, 1, &b.fence));
    }
    auto t1 = std::chrono::steady_clock::now();

    StreamFrameOutput out;
    out.time_seconds = std::chrono::duration<float>(t1 - t0).count();

    if (capture) {
        out.image.assign(size_t(b.width) * size_t(b.height) * 4, 0);
        b.copyResolveToHost(out.image.data());
    }

    return out;
}

uint32_t StreamRenderer::width()     const { return impl_->bench.width;  }
uint32_t StreamRenderer::height()    const { return impl_->bench.height; }
uint32_t StreamRenderer::num_verts() const { return impl_->bench.nVerts; }

} // namespace fuzzydr_viewer

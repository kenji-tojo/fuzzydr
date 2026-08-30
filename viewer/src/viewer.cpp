// src/viewer.cpp
#include "viewer.h"
#include "camera.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <algorithm>
#include <array>
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

// ---------------------------------------------------------------------------
// Resolve the directory that contains the _core extension at runtime so SPV
// files are always found next to the module.
// ---------------------------------------------------------------------------
inline std::string moduleDir() {
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

inline std::string shaderPath(const char* filename) {
#ifdef _WIN32
    return moduleDir() + "\\" + filename;
#else
    return moduleDir() + "/" + filename;
#endif
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
#define VK_CHECK(x)                                                          \
    do {                                                                     \
        VkResult _r = (x);                                                   \
        if (_r != VK_SUCCESS)                                                \
            throw std::runtime_error(std::string(__FILE__) + ":" +           \
                                     std::to_string(__LINE__) +              \
                                     "  VkResult=" + std::to_string(_r));    \
    } while (0)

static std::vector<uint32_t> readSpv(const std::string& path) {
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

static VkShaderModule makeShader(VkDevice dev,
                                 const std::vector<uint32_t>& spv) {
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = spv.size() * 4;
    ci.pCode    = spv.data();
    VkShaderModule m;
    VK_CHECK(vkCreateShaderModule(dev, &ci, nullptr, &m));
    return m;
}

static uint32_t findMemType(VkPhysicalDevice pd,
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
// Push constant layout shared by the graphics shaders.
// viewer.vert uses only viewproj (first 64 bytes).
// viewer_line_quads.vert uses all 84 bytes.
// ---------------------------------------------------------------------------
struct PushConst {
    float    viewproj[16];  // offset  0 - 64 bytes
    float    campos[3];     // offset 64 - 12 bytes
    uint32_t width;         // offset 76 -  4 bytes
    uint32_t height;        // offset 80 -  4 bytes
};
static_assert(sizeof(PushConst) == 84, "PushConst must be 84 bytes");

// ---------------------------------------------------------------------------
// Push constants for the SH evaluation compute pass.  Mirrors the
// `layout(push_constant) uniform PC` block in viewer_sh_eval.comp.
// ---------------------------------------------------------------------------
struct CompPushConst {
    float    campos[3];
    uint32_t nVerts;
};
static_assert(sizeof(CompPushConst) == 16, "CompPushConst must be 16 bytes");

// ---------------------------------------------------------------------------
// AA mode enum - mirrors the int form accepted by the Python API.
// ---------------------------------------------------------------------------
enum AaMode : uint32_t {
    AA_HW_MSAA    = 0,
    AA_GAUSS_MSAA = 1,
    AA_NONE       = 2,   // hardware path, sample count pinned to 1
};

// ---------------------------------------------------------------------------
// ViewerImpl
// ---------------------------------------------------------------------------
struct ViewerImpl {

    // -- GLFW -----------------------------------------------------------------
    GLFWwindow* win   = nullptr;
    bool        dirty = false;  // set by resize callback

    // -- Vulkan core ----------------------------------------------------------
    VkInstance       inst  = VK_NULL_HANDLE;
    VkSurfaceKHR     surf  = VK_NULL_HANDLE;
    VkPhysicalDevice pDev  = VK_NULL_HANDLE;
    VkDevice         dev   = VK_NULL_HANDLE;
    uint32_t         qFam  = 0;
    VkQueue          queue = VK_NULL_HANDLE;

    // -- Swapchain ------------------------------------------------------------
    VkSwapchainKHR           sc    = VK_NULL_HANDLE;
    std::vector<VkImage>     scImg;
    std::vector<VkImageView> scView;
    VkFormat                 scFmt = VK_FORMAT_UNDEFINED;
    VkExtent2D               scExt = {};

    // -- AA mode --------------------------------------------------------------
    // hw_msaa   : scene renders directly into scView[i] with hardware MSAA
    //             (samples 2 or 4); depth and MSAA colour are N-sample.
    // gauss_msaa: scene renders at 2x (scExt.width*2, scExt.height*2) into a
    //             1-sample fp16 image; a fullscreen Gaussian-resolve pass
    //             downsamples to scView[i].
    // none      : the hw_msaa path with the sample count pinned to 1.
    AaMode                aaMode        = AA_HW_MSAA;
    AaMode                pendingAaMode = AA_HW_MSAA;
    VkSampleCountFlagBits msaaSamples        = VK_SAMPLE_COUNT_2_BIT;
    VkSampleCountFlagBits pendingMsaaSamples = VK_SAMPLE_COUNT_2_BIT;

    // Scene-pass extent (= scExt in hw_msaa, 2xscExt in gauss_msaa).
    VkExtent2D sceneExt = {};

    // -- Depth buffer (scene-pass dims, scene-pass sample count) --------------
    VkImage        depImg  = VK_NULL_HANDLE;
    VkDeviceMemory depMem  = VK_NULL_HANDLE;
    VkImageView    depView = VK_NULL_HANDLE;

    // -- MSAA colour (only when the resolved sample count is > 1) -------------
    VkImage        msaaImg  = VK_NULL_HANDLE;
    VkDeviceMemory msaaMem  = VK_NULL_HANDLE;
    VkImageView    msaaView = VK_NULL_HANDLE;

    // -- Scene colour (gauss_msaa only, 2xscExt, 1-sample, sampled by resolve)-
    VkImage        sceneColorImg  = VK_NULL_HANDLE;
    VkDeviceMemory sceneColorMem  = VK_NULL_HANDLE;
    VkImageView    sceneColorView = VK_NULL_HANDLE;

    // -- Render passes --------------------------------------------------------
    // sceneRp  : draws the scene.
    //   hw_msaa  -> writes scView[i] (N-sample MSAA resolve or direct 1-sample).
    //              finalLayout = COLOR_ATTACHMENT_OPTIMAL (handed to ImGui).
    //   gauss_msaa -> writes sceneColorImg at 2x res.
    //              finalLayout = SHADER_READ so gaussRp can sample it.
    // gaussRp  : fullscreen Gaussian resolve, writes scView[i].  finalLayout =
    //            COLOR_ATTACHMENT_OPTIMAL.  Only built under gauss_msaa.
    // imguiRp  : LOADs scView[i], writes it, finalLayout = PRESENT_SRC_KHR.
    VkRenderPass               sceneRp = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> sceneFbs;
    VkRenderPass               imguiRp = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> imguiFbs;

    // Gauss-resolve pass resources (built only under gauss_msaa).
    VkRenderPass               gaussRp          = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> gaussFbs;
    VkSampler                  gaussSampler     = VK_NULL_HANDLE;
    VkDescriptorSetLayout      gaussDescLayout  = VK_NULL_HANDLE;
    VkPipelineLayout           gaussPipeLayout  = VK_NULL_HANDLE;
    VkDescriptorPool           gaussDescPool    = VK_NULL_HANDLE;
    VkDescriptorSet            gaussDescSet     = VK_NULL_HANDLE;
    VkPipeline                 gaussPipeline    = VK_NULL_HANDLE;

    // -- Shared pipeline layout (face + bresen + quad-line) -------------------
    // Descriptor set 0: storage buffers for the quad-line SSBO reads.
    VkDescriptorSetLayout descLayout = VK_NULL_HANDLE;
    VkPipelineLayout      pipeLayout = VK_NULL_HANDLE;

    // -- Graphics pipelines ---------------------------------------------------
    VkPipeline facePipeline     = VK_NULL_HANDLE;
    VkPipeline facePipelineCull = VK_NULL_HANDLE;
    VkPipeline bresenPipeline   = VK_NULL_HANDLE;
    VkPipeline quadLinePipeline = VK_NULL_HANDLE;

    // -- Per-vertex data + SH/shading compute ---------------------------------
    // When hasSh=true a compute pass runs every frame before the scene pass.
    // shBuf   : SSBO  float[nVerts * shCols]   - per-vertex data, transposed.
    // evalBuf : SSBO + VERTEX_BUFFER, float[nVerts * 7]  - compute output,
    //           bound as VBO for graphics draws and as SSBO binding 0 of the
    //           quad-line descriptor set.
    bool           hasSh    = false;
    uint32_t       shCols   = 0;
    VkBuffer       shBuf    = VK_NULL_HANDLE;  VkDeviceMemory shMem   = VK_NULL_HANDLE;
    VkBuffer       evalBuf  = VK_NULL_HANDLE;  VkDeviceMemory evalMem = VK_NULL_HANDLE;

    VkDescriptorSetLayout compDescLayout   = VK_NULL_HANDLE;
    VkPipelineLayout      compPipeLayout   = VK_NULL_HANDLE;
    VkPipeline            compPipeline     = VK_NULL_HANDLE;
    VkDescriptorPool      compDescPool     = VK_NULL_HANDLE;
    VkDescriptorSet       compDescSet      = VK_NULL_HANDLE;

    // -- Quad-line descriptor set ---------------------------------------------
    VkDescriptorPool descPool = VK_NULL_HANDLE;
    VkDescriptorSet  descSet  = VK_NULL_HANDLE;

    // -- Commands + sync (double-buffered) ------------------------------------
    static constexpr int MF = 2;
    VkCommandPool   cmdPool    = VK_NULL_HANDLE;
    VkCommandBuffer cmdBuf[MF] = {};
    VkSemaphore     imgSem[MF] = {};
    VkSemaphore     renSem[MF] = {};
    VkFence         fence[MF]  = {};
    int             frameIdx   = 0;

    // -- Geometry buffers -----------------------------------------------------
    VkBuffer vBuf = VK_NULL_HANDLE;  VkDeviceMemory vMem = VK_NULL_HANDLE;
    VkBuffer fBuf = VK_NULL_HANDLE;  VkDeviceMemory fMem = VK_NULL_HANDLE;
    VkBuffer lBuf = VK_NULL_HANDLE;  VkDeviceMemory lMem = VK_NULL_HANDLE;
    // Bresenham draws from the strip rebake when the caller supplies one; the
    // quad styles always index lBuf by instance.  See SceneData::line_strip.
    VkBuffer lStripBuf = VK_NULL_HANDLE;  VkDeviceMemory lStripMem = VK_NULL_HANDLE;
    uint32_t nVerts = 0, nFaces = 0, nLines = 0, nStripIndices = 0;

    // -- Camera ---------------------------------------------------------------
    OrbitCamera cam;
    double prevX = 0, prevY = 0;
    bool   btnL = false, btnR = false, btnM = false;
    float  prevCampos[3] = {1e30f, 1e30f, 1e30f};

    // -- ImGui ----------------------------------------------------------------
    VkDescriptorPool imguiPool = VK_NULL_HANDLE;

    // -- UI state -------------------------------------------------------------
    float bgColor[3]  = {1.0f, 1.0f, 1.0f};
    int   lineMode    = 0;   // 0 = Bresenham, 1 = Quads uniform, 2 = Quads per-vertex
    float lineRadius  = 5e-4f;  // world-space quad half-width for lineMode 1
    bool  hasVertexRadius = false;  // per-vertex mode offered only when true
    // A uniform line width is applied by rewriting the radius column of vBuf
    // rather than by a shader branch, so the shaders stay exactly as fuzzydr's.
    // scene outlives the run() call, so the untouched radii remain available.
    const SceneData* scene = nullptr;
    float uploadedRadius = -1.f;   // radius now in vBuf; < 0 means per-vertex
    bool  backfaceCull = false;
#ifndef __APPLE__
    bool  vsync        = true;
    bool  pendingVsync = true;
    bool  mailboxAvail = false;
#endif

    // -- Screenshot -----------------------------------------------------------
    bool screenshotSupported = false;   // surface allows TRANSFER_SRC
    bool screenshotPending   = false;   // set by the panel, cleared once written
    bool screenshotRecorded  = false;   // a copy is in flight this frame
    int  screenshotIndex     = 0;
    VkBuffer       shotBuf = VK_NULL_HANDLE;
    VkDeviceMemory shotMem = VK_NULL_HANDLE;
    std::string    shotDir;             // absolute; supplied by the caller

    // -- FPS counter ----------------------------------------------------------
    double lastFpsTime = 0.0;
    int    fpsFrames   = 0;
    float  fpsDisplay  = 0.f;

    // -- Methods --------------------------------------------------------------
    void run(const SceneData& s, int w, int h, AaMode initialAa,
             VkSampleCountFlagBits initialSamples, const float bg[3],
             bool flipUp, const std::string& shotDirIn);

    void initGlfw(int w, int h);
    void initInstance();
    void initSurface();
    void pickPhysDevice();
    void initDevice();
    void resolveMsaaSamples();
    void initSwapchain(VkSwapchainKHR old = VK_NULL_HANDLE);
    void updateSceneExtent();
    void initDepth();
    void initMsaaColor();
    void initSceneColor();          // gauss_msaa only
    void initSceneRenderPass();
    void initImguiRenderPass();
    void initSceneFbs();
    void initImguiFbs();
    void initGaussResolve();        // gauss_msaa only: RP + sampler + desc + pipeline + FBs
    void updateGaussDescriptor();
    void initPipelineLayout();
    void initPipelines();
    void initCmds();
    void initSync();
    void initImgui();

    void destroySceneFbs();
    void destroyImguiFbs();
    void destroyGaussFbs();
    void destroyDepth();
    void destroyMsaaColor();
    void destroySceneColor();
    void destroySCDependents();
    void destroyPipelines();
    void destroyGaussResolve();     // includes FBs + sampler + desc + pipeline + RP

    void mkBuf(VkDeviceSize sz, VkBufferUsageFlags usage,
               VkMemoryPropertyFlags memFlags,
               VkBuffer& buf, VkDeviceMemory& mem);
    void stageUpload(const void* data, VkDeviceSize sz,
                     VkBufferUsageFlags dstUsage,
                     VkBuffer& buf, VkDeviceMemory& mem);
    void uploadGeom(const SceneData& s);
    /// Rewrite the radius column of vBuf.  radius < 0 restores the
    /// checkpoint's per-vertex radii.
    void applyLineWidth(float radius);
    void initQuadLineDescriptors();

    VkPipeline buildVboPipeline(VkShaderModule vs, VkShaderModule fs,
                                VkPrimitiveTopology topo,
                                VkCullModeFlags cullMode = VK_CULL_MODE_NONE,
                                bool primitiveRestart = false);
    VkPipeline buildQuadLinePipeline(VkShaderModule vs, VkShaderModule fs);

    void initShPipeline();
    void initShDescriptors();
    void dispatchShEval(VkCommandBuffer cb, const float campos[3]);

    void buildImguiFrame();
    void drawFrame();
    void recordScreenshotCopy(VkCommandBuffer cb, uint32_t imgIdx);
    void finishScreenshot();
    void rebuildSC();
    void rebuildForModeChange();    // on aaMode or samples change

    void cleanup();

    static void cbResize   (GLFWwindow*, int, int);
    static void cbMouseBtn (GLFWwindow*, int, int, int);
    static void cbCursorPos(GLFWwindow*, double, double);
    static void cbScroll   (GLFWwindow*, double, double);
    static void cbKey      (GLFWwindow*, int, int, int, int);
};

// ---------------------------------------------------------------------------
// GLFW callbacks
// ---------------------------------------------------------------------------
void ViewerImpl::cbResize(GLFWwindow* w, int, int) {
    static_cast<ViewerImpl*>(glfwGetWindowUserPointer(w))->dirty = true;
}

void ViewerImpl::cbMouseBtn(GLFWwindow* w, int btn, int act, int) {
    auto* v = static_cast<ViewerImpl*>(glfwGetWindowUserPointer(w));
    if (act == GLFW_PRESS) glfwGetCursorPos(w, &v->prevX, &v->prevY);
    if (ImGui::GetIO().WantCaptureMouse) return;
    bool press = (act == GLFW_PRESS);
    if (btn == GLFW_MOUSE_BUTTON_LEFT)   v->btnL = press;
    if (btn == GLFW_MOUSE_BUTTON_RIGHT)  v->btnR = press;
    if (btn == GLFW_MOUSE_BUTTON_MIDDLE) v->btnM = press;
}

void ViewerImpl::cbCursorPos(GLFWwindow* w, double x, double y) {
    auto* v  = static_cast<ViewerImpl*>(glfwGetWindowUserPointer(w));
    double dx = x - v->prevX, dy = y - v->prevY;
    v->prevX = x;  v->prevY = y;
    if (ImGui::GetIO().WantCaptureMouse) return;
    if (v->btnL)            v->cam.orbit(float(-dy * 0.005), float(-dx * 0.005));
    if (v->btnR || v->btnM) v->cam.pan  (float(dx),          float(dy));
}

void ViewerImpl::cbScroll(GLFWwindow* w, double, double dy) {
    if (ImGui::GetIO().WantCaptureMouse) return;
    static_cast<ViewerImpl*>(glfwGetWindowUserPointer(w))
        ->cam.zoom(dy < 0 ? 1.12f : 0.88f);
}

void ViewerImpl::cbKey(GLFWwindow* w, int key, int, int act, int) {
    if (ImGui::GetIO().WantCaptureKeyboard) return;
    if (key == GLFW_KEY_ESCAPE && act == GLFW_PRESS)
        glfwSetWindowShouldClose(w, GLFW_TRUE);
}

// ---------------------------------------------------------------------------
// GLFW / instance / surface / device
// ---------------------------------------------------------------------------
void ViewerImpl::initGlfw(int w, int h) {
    if (!glfwInit()) throw std::runtime_error("glfwInit failed");
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    win = glfwCreateWindow(w, h, "fuzzydr_viewer", nullptr, nullptr);
    if (!win) throw std::runtime_error("glfwCreateWindow failed");
    glfwSetWindowUserPointer      (win, this);
    glfwSetFramebufferSizeCallback(win, cbResize);
    glfwSetMouseButtonCallback    (win, cbMouseBtn);
    glfwSetCursorPosCallback      (win, cbCursorPos);
    glfwSetScrollCallback         (win, cbScroll);
    glfwSetKeyCallback            (win, cbKey);
}

void ViewerImpl::initInstance() {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "fuzzydr_viewer";
    app.apiVersion       = VK_API_VERSION_1_2;

    uint32_t     n = 0;
    const char** e = glfwGetRequiredInstanceExtensions(&n);
    std::vector<const char*> exts(e, e + n);

    VkInstanceCreateFlags instFlags = 0;
#ifdef __APPLE__
    exts.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    instFlags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.flags                   = instFlags;
    ci.pApplicationInfo        = &app;
    ci.enabledExtensionCount   = uint32_t(exts.size());
    ci.ppEnabledExtensionNames = exts.data();
    VK_CHECK(vkCreateInstance(&ci, nullptr, &inst));
}

void ViewerImpl::initSurface() {
    VK_CHECK(glfwCreateWindowSurface(inst, win, nullptr, &surf));
}

void ViewerImpl::pickPhysDevice() {
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
            VkBool32 ok = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(d, i, surf, &ok);
            if (!ok) continue;
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

void ViewerImpl::initDevice() {
    float prio = 1.f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = qFam;
    qci.queueCount       = 1;
    qci.pQueuePriorities = &prio;

    std::vector<const char*> devExts = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
#ifdef __APPLE__
    devExts.push_back("VK_KHR_portability_subset");
#endif

    VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    ci.queueCreateInfoCount    = 1;
    ci.pQueueCreateInfos       = &qci;
    ci.enabledExtensionCount   = uint32_t(devExts.size());
    ci.ppEnabledExtensionNames = devExts.data();
    VK_CHECK(vkCreateDevice(pDev, &ci, nullptr, &dev));
    vkGetDeviceQueue(dev, qFam, 0, &queue);
}

// ---------------------------------------------------------------------------
// resolveMsaaSamples - clamp the requested sample count to the device's
// support for the swapchain colour format and depth.  Only meaningful under
// hw_msaa; gauss_msaa and none force 1-sample regardless.
// ---------------------------------------------------------------------------
void ViewerImpl::resolveMsaaSamples() {
    if (aaMode == AA_GAUSS_MSAA || aaMode == AA_NONE) {
        msaaSamples = VK_SAMPLE_COUNT_1_BIT;
        return;
    }
    if (msaaSamples == VK_SAMPLE_COUNT_1_BIT) return;

    auto queryCounts = [&](VkFormat fmt, VkImageUsageFlags usage) {
        VkImageFormatProperties fp{};
        if (vkGetPhysicalDeviceImageFormatProperties(
                pDev, fmt, VK_IMAGE_TYPE_2D,
                VK_IMAGE_TILING_OPTIMAL, usage, 0, &fp) != VK_SUCCESS)
            return static_cast<VkSampleCountFlags>(VK_SAMPLE_COUNT_1_BIT);
        return fp.sampleCounts;
    };
    VkSampleCountFlags counts =
        queryCounts(scFmt, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) &
        queryCounts(VK_FORMAT_D32_SFLOAT,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);

    for (auto c : {VK_SAMPLE_COUNT_4_BIT,
                   VK_SAMPLE_COUNT_2_BIT,
                   VK_SAMPLE_COUNT_1_BIT}) {
        if (c <= msaaSamples && (counts & c)) { msaaSamples = c; return; }
    }
    msaaSamples = VK_SAMPLE_COUNT_1_BIT;
}

// ---------------------------------------------------------------------------
// initSwapchain - UNORM preferred; pick present mode from the vsync toggle.
// ---------------------------------------------------------------------------
void ViewerImpl::initSwapchain(VkSwapchainKHR old) {
    uint32_t fmtN = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(pDev, surf, &fmtN, nullptr);
    std::vector<VkSurfaceFormatKHR> fmts(fmtN);
    vkGetPhysicalDeviceSurfaceFormatsKHR(pDev, surf, &fmtN, fmts.data());

    // UNORM preferred: shaders output linear colour; SRGB would oversaturate.
    struct Candidate { VkFormat fmt; VkColorSpaceKHR cs; };
    const std::vector<Candidate> wanted = {
        { VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR },
        { VK_FORMAT_R8G8B8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR },
    };
    scFmt = fmts[0].format;
    for (auto& w : wanted)
        for (auto& f : fmts)
            if (f.format == w.fmt && f.colorSpace == w.cs) {
                scFmt = f.format;  goto fmt_done;
            }
    fmt_done:;

    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pDev, surf, &caps);
    scExt = caps.currentExtent;
    if (scExt.width == UINT32_MAX) {
        int w, h;
        glfwGetFramebufferSize(win, &w, &h);
        scExt.width  = std::clamp(uint32_t(w),
                                  caps.minImageExtent.width,
                                  caps.maxImageExtent.width);
        scExt.height = std::clamp(uint32_t(h),
                                  caps.minImageExtent.height,
                                  caps.maxImageExtent.height);
    }

    uint32_t imgCnt = caps.minImageCount + 1;
    if (caps.maxImageCount > 0) imgCnt = std::min(imgCnt, caps.maxImageCount);

    uint32_t pmN = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(pDev, surf, &pmN, nullptr);
    std::vector<VkPresentModeKHR> pms(pmN);
    vkGetPhysicalDeviceSurfacePresentModesKHR(pDev, surf, &pmN, pms.data());
    VkPresentModeKHR pm = VK_PRESENT_MODE_FIFO_KHR;
#ifndef __APPLE__
    mailboxAvail = false;
    for (auto m : pms)
        if (m == VK_PRESENT_MODE_MAILBOX_KHR) { mailboxAvail = true; break; }
    if (!vsync && mailboxAvail)
        pm = VK_PRESENT_MODE_MAILBOX_KHR;
#endif

    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    ci.surface          = surf;
    ci.minImageCount    = imgCnt;
    ci.imageFormat      = scFmt;
    ci.imageColorSpace  = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    ci.imageExtent      = scExt;
    ci.imageArrayLayers = 1;
    ci.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    // Screenshots copy out of the swapchain image; the bit is requested only
    // when the surface supports it, and screenshotSupported gates the UI.
    screenshotSupported = (caps.supportedUsageFlags &
                           VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0;
    if (screenshotSupported)
        ci.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform     = caps.currentTransform;
    ci.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode      = pm;
    ci.clipped          = VK_TRUE;
    ci.oldSwapchain     = old;
    VK_CHECK(vkCreateSwapchainKHR(dev, &ci, nullptr, &sc));

    uint32_t n = 0;
    vkGetSwapchainImagesKHR(dev, sc, &n, nullptr);
    scImg.resize(n);
    vkGetSwapchainImagesKHR(dev, sc, &n, scImg.data());

    scView.resize(n);
    for (uint32_t i = 0; i < n; i++) {
        VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vci.image            = scImg[i];
        vci.viewType         = VK_IMAGE_VIEW_TYPE_2D;
        vci.format           = scFmt;
        vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VK_CHECK(vkCreateImageView(dev, &vci, nullptr, &scView[i]));
    }

    updateSceneExtent();
}

// Scene pass renders at 2x swapchain size under gauss_msaa, 1x under hw_msaa.
void ViewerImpl::updateSceneExtent() {
    if (aaMode == AA_GAUSS_MSAA)
        sceneExt = {scExt.width * 2, scExt.height * 2};
    else
        sceneExt = scExt;
}

// ---------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------
static void mkImage(VkDevice dev, VkPhysicalDevice pDev,
                    uint32_t w, uint32_t h,
                    VkFormat fmt, VkSampleCountFlagBits samples,
                    VkImageUsageFlags usage, VkImageAspectFlags aspect,
                    VkImage& outImg, VkDeviceMemory& outMem, VkImageView& outView) {
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType   = VK_IMAGE_TYPE_2D;
    ici.format      = fmt;
    ici.extent      = {w, h, 1};
    ici.mipLevels   = 1;
    ici.arrayLayers = 1;
    ici.samples     = samples;
    ici.tiling      = VK_IMAGE_TILING_OPTIMAL;
    ici.usage       = usage;
    VK_CHECK(vkCreateImage(dev, &ici, nullptr, &outImg));

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

void ViewerImpl::initDepth() {
    mkImage(dev, pDev, sceneExt.width, sceneExt.height,
            VK_FORMAT_D32_SFLOAT, msaaSamples,
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            VK_IMAGE_ASPECT_DEPTH_BIT,
            depImg, depMem, depView);
}

void ViewerImpl::initMsaaColor() {
    if (msaaSamples == VK_SAMPLE_COUNT_1_BIT) return;
    mkImage(dev, pDev, sceneExt.width, sceneExt.height,
            scFmt, msaaSamples,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
            VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            msaaImg, msaaMem, msaaView);
}

// Only allocated under gauss_msaa: 1-sample fp16 RGBA at 2x res, sampled
// by the Gaussian-resolve pass.  fp16 gives the resolve enough precision
// in [0, 1] without the bandwidth cost of fp32; u8 here would quantise
// the supersampled buffer below the resolve filter.
void ViewerImpl::initSceneColor() {
    if (aaMode != AA_GAUSS_MSAA) return;
    mkImage(dev, pDev, sceneExt.width, sceneExt.height,
            VK_FORMAT_R16G16B16A16_SFLOAT, VK_SAMPLE_COUNT_1_BIT,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
            VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            sceneColorImg, sceneColorMem, sceneColorView);
}

// ---------------------------------------------------------------------------
// Scene render pass.
//   hw_msaa 1-sample : [scView, depth].  Direct write.
//   hw_msaa Nx-MSAA  : [msaaColor@Nx, depth@Nx, scView].  HW resolve.
//   gauss_msaa       : [sceneColor@2x, depth@2x].  finalLayout = SHADER_READ.
// ---------------------------------------------------------------------------
void ViewerImpl::initSceneRenderPass() {
    const bool msaa = (msaaSamples != VK_SAMPLE_COUNT_1_BIT);

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
                            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                            VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;

    // hw_msaa hands off to ImGui (COLOR_ATTACHMENT_OPTIMAL) via its final
    // layout; gauss_msaa hands off to the resolve pass (SHADER_READ).
    const VkImageLayout colorFinal = (aaMode == AA_GAUSS_MSAA)
        ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription depthAtt{};
    depthAtt.format         = VK_FORMAT_D32_SFLOAT;
    depthAtt.samples        = msaaSamples;
    depthAtt.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAtt.storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAtt.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAtt.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAtt.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAtt.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    if (msaa) {
        std::array<VkAttachmentDescription, 3> att{};
        att[0].format         = scFmt;
        att[0].samples        = msaaSamples;
        att[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        att[0].storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        att[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        att[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        att[0].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        att[0].finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        att[1]                = depthAtt;
        att[2].format         = scFmt;
        att[2].samples        = VK_SAMPLE_COUNT_1_BIT;
        att[2].loadOp         = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        att[2].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        att[2].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        att[2].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        att[2].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        att[2].finalLayout    = colorFinal;

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

    // 1-sample path.  Under gauss_msaa the colour target is sceneColorImg
    // (fp16); under hw_msaa-1sample fallback it's the swapchain image.
    std::array<VkAttachmentDescription, 2> att{};
    att[0].format         = (aaMode == AA_GAUSS_MSAA)
                            ? VK_FORMAT_R16G16B16A16_SFLOAT
                            : scFmt;
    att[0].samples        = VK_SAMPLE_COUNT_1_BIT;
    att[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    att[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att[0].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    att[0].finalLayout    = colorFinal;
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

// ---------------------------------------------------------------------------
// ImGui render pass - always 1-sample, LOAD -> PRESENT_SRC_KHR.
// ---------------------------------------------------------------------------
void ViewerImpl::initImguiRenderPass() {
    VkAttachmentDescription att{};
    att.format         = scFmt;
    att.samples        = VK_SAMPLE_COUNT_1_BIT;
    att.loadOp         = VK_ATTACHMENT_LOAD_OP_LOAD;
    att.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    att.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att.initialLayout  = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    att.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

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
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo ci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    ci.attachmentCount = 1;
    ci.pAttachments    = &att;
    ci.subpassCount    = 1;
    ci.pSubpasses      = &sub;
    ci.dependencyCount = 1;
    ci.pDependencies   = &dep;
    VK_CHECK(vkCreateRenderPass(dev, &ci, nullptr, &imguiRp));
}

// ---------------------------------------------------------------------------
// Scene / ImGui / Gauss framebuffers - one per swapchain image.
// ---------------------------------------------------------------------------
void ViewerImpl::initSceneFbs() {
    sceneFbs.resize(scView.size());
    const bool msaa = (msaaSamples != VK_SAMPLE_COUNT_1_BIT);
    for (size_t i = 0; i < scView.size(); i++) {
        VkFramebufferCreateInfo ci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        ci.renderPass = sceneRp;
        ci.width      = sceneExt.width;
        ci.height     = sceneExt.height;
        ci.layers     = 1;

        // In gauss_msaa the scene pass target is sceneColor (shared across
        // swapchain images); the per-sc-image framebuffer still exists for
        // symmetry but all point at the same colour attachment.
        VkImageView colorView = (aaMode == AA_GAUSS_MSAA) ? sceneColorView : scView[i];
        VkImageView atts3[3] = {msaaView, depView, colorView};
        VkImageView atts2[2] = {colorView, depView};
        if (msaa) {
            ci.attachmentCount = 3;
            ci.pAttachments    = atts3;
        } else {
            ci.attachmentCount = 2;
            ci.pAttachments    = atts2;
        }
        VK_CHECK(vkCreateFramebuffer(dev, &ci, nullptr, &sceneFbs[i]));
    }
}

void ViewerImpl::initImguiFbs() {
    imguiFbs.resize(scView.size());
    for (size_t i = 0; i < scView.size(); i++) {
        VkFramebufferCreateInfo ci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        ci.renderPass      = imguiRp;
        ci.attachmentCount = 1;
        ci.pAttachments    = &scView[i];
        ci.width           = scExt.width;
        ci.height          = scExt.height;
        ci.layers          = 1;
        VK_CHECK(vkCreateFramebuffer(dev, &ci, nullptr, &imguiFbs[i]));
    }
}

// ---------------------------------------------------------------------------
// Gaussian resolve pass - built only under gauss_msaa.  Render pass,
// sampler, descriptor layout and pipeline are created once; framebuffers
// and the descriptor set point at swapchain-sized resources and are
// recreated on resize / mode change.
// ---------------------------------------------------------------------------
void ViewerImpl::initGaussResolve() {
    if (aaMode != AA_GAUSS_MSAA) return;

    // Render pass.
    {
        VkAttachmentDescription att{};
        att.format         = scFmt;
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

    // Descriptor set + pipeline layout.
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
    }

    // Pipeline.
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

    // Framebuffers (one per swapchain image) - write into scView[i].
    gaussFbs.resize(scView.size());
    for (size_t i = 0; i < scView.size(); i++) {
        VkFramebufferCreateInfo ci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        ci.renderPass      = gaussRp;
        ci.attachmentCount = 1;
        ci.pAttachments    = &scView[i];
        ci.width           = scExt.width;
        ci.height          = scExt.height;
        ci.layers          = 1;
        VK_CHECK(vkCreateFramebuffer(dev, &ci, nullptr, &gaussFbs[i]));
    }

    updateGaussDescriptor();
}

void ViewerImpl::updateGaussDescriptor() {
    if (aaMode != AA_GAUSS_MSAA || !gaussDescSet) return;
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

// ---------------------------------------------------------------------------
// Pipeline layout (face + bresen + quad-line share it)
// ---------------------------------------------------------------------------
void ViewerImpl::initPipelineLayout() {
    std::array<VkDescriptorSetLayoutBinding, 2> binds{};
    binds[0].binding         = 0;
    binds[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binds[0].descriptorCount = 1;
    binds[0].stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;
    binds[1].binding         = 1;
    binds[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binds[1].descriptorCount = 1;
    binds[1].stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo dlci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dlci.bindingCount = uint32_t(binds.size());
    dlci.pBindings    = binds.data();
    VK_CHECK(vkCreateDescriptorSetLayout(dev, &dlci, nullptr, &descLayout));

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pcr.offset     = 0;
    pcr.size       = sizeof(PushConst);

    VkPipelineLayoutCreateInfo lci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    lci.setLayoutCount         = 1;
    lci.pSetLayouts            = &descLayout;
    lci.pushConstantRangeCount = 1;
    lci.pPushConstantRanges    = &pcr;
    VK_CHECK(vkCreatePipelineLayout(dev, &lci, nullptr, &pipeLayout));
}

// ---------------------------------------------------------------------------
// Graphics pipeline builders
// ---------------------------------------------------------------------------
VkPipeline ViewerImpl::buildVboPipeline(VkShaderModule vs, VkShaderModule fs,
                                        VkPrimitiveTopology topo,
                                        VkCullModeFlags cullMode,
                                        bool primitiveRestart) {
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs;  stages[0].pName = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fs;  stages[1].pName = "main";

    // Vertex layout: stride 28, {pos @0, rgb @16}.  The radius slot at
    // offset 12 is ignored by Bresenham lines and faces.
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
    ia.topology = topo;
    ia.primitiveRestartEnable = primitiveRestart ? VK_TRUE : VK_FALSE;

    VkPipelineViewportStateCreateInfo vp{
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1;
    vp.scissorCount  = 1;

    VkPipelineRasterizationStateCreateInfo rs{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode    = cullMode;
    rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth   = 1.f;

    VkPipelineMultisampleStateCreateInfo ms{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = msaaSamples;

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
    ci.layout              = pipeLayout;
    ci.renderPass          = sceneRp;

    VkPipeline pipe;
    VK_CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr, &pipe));
    return pipe;
}

VkPipeline ViewerImpl::buildQuadLinePipeline(VkShaderModule vs, VkShaderModule fs) {
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
    ms.rasterizationSamples = msaaSamples;
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
    ci.layout              = pipeLayout;
    ci.renderPass          = sceneRp;
    VkPipeline pipe;
    VK_CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr, &pipe));
    return pipe;
}

void ViewerImpl::initPipelines() {
    VkShaderModule vs = makeShader(dev, readSpv(shaderPath("viewer.vert.spv")));
    VkShaderModule fs = makeShader(dev, readSpv(shaderPath("viewer.frag.spv")));
    facePipeline     = buildVboPipeline(vs, fs, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                                        VK_CULL_MODE_NONE);
    facePipelineCull = buildVboPipeline(vs, fs, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                                        VK_CULL_MODE_BACK_BIT);
    // LINE_STRIP + primitive restart: UINT32_MAX in the index buffer (the
    // standard sentinel for VK_INDEX_TYPE_UINT32) breaks the strip between
    // maximal walks.  Without a strip buffer the pair list is drawn directly.
    bresenPipeline   = nStripIndices > 0
        ? buildVboPipeline(vs, fs, VK_PRIMITIVE_TOPOLOGY_LINE_STRIP,
                           VK_CULL_MODE_NONE, true)
        : buildVboPipeline(vs, fs, VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
                           VK_CULL_MODE_NONE);
    vkDestroyShaderModule(dev, vs, nullptr);
    vkDestroyShaderModule(dev, fs, nullptr);

    VkShaderModule qvs = makeShader(dev, readSpv(shaderPath("viewer_line_quads.vert.spv")));
    VkShaderModule qfs = makeShader(dev, readSpv(shaderPath("viewer.frag.spv")));
    quadLinePipeline = buildQuadLinePipeline(qvs, qfs);
    vkDestroyShaderModule(dev, qvs, nullptr);
    vkDestroyShaderModule(dev, qfs, nullptr);
}

// ---------------------------------------------------------------------------
// Cmds / sync / ImGui
// ---------------------------------------------------------------------------
void ViewerImpl::initCmds() {
    VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    ci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    ci.queueFamilyIndex = qFam;
    VK_CHECK(vkCreateCommandPool(dev, &ci, nullptr, &cmdPool));

    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool        = cmdPool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = MF;
    VK_CHECK(vkAllocateCommandBuffers(dev, &ai, cmdBuf));
}

void ViewerImpl::initSync() {
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo     fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (int i = 0; i < MF; i++) {
        VK_CHECK(vkCreateSemaphore(dev, &sci, nullptr, &imgSem[i]));
        VK_CHECK(vkCreateSemaphore(dev, &sci, nullptr, &renSem[i]));
        VK_CHECK(vkCreateFence    (dev, &fci, nullptr, &fence[i]));
    }
}

void ViewerImpl::initImgui() {
    VkDescriptorPoolSize ps[] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pci.maxSets       = 1;
    pci.poolSizeCount = 1;
    pci.pPoolSizes    = ps;
    VK_CHECK(vkCreateDescriptorPool(dev, &pci, nullptr, &imguiPool));

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForVulkan(win, true);
    ImGui_ImplVulkan_InitInfo info{};
    info.Instance        = inst;
    info.PhysicalDevice  = pDev;
    info.Device          = dev;
    info.QueueFamily     = qFam;
    info.Queue           = queue;
    info.DescriptorPool  = imguiPool;
    info.MinImageCount   = uint32_t(scImg.size());
    info.ImageCount      = uint32_t(scImg.size());
    info.MSAASamples     = VK_SAMPLE_COUNT_1_BIT;
    info.RenderPass      = imguiRp;
    ImGui_ImplVulkan_Init(&info);
}

// ---------------------------------------------------------------------------
// Buffers
// ---------------------------------------------------------------------------
void ViewerImpl::mkBuf(VkDeviceSize sz, VkBufferUsageFlags usage,
                       VkMemoryPropertyFlags memFlags,
                       VkBuffer& buf, VkDeviceMemory& mem) {
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size        = sz;
    ci.usage       = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(dev, &ci, nullptr, &buf));

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev, buf, &mr);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize  = mr.size;
    ai.memoryTypeIndex = findMemType(pDev, mr.memoryTypeBits, memFlags);
    VK_CHECK(vkAllocateMemory(dev, &ai, nullptr, &mem));
    vkBindBufferMemory(dev, buf, mem, 0);
}

void ViewerImpl::stageUpload(const void* data, VkDeviceSize sz,
                             VkBufferUsageFlags dstUsage,
                             VkBuffer& buf, VkDeviceMemory& mem) {
    VkBuffer       stageBuf = VK_NULL_HANDLE;
    VkDeviceMemory stageMem = VK_NULL_HANDLE;
    mkBuf(sz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          stageBuf, stageMem);
    void* ptr;
    vkMapMemory(dev, stageMem, 0, sz, 0, &ptr);
    memcpy(ptr, data, sz);
    vkUnmapMemory(dev, stageMem);

    mkBuf(sz, dstUsage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
          buf, mem);

    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool        = cmdPool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb;
    VK_CHECK(vkAllocateCommandBuffers(dev, &ai, &cb));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cb, &bi));
    VkBufferCopy region{0, 0, sz};
    vkCmdCopyBuffer(cb, stageBuf, buf, 1, &region);
    VK_CHECK(vkEndCommandBuffer(cb));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cb;
    VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue));
    vkFreeCommandBuffers(dev, cmdPool, 1, &cb);
    vkDestroyBuffer(dev, stageBuf, nullptr);
    vkFreeMemory   (dev, stageMem, nullptr);
}

// ---------------------------------------------------------------------------
// Quad-line descriptor set
// ---------------------------------------------------------------------------
void ViewerImpl::initQuadLineDescriptors() {
    VkDescriptorPoolSize poolSizes[1] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2}
    };
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pci.maxSets       = 1;
    pci.poolSizeCount = 1;
    pci.pPoolSizes    = poolSizes;
    VK_CHECK(vkCreateDescriptorPool(dev, &pci, nullptr, &descPool));

    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool     = descPool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts        = &descLayout;
    VK_CHECK(vkAllocateDescriptorSets(dev, &ai, &descSet));

    VkBuffer vattrBuf = hasSh ? evalBuf : vBuf;
    VkDescriptorBufferInfo vattrInfo{};
    vattrInfo.buffer = vattrBuf;  vattrInfo.range = VK_WHOLE_SIZE;
    VkDescriptorBufferInfo linesInfo{};
    linesInfo.buffer = lBuf;      linesInfo.range = VK_WHOLE_SIZE;

    std::array<VkWriteDescriptorSet, 2> writes{};
    writes[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet          = descSet;
    writes[0].dstBinding      = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo     = &vattrInfo;
    writes[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet          = descSet;
    writes[1].dstBinding      = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo     = &linesInfo;
    vkUpdateDescriptorSets(dev, uint32_t(writes.size()), writes.data(), 0, nullptr);
}

// ---------------------------------------------------------------------------
// uploadGeom
// ---------------------------------------------------------------------------
void ViewerImpl::uploadGeom(const SceneData& s) {
    scene = &s;
    uploadedRadius = -1.f;   // vBuf starts with the checkpoint's own radii
    hasVertexRadius = s.has_vertex_radius;
    // Checkpoints carrying real radii open in the style they were trained for.
    lineMode = hasVertexRadius ? 2 : 0;
    nVerts = s.num_verts;
    nFaces = s.num_faces;
    nLines = s.num_lines;
    shCols = s.sh_cols;
    hasSh  = (shCols > 0 && s.sh_coeffs.size() == size_t(nVerts) * shCols);

    if (nVerts > 0) {
        VkDeviceSize sz = nVerts * 7 * sizeof(float);
        stageUpload(s.vert_attrs.data(), sz,
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    vBuf, vMem);

        if (hasSh) {
            VkDeviceSize shSz = nVerts * shCols * sizeof(float);
            std::vector<float> transposed(size_t(nVerts) * shCols);
            {
                const float* src = s.sh_coeffs.data();
                float*       dst = transposed.data();
                for (uint32_t vi = 0; vi < nVerts; vi++)
                    for (uint32_t c = 0; c < shCols; c++)
                        dst[c * nVerts + vi] = src[vi * shCols + c];
            }
            stageUpload(transposed.data(), shSz,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        shBuf, shMem);
            mkBuf(sz,
                  VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                  evalBuf, evalMem);
        }
    }
    if (nFaces > 0) {
        VkDeviceSize sz = nFaces * 3 * sizeof(uint32_t);
        stageUpload(s.faces.data(), sz,
                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                    fBuf, fMem);
    }
    if (nLines > 0) {
        VkDeviceSize sz = nLines * 2 * sizeof(uint32_t);
        stageUpload(s.lines.data(), sz,
                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    lBuf, lMem);
        initQuadLineDescriptors();
    }
    // Only the Bresenham style reads this, and only as an index buffer; the
    // quad styles keep pulling pairs out of lBuf.
    if (nStripIndices > 0 && s.line_strip.size() == size_t(nStripIndices)) {
        stageUpload(s.line_strip.data(), nStripIndices * sizeof(uint32_t),
                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                    lStripBuf, lStripMem);
    } else {
        nStripIndices = 0;
    }
    if (hasSh) {
        initShPipeline();
        initShDescriptors();
    }
}

void ViewerImpl::applyLineWidth(float radius) {
    if (!scene || nVerts == 0 || !vBuf) return;
    if (radius == uploadedRadius) return;

    // vBuf may be referenced by frames still in flight.
    vkDeviceWaitIdle(dev);

    std::vector<float> va(scene->vert_attrs);
    if (radius >= 0.f)
        for (uint32_t i = 0; i < nVerts; i++)
            va[size_t(i) * 7 + 3] = radius;

    // Copy into the existing vBuf; recreating it would dangle the quad-line
    // descriptor set, which binds vBuf when there are no SH coefficients.
    const VkDeviceSize sz = VkDeviceSize(nVerts) * 7 * sizeof(float);
    VkBuffer       stageBuf = VK_NULL_HANDLE;
    VkDeviceMemory stageMem = VK_NULL_HANDLE;
    mkBuf(sz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          stageBuf, stageMem);
    void* ptr = nullptr;
    vkMapMemory(dev, stageMem, 0, sz, 0, &ptr);
    memcpy(ptr, va.data(), sz);
    vkUnmapMemory(dev, stageMem);

    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool        = cmdPool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb;
    VK_CHECK(vkAllocateCommandBuffers(dev, &ai, &cb));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cb, &bi));
    VkBufferCopy region{0, 0, sz};
    vkCmdCopyBuffer(cb, stageBuf, vBuf, 1, &region);
    VK_CHECK(vkEndCommandBuffer(cb));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cb;
    VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue));
    vkFreeCommandBuffers(dev, cmdPool, 1, &cb);
    vkDestroyBuffer(dev, stageBuf, nullptr);
    vkFreeMemory   (dev, stageMem, nullptr);

    uploadedRadius = radius;
    // Under SH the draw reads evalBuf, which the compute pass fills from vBuf
    // and which is skipped while the camera is still; force one re-dispatch.
    prevCampos[0] = prevCampos[1] = prevCampos[2] = 1e30f;
}

// ---------------------------------------------------------------------------
// SH / shading compute pipeline
// ---------------------------------------------------------------------------
void ViewerImpl::initShPipeline() {
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

    VkShaderModule cs = makeShader(dev, readSpv(shaderPath("viewer_sh_eval.comp.spv")));
    VkPipelineShaderStageCreateInfo stage{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = cs;
    stage.pName  = "main";
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage  = stage;
    ci.layout = compPipeLayout;
    VK_CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr,
                                      &compPipeline));
    vkDestroyShaderModule(dev, cs, nullptr);
}

void ViewerImpl::initShDescriptors() {
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

void ViewerImpl::dispatchShEval(VkCommandBuffer cb, const float campos[3]) {
    // The evaluated colours depend only on the camera position, so skip the
    // dispatch entirely while the camera is still.
    if (campos[0] == prevCampos[0] &&
        campos[1] == prevCampos[1] &&
        campos[2] == prevCampos[2])
        return;
    prevCampos[0] = campos[0];
    prevCampos[1] = campos[1];
    prevCampos[2] = campos[2];

    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, compPipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                            compPipeLayout, 0, 1, &compDescSet, 0, nullptr);

    CompPushConst pc{};
    pc.campos[0] = campos[0];  pc.campos[1] = campos[1];  pc.campos[2] = campos[2];
    pc.nVerts    = nVerts;
    vkCmdPushConstants(cb, compPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);

    uint32_t groups = (nVerts + 255u) / 256u;
    vkCmdDispatch(cb, groups, 1, 1);

    VkBufferMemoryBarrier bar{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    bar.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
    bar.dstAccessMask       = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT |
                              VK_ACCESS_SHADER_READ_BIT;
    bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.buffer              = evalBuf;
    bar.offset              = 0;
    bar.size                = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cb,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_VERTEX_INPUT_BIT |
                         VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                         0, 0, nullptr, 1, &bar, 0, nullptr);
}

// ---------------------------------------------------------------------------
// ImGui frame
// ---------------------------------------------------------------------------
void ViewerImpl::buildImguiFrame() {
    double now = glfwGetTime();
    ++fpsFrames;
    if (now - lastFpsTime >= 0.5) {
        fpsDisplay  = float(fpsFrames / (now - lastFpsTime));
        fpsFrames   = 0;
        lastFpsTime = now;
    }

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::SetNextWindowPos ({10.f, 10.f}, ImGuiCond_Once);
    // Height is set explicitly rather than auto-fitted: the panel gains rows
    // for the sample count, the line-width slider and the per-vertex line
    // style, and the initial size has to leave room for all of them.
    ImGui::SetNextWindowSize({270.f, 450.f}, ImGuiCond_Once);
    ImGui::Begin("fuzzydr_viewer");

    ImGui::Text("FPS  : %.1f", fpsDisplay);
    ImGui::Text("Size : %u x %u px", scExt.width, scExt.height);
    ImGui::Separator();

#ifndef __APPLE__
    if (!mailboxAvail) ImGui::BeginDisabled();
    ImGui::Checkbox("VSync (FIFO)", &pendingVsync);
    if (!mailboxAvail) {
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::TextDisabled("(no MAILBOX)");
    }
    if (pendingVsync != vsync)
        ImGui::TextColored({1.f, 1.f, 0.f, 1.f}, "Applying...");
    ImGui::Separator();
#endif

    ImGui::ColorEdit3("Background", bgColor);
    ImGui::Separator();

    // Datasets differ in which way their Z axis points, so the scene can load
    // upside down.  This flips the camera's up reference, not the geometry.
    bool flipUp = (cam.up_sign < 0);
    if (ImGui::Checkbox("Flip up", &flipUp))
        cam.up_sign = flipUp ? -1 : +1;
    ImGui::Separator();

    if (nFaces > 0) {
        ImGui::Checkbox("Backface culling", &backfaceCull);
        ImGui::Separator();
    }

    // AA mode.  Gaussian renders on a 2x grid in each axis (4 samples per
    // output pixel) and resolves with a 6x6 sigma=0.5 kernel, so its sample
    // count is fixed and not exposed.  Only hardware MSAA is adjustable.
    const char* aaLabels[] = {"No AA", "Gaussian MSAA (4x)", "Hardware MSAA"};
    int aaIdx = (pendingAaMode == AA_NONE)       ? 0
              : (pendingAaMode == AA_GAUSS_MSAA) ? 1 : 2;
    if (ImGui::Combo("AA mode", &aaIdx, aaLabels, 3)) {
        pendingAaMode = (aaIdx == 0) ? AA_NONE
                      : (aaIdx == 1) ? AA_GAUSS_MSAA : AA_HW_MSAA;
        // "No AA" is the hardware path at one sample; entering hardware MSAA
        // from it must not leave the count at 1 (that is the No-AA entry).
        if (pendingAaMode == AA_NONE)
            pendingMsaaSamples = VK_SAMPLE_COUNT_1_BIT;
        else if (pendingAaMode == AA_HW_MSAA &&
                 pendingMsaaSamples == VK_SAMPLE_COUNT_1_BIT)
            pendingMsaaSamples = VK_SAMPLE_COUNT_2_BIT;
    }

    // Sample count - hardware MSAA only; 1x lives under the "No AA" entry.
    if (pendingAaMode == AA_HW_MSAA) {
        const char* msaaLabels[] = {"2x", "4x"};
        int msaaIdx = (pendingMsaaSamples == VK_SAMPLE_COUNT_4_BIT) ? 1 : 0;
        if (ImGui::Combo("Samples", &msaaIdx, msaaLabels, 2))
            pendingMsaaSamples = (msaaIdx == 1) ? VK_SAMPLE_COUNT_4_BIT
                                                : VK_SAMPLE_COUNT_2_BIT;
    }
    if (pendingAaMode != aaMode || pendingMsaaSamples != msaaSamples)
        ImGui::TextColored({1.f, 1.f, 0.f, 1.f}, "Applying...");

    if (nLines > 0) {
        ImGui::Separator();
        ImGui::Text("Line style");
        int prevLineMode = lineMode;
        ImGui::RadioButton("Bresenham (1 px)",     &lineMode, 0);
        ImGui::RadioButton("Quads - constant width", &lineMode, 1);
        // Offered only for checkpoints saved with bresen_lines=False, which
        // are the only ones carrying meaningful per-vertex radii.
        if (hasVertexRadius)
            ImGui::RadioButton("Quads - per-vertex radius", &lineMode, 2);

        // Only meaningful for the uniform-width style; hidden otherwise.
        // Driven in units of 1e-3 world units: ImGui's logarithmic slider
        // derives an epsilon of pow(0.1, format_precision) and fudges any
        // endpoint below it, so a raw 1e-4..1e-2 range with a low-precision
        // format collapses.  0.1..10 with "%.2f" keeps the epsilon (0.01)
        // safely under the minimum.
        if (lineMode == 1) {
            float widthMilli = lineRadius * 1e3f;
            ImGui::SliderFloat("Line width (x1e-3)", &widthMilli,
                               0.1f, 10.0f, "%.2f",
                               ImGuiSliderFlags_Logarithmic);
            lineRadius = widthMilli * 1e-3f;
            // Applying on release rather than per-frame: each apply rewrites
            // and re-uploads the whole radius column, so dragging would stall
            // the loop once per frame.
            if (ImGui::IsItemDeactivatedAfterEdit())
                applyLineWidth(lineRadius);
        }
        if (lineMode != prevLineMode)
            applyLineWidth(lineMode == 1 ? lineRadius : -1.f);
    }

    ImGui::Separator();
    if (!screenshotSupported) ImGui::BeginDisabled();
    if (ImGui::Button("Save screenshot"))
        screenshotPending = true;
    if (!screenshotSupported) {
        ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::TextDisabled("(surface has no TRANSFER_SRC)");
    }

    ImGui::End();
    ImGui::Render();
}

// ---------------------------------------------------------------------------
// Screenshot.  The copy is recorded inside the frame's command buffer, after
// the scene (and Gaussian resolve) pass but before the ImGui pass, so the
// panel is not in the picture.  The file is written once that frame is done.
// ---------------------------------------------------------------------------
void ViewerImpl::recordScreenshotCopy(VkCommandBuffer cb, uint32_t imgIdx) {
    const VkDeviceSize bytes = VkDeviceSize(scExt.width) * scExt.height * 4;
    mkBuf(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
          shotBuf, shotMem);

    auto barrier = [&](VkImageLayout from, VkImageLayout to,
                       VkAccessFlags src, VkAccessFlags dst) {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.oldLayout = from;  b.newLayout = to;
        b.srcAccessMask = src;  b.dstAccessMask = dst;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = scImg[imgIdx];
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &b);
    };
    // The scene and Gaussian passes both leave the image COLOR_ATTACHMENT_OPTIMAL,
    // and the ImGui pass expects it back in that layout.
    barrier(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent      = {scExt.width, scExt.height, 1};
    vkCmdCopyImageToBuffer(cb, scImg[imgIdx], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           shotBuf, 1, &region);

    barrier(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
}

void ViewerImpl::finishScreenshot() {
    const uint32_t w = scExt.width, h = scExt.height;
    const VkDeviceSize bytes = VkDeviceSize(w) * h * 4;

    void* src = nullptr;
    vkMapMemory(dev, shotMem, 0, bytes, 0, &src);
    std::vector<uint8_t> rgb(size_t(w) * h * 3);
    const auto* p = static_cast<const uint8_t*>(src);
    const bool bgr = (scFmt == VK_FORMAT_B8G8R8A8_UNORM);
    for (size_t i = 0; i < size_t(w) * h; i++) {
        rgb[i*3+0] = bgr ? p[i*4+2] : p[i*4+0];
        rgb[i*3+1] = p[i*4+1];
        rgb[i*3+2] = bgr ? p[i*4+0] : p[i*4+2];
    }
    vkUnmapMemory(dev, shotMem);
    vkDestroyBuffer(dev, shotBuf, nullptr);  shotBuf = VK_NULL_HANDLE;
    vkFreeMemory(dev, shotMem, nullptr);     shotMem = VK_NULL_HANDLE;

    char stem[64];
    snprintf(stem, sizeof(stem), "screenshot_%04d", screenshotIndex);
    screenshotIndex++;
    const std::string base = shotDir.empty() ? std::string(stem)
                                             : shotDir + "/" + stem;
    const std::string png = base + ".png";

    if (!stbi_write_png(png.c_str(), int(w), int(h), 3, rgb.data(), int(w) * 3)) {
        fprintf(stderr, "[screenshot] could not write %s\n", png.c_str());
        fflush(stderr);
        return;
    }

    printf("[screenshot] wrote %s (%ux%u)\n", png.c_str(), w, h);
    fflush(stdout);
}

// ---------------------------------------------------------------------------
// drawFrame
// ---------------------------------------------------------------------------
void ViewerImpl::drawFrame() {
    buildImguiFrame();

    VK_CHECK(vkWaitForFences(dev, 1, &fence[frameIdx], VK_TRUE, UINT64_MAX));

    uint32_t imgIdx = 0;
    VkResult acqRes = vkAcquireNextImageKHR(dev, sc, UINT64_MAX,
                                             imgSem[frameIdx],
                                             VK_NULL_HANDLE, &imgIdx);
    if (acqRes == VK_ERROR_OUT_OF_DATE_KHR) { rebuildSC(); return; }
    VK_CHECK(vkResetFences(dev, 1, &fence[frameIdx]));

    VkCommandBuffer cb = cmdBuf[frameIdx];
    VK_CHECK(vkResetCommandBuffer(cb, 0));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VK_CHECK(vkBeginCommandBuffer(cb, &bi));

    PushConst pc{};
    cam.viewprojMatrix(pc.viewproj,
                       float(scExt.width) / float(scExt.height));
    cam.eye(pc.campos);
    pc.width  = sceneExt.width;
    pc.height = sceneExt.height;

    if (hasSh)
        dispatchShEval(cb, pc.campos);

    // -- Scene pass ---------------------------------------------------------
    const bool msaaActive = (msaaSamples != VK_SAMPLE_COUNT_1_BIT);
    VkClearValue clears[3] = {};
    clears[0].color        = {{bgColor[0], bgColor[1], bgColor[2], 1.f}};
    clears[1].depthStencil = {1.f, 0};

    VkRenderPassBeginInfo rpbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rpbi.renderPass        = sceneRp;
    rpbi.framebuffer       = sceneFbs[imgIdx];
    rpbi.renderArea.extent = sceneExt;
    rpbi.clearValueCount   = msaaActive ? 3u : 2u;
    rpbi.pClearValues      = clears;
    vkCmdBeginRenderPass(cb, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport vport{0.f, 0.f,
                     float(sceneExt.width), float(sceneExt.height),
                     0.f, 1.f};
    VkRect2D scissor{{0, 0}, sceneExt};
    vkCmdSetViewport(cb, 0, 1, &vport);
    vkCmdSetScissor (cb, 0, 1, &scissor);

    VkBuffer drawVbo = hasSh ? evalBuf : vBuf;

    if (nFaces > 0 && vBuf && fBuf) {
        VkPipeline activeFacePipe = backfaceCull ? facePipelineCull : facePipeline;
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, activeFacePipe);
        vkCmdPushConstants(cb, pipeLayout, VK_SHADER_STAGE_VERTEX_BIT,
                           0, sizeof(PushConst), &pc);
        VkDeviceSize off = 0;
        vkCmdBindVertexBuffers(cb, 0, 1, &drawVbo, &off);
        vkCmdBindIndexBuffer  (cb, fBuf, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cb, nFaces * 3, 1, 0, 0, 0);
    }

    if (nLines > 0 && vBuf && lBuf) {
        if (lineMode == 0) {
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, bresenPipeline);
            vkCmdPushConstants(cb, pipeLayout, VK_SHADER_STAGE_VERTEX_BIT,
                               0, sizeof(PushConst), &pc);
            VkDeviceSize off = 0;
            vkCmdBindVertexBuffers(cb, 0, 1, &drawVbo, &off);
            const bool strip = (nStripIndices > 0 && lStripBuf);
            vkCmdBindIndexBuffer  (cb, strip ? lStripBuf : lBuf, 0,
                                   VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cb, strip ? nStripIndices : nLines * 2,
                             1, 0, 0, 0);
        } else {   // lineMode 1 = uniform width, 2 = per-vertex radius
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, quadLinePipeline);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    pipeLayout, 0, 1, &descSet, 0, nullptr);
            vkCmdPushConstants(cb, pipeLayout, VK_SHADER_STAGE_VERTEX_BIT,
                               0, sizeof(PushConst), &pc);
            vkCmdDraw(cb, 6, nLines, 0, 0);
        }
    }
    vkCmdEndRenderPass(cb);

    // -- Gaussian resolve (gauss_msaa only) ---------------------------------
    if (aaMode == AA_GAUSS_MSAA) {
        VkRenderPassBeginInfo gb{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        gb.renderPass        = gaussRp;
        gb.framebuffer       = gaussFbs[imgIdx];
        gb.renderArea.extent = scExt;
        gb.clearValueCount   = 0;
        vkCmdBeginRenderPass(cb, &gb, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport outVp{0.f, 0.f, float(scExt.width), float(scExt.height), 0.f, 1.f};
        VkRect2D   outSc{{0, 0}, scExt};
        vkCmdSetViewport(cb, 0, 1, &outVp);
        vkCmdSetScissor (cb, 0, 1, &outSc);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, gaussPipeline);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                gaussPipeLayout, 0, 1, &gaussDescSet, 0, nullptr);
        vkCmdDraw(cb, 3, 1, 0, 0);
        vkCmdEndRenderPass(cb);
    }

    // Capture before ImGui draws, so the panel stays out of the image.
    screenshotRecorded = false;
    if (screenshotPending && screenshotSupported) {
        recordScreenshotCopy(cb, imgIdx);
        screenshotRecorded = true;
    }
    screenshotPending = false;

    // -- ImGui pass ---------------------------------------------------------
    VkRenderPassBeginInfo imrpbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    imrpbi.renderPass        = imguiRp;
    imrpbi.framebuffer       = imguiFbs[imgIdx];
    imrpbi.renderArea.extent = scExt;
    imrpbi.clearValueCount   = 0;
    vkCmdBeginRenderPass(cb, &imrpbi, VK_SUBPASS_CONTENTS_INLINE);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cb);
    vkCmdEndRenderPass(cb);

    VK_CHECK(vkEndCommandBuffer(cb));

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount   = 1;
    si.pWaitSemaphores      = &imgSem[frameIdx];
    si.pWaitDstStageMask    = &waitStage;
    si.commandBufferCount   = 1;
    si.pCommandBuffers      = &cb;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores    = &renSem[frameIdx];
    VK_CHECK(vkQueueSubmit(queue, 1, &si, fence[frameIdx]));

    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores    = &renSem[frameIdx];
    pi.swapchainCount     = 1;
    pi.pSwapchains        = &sc;
    pi.pImageIndices      = &imgIdx;
    if (screenshotRecorded) {
        VK_CHECK(vkQueueWaitIdle(queue));
        finishScreenshot();
        screenshotRecorded = false;
    }

    VkResult presRes = vkQueuePresentKHR(queue, &pi);
    if (presRes == VK_ERROR_OUT_OF_DATE_KHR || presRes == VK_SUBOPTIMAL_KHR
        || dirty) {
        dirty = false;
        rebuildSC();
    }

    frameIdx = (frameIdx + 1) % MF;
}

// ---------------------------------------------------------------------------
// Destroy helpers
// ---------------------------------------------------------------------------
void ViewerImpl::destroySceneFbs() {
    for (auto f : sceneFbs) if (f) vkDestroyFramebuffer(dev, f, nullptr);
    sceneFbs.clear();
}
void ViewerImpl::destroyImguiFbs() {
    for (auto f : imguiFbs) if (f) vkDestroyFramebuffer(dev, f, nullptr);
    imguiFbs.clear();
}
void ViewerImpl::destroyGaussFbs() {
    for (auto f : gaussFbs) if (f) vkDestroyFramebuffer(dev, f, nullptr);
    gaussFbs.clear();
}
void ViewerImpl::destroyDepth() {
    if (depView) { vkDestroyImageView(dev, depView, nullptr); depView = VK_NULL_HANDLE; }
    if (depImg)  { vkDestroyImage    (dev, depImg,  nullptr); depImg  = VK_NULL_HANDLE; }
    if (depMem)  { vkFreeMemory      (dev, depMem,  nullptr); depMem  = VK_NULL_HANDLE; }
}
void ViewerImpl::destroyMsaaColor() {
    if (msaaView) { vkDestroyImageView(dev, msaaView, nullptr); msaaView = VK_NULL_HANDLE; }
    if (msaaImg)  { vkDestroyImage    (dev, msaaImg,  nullptr); msaaImg  = VK_NULL_HANDLE; }
    if (msaaMem)  { vkFreeMemory      (dev, msaaMem,  nullptr); msaaMem  = VK_NULL_HANDLE; }
}
void ViewerImpl::destroySceneColor() {
    if (sceneColorView) { vkDestroyImageView(dev, sceneColorView, nullptr); sceneColorView = VK_NULL_HANDLE; }
    if (sceneColorImg)  { vkDestroyImage    (dev, sceneColorImg,  nullptr); sceneColorImg  = VK_NULL_HANDLE; }
    if (sceneColorMem)  { vkFreeMemory      (dev, sceneColorMem,  nullptr); sceneColorMem  = VK_NULL_HANDLE; }
}
void ViewerImpl::destroyPipelines() {
    if (facePipeline)     { vkDestroyPipeline(dev, facePipeline,     nullptr); facePipeline     = VK_NULL_HANDLE; }
    if (facePipelineCull) { vkDestroyPipeline(dev, facePipelineCull, nullptr); facePipelineCull = VK_NULL_HANDLE; }
    if (bresenPipeline)   { vkDestroyPipeline(dev, bresenPipeline,   nullptr); bresenPipeline   = VK_NULL_HANDLE; }
    if (quadLinePipeline) { vkDestroyPipeline(dev, quadLinePipeline, nullptr); quadLinePipeline = VK_NULL_HANDLE; }
}
void ViewerImpl::destroyGaussResolve() {
    destroyGaussFbs();
    if (gaussPipeline)   { vkDestroyPipeline         (dev, gaussPipeline,   nullptr); gaussPipeline   = VK_NULL_HANDLE; }
    if (gaussDescPool)   { vkDestroyDescriptorPool   (dev, gaussDescPool,   nullptr); gaussDescPool   = VK_NULL_HANDLE; gaussDescSet = VK_NULL_HANDLE; }
    if (gaussPipeLayout) { vkDestroyPipelineLayout   (dev, gaussPipeLayout, nullptr); gaussPipeLayout = VK_NULL_HANDLE; }
    if (gaussDescLayout) { vkDestroyDescriptorSetLayout(dev, gaussDescLayout, nullptr); gaussDescLayout = VK_NULL_HANDLE; }
    if (gaussSampler)    { vkDestroySampler          (dev, gaussSampler,    nullptr); gaussSampler    = VK_NULL_HANDLE; }
    if (gaussRp)         { vkDestroyRenderPass       (dev, gaussRp,         nullptr); gaussRp         = VK_NULL_HANDLE; }
}

// Swap-chain-sized resources (called on resize and on mode change).
void ViewerImpl::destroySCDependents() {
    destroyGaussFbs();
    destroySceneFbs();
    destroyImguiFbs();
    destroySceneColor();
    destroyMsaaColor();
    destroyDepth();
    for (auto v : scView) vkDestroyImageView(dev, v, nullptr);
    scView.clear();
    scImg.clear();
}

// ---------------------------------------------------------------------------
// Rebuilds
// ---------------------------------------------------------------------------
void ViewerImpl::rebuildSC() {
    int w = 0, h = 0;
    while (w == 0 || h == 0) {
        glfwGetFramebufferSize(win, &w, &h);
        glfwWaitEvents();
    }
    vkDeviceWaitIdle(dev);
    destroySCDependents();
#ifndef __APPLE__
    vsync = pendingVsync;
#endif
    VkSwapchainKHR old = sc;
    initSwapchain(old);
    vkDestroySwapchainKHR(dev, old, nullptr);
    ImGui_ImplVulkan_SetMinImageCount(uint32_t(scImg.size()));
    initDepth();
    initMsaaColor();
    initSceneColor();
    initSceneFbs();
    initImguiFbs();
    if (aaMode == AA_GAUSS_MSAA) {
        // Render pass, sampler, desc layout and pipeline survive; only the
        // framebuffers and descriptor point at scView/sceneColorView.
        gaussFbs.resize(scView.size());
        for (size_t i = 0; i < scView.size(); i++) {
            VkFramebufferCreateInfo ci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
            ci.renderPass      = gaussRp;
            ci.attachmentCount = 1;
            ci.pAttachments    = &scView[i];
            ci.width           = scExt.width;
            ci.height          = scExt.height;
            ci.layers          = 1;
            VK_CHECK(vkCreateFramebuffer(dev, &ci, nullptr, &gaussFbs[i]));
        }
        updateGaussDescriptor();
    }
}

void ViewerImpl::rebuildForModeChange() {
    vkDeviceWaitIdle(dev);
    destroySCDependents();
    VkSwapchainKHR old = sc;
    sc = VK_NULL_HANDLE;

    destroyPipelines();
    if (sceneRp) { vkDestroyRenderPass(dev, sceneRp, nullptr); sceneRp = VK_NULL_HANDLE; }
    destroyGaussResolve();

    aaMode      = pendingAaMode;
    msaaSamples = pendingMsaaSamples;
    resolveMsaaSamples();
    pendingMsaaSamples = msaaSamples;

    initSwapchain(old);
    vkDestroySwapchainKHR(dev, old, nullptr);
    ImGui_ImplVulkan_SetMinImageCount(uint32_t(scImg.size()));
    initDepth();
    initMsaaColor();
    initSceneColor();
    initSceneRenderPass();
    initSceneFbs();
    initImguiFbs();
    initPipelines();
    initGaussResolve();
}

// ---------------------------------------------------------------------------
// cleanup
// ---------------------------------------------------------------------------
void ViewerImpl::cleanup() {
    if (dev) vkDeviceWaitIdle(dev);

    auto destroyBuf = [&](VkBuffer& b, VkDeviceMemory& m) {
        if (b) { vkDestroyBuffer(dev, b, nullptr); b = VK_NULL_HANDLE; }
        if (m) { vkFreeMemory   (dev, m, nullptr); m = VK_NULL_HANDLE; }
    };
    destroyBuf(vBuf,    vMem);
    destroyBuf(shBuf,   shMem);
    destroyBuf(evalBuf, evalMem);
    destroyBuf(fBuf,    fMem);
    destroyBuf(lBuf,    lMem);
    destroyBuf(lStripBuf, lStripMem);

    if (compPipeline)     { vkDestroyPipeline(dev, compPipeline,     nullptr); compPipeline     = VK_NULL_HANDLE; }
    if (compPipeLayout)   { vkDestroyPipelineLayout(dev, compPipeLayout, nullptr); compPipeLayout = VK_NULL_HANDLE; }
    if (compDescPool)     { vkDestroyDescriptorPool(dev, compDescPool, nullptr); compDescPool = VK_NULL_HANDLE; }
    if (compDescLayout)   { vkDestroyDescriptorSetLayout(dev, compDescLayout, nullptr); compDescLayout = VK_NULL_HANDLE; }

    if (descPool)   { vkDestroyDescriptorPool(dev, descPool, nullptr); descPool = VK_NULL_HANDLE; }

    destroyGaussResolve();
    destroyPipelines();
    if (pipeLayout) { vkDestroyPipelineLayout(dev, pipeLayout, nullptr); pipeLayout = VK_NULL_HANDLE; }
    if (descLayout) { vkDestroyDescriptorSetLayout(dev, descLayout, nullptr); descLayout = VK_NULL_HANDLE; }

    for (int i = 0; i < MF; i++) {
        if (imgSem[i]) vkDestroySemaphore(dev, imgSem[i], nullptr);
        if (renSem[i]) vkDestroySemaphore(dev, renSem[i], nullptr);
        if (fence[i])  vkDestroyFence    (dev, fence[i],  nullptr);
    }
    if (cmdPool) vkDestroyCommandPool(dev, cmdPool, nullptr);

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    if (imguiPool) vkDestroyDescriptorPool(dev, imguiPool, nullptr);

    destroySCDependents();
    if (sceneRp) vkDestroyRenderPass(dev, sceneRp, nullptr);
    if (imguiRp) vkDestroyRenderPass(dev, imguiRp, nullptr);
    if (sc)      vkDestroySwapchainKHR(dev, sc, nullptr);

    if (dev)  vkDestroyDevice(dev, nullptr);
    if (surf) vkDestroySurfaceKHR(inst, surf, nullptr);
    if (inst) vkDestroyInstance(inst, nullptr);
    if (win)  glfwDestroyWindow(win);
    glfwTerminate();
}

// ---------------------------------------------------------------------------
// run - the main loop.
// ---------------------------------------------------------------------------
void ViewerImpl::run(const SceneData& s, int w, int h, AaMode initialAa,
                     VkSampleCountFlagBits initialSamples, const float bg[3],
                     bool flipUp, const std::string& shotDirIn) {
    cam.up_sign = flipUp ? -1 : +1;
    shotDir     = shotDirIn;
    // initPipelines() runs before uploadGeom(), and again on every swapchain
    // recreate.  Latch the strip count here so the Bresenham pipeline is built
    // for the buffer it will bind, identically on both paths.
    nStripIndices = s.num_strip_indices;
    aaMode        = initialAa;
    pendingAaMode = initialAa;
    bgColor[0] = bg[0];  bgColor[1] = bg[1];  bgColor[2] = bg[2];
    // resolveMsaaSamples() clamps this against device support after
    // initDevice(); AA_GAUSS_MSAA and AA_NONE force it to one sample.
    msaaSamples        = initialSamples;
    pendingMsaaSamples = initialSamples;

    initGlfw(w, h);
    initInstance();
    initSurface();
    pickPhysDevice();
    initDevice();

    initSwapchain();
    resolveMsaaSamples();
    pendingMsaaSamples = msaaSamples;

    initDepth();
    initMsaaColor();
    initSceneColor();
    initSceneRenderPass();
    initImguiRenderPass();
    initSceneFbs();
    initImguiFbs();
    initPipelineLayout();
    initPipelines();
    initGaussResolve();
    initCmds();
    initSync();
    initImgui();

    uploadGeom(s);
    cam.autoFit(s.vert_attrs.data(), s.num_verts);
    cam.setInitialOrbit();

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();
        bool modeChange =
            (pendingAaMode      != aaMode)       ||
            (pendingMsaaSamples != msaaSamples);
#ifndef __APPLE__
        bool vsyncChange = (pendingVsync != vsync);
#else
        bool vsyncChange = false;
#endif
        if (modeChange) {
            rebuildForModeChange();
            continue;
        }
        if (vsyncChange) {
            rebuildSC();
            continue;
        }
        drawFrame();
    }

    cleanup();
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
void launch(const SceneData& scene, int width, int height, int aa_mode,
            int msaa_samples, float bg_r, float bg_g, float bg_b,
            bool flip_up, const std::string& screenshot_dir) {
    if (aa_mode < 0 || aa_mode > 2)
        throw std::runtime_error(
            "aa_mode must be 0 (hw_msaa), 1 (gauss_msaa) or 2 (none)");
    VkSampleCountFlagBits samples;
    switch (msaa_samples) {
        case 1:  samples = VK_SAMPLE_COUNT_1_BIT; break;
        case 2:  samples = VK_SAMPLE_COUNT_2_BIT; break;
        case 4:  samples = VK_SAMPLE_COUNT_4_BIT; break;
        default: throw std::runtime_error("msaa_samples must be 1, 2 or 4");
    }
    const float bg[3] = {bg_r, bg_g, bg_b};
    ViewerImpl impl;
    impl.run(scene, width, height, static_cast<AaMode>(aa_mode), samples, bg,
             flip_up, screenshot_dir);
}

} // namespace fuzzydr_viewer

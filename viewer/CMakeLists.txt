cmake_minimum_required(VERSION 3.20)
project(fuzzydr_viewer VERSION 0.1.0 LANGUAGES CXX)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ----------------------------------
# Python + nanobind
# ----------------------------------
find_package(Python COMPONENTS Interpreter Development REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE nanobind_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${nanobind_DIR}")
find_package(nanobind CONFIG REQUIRED)

# ----------------------------------
# Vulkan
# ----------------------------------
find_package(Vulkan REQUIRED)

# ----------------------------------
# GLFW (fetched)
# ----------------------------------
include(FetchContent)

FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG        3.4
  GIT_SHALLOW    TRUE
)
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# ----------------------------------
# Dear ImGui (fetched)
# ----------------------------------
FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG        v1.91.4
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(imgui)

add_library(imgui_vk STATIC
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
)
target_include_directories(imgui_vk PUBLIC
  ${imgui_SOURCE_DIR}
  ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(imgui_vk PUBLIC Vulkan::Vulkan glfw)

# ----------------------------------
# Shader compilation (same pattern as fuzzydr)
# ----------------------------------
find_program(GLSLANG_VALIDATOR glslangValidator
  HINTS ENV VULKAN_SDK
  PATH_SUFFIXES Bin bin
)
if (NOT GLSLANG_VALIDATOR)
  message(FATAL_ERROR "glslangValidator not found. Install Vulkan SDK.")
endif()

set(SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(SPIRV_DIR  "${CMAKE_CURRENT_BINARY_DIR}/spv")
file(MAKE_DIRECTORY "${SPIRV_DIR}")

set(SHADER_OUTPUTS)

function(compile_glsl OUT_VAR STAGE INPUT_FILE)
  get_filename_component(_abs  "${INPUT_FILE}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  get_filename_component(_name "${_abs}" NAME)
  set(_out "${SPIRV_DIR}/${_name}.spv")
  add_custom_command(
    OUTPUT  "${_out}"
    COMMAND "${GLSLANG_VALIDATOR}" -V -S "${STAGE}" "${_abs}" -o "${_out}"
    DEPENDS "${_abs}" "${GLSLANG_VALIDATOR}"
    VERBATIM
  )
  set(${OUT_VAR} "${_out}" PARENT_SCOPE)
  set(SHADER_OUTPUTS "${SHADER_OUTPUTS};${_out}" PARENT_SCOPE)
endfunction()

compile_glsl(VIEWER_VERT_SPV            vert "${SHADER_DIR}/viewer.vert")
compile_glsl(VIEWER_LINE_QUADS_VERT_SPV vert "${SHADER_DIR}/viewer_line_quads.vert")
compile_glsl(VIEWER_FRAG_SPV            frag "${SHADER_DIR}/viewer.frag")
compile_glsl(VIEWER_SH_EVAL_COMP_SPV    comp "${SHADER_DIR}/viewer_sh_eval.comp")
compile_glsl(VIEWER_GAUSS_RESOLVE_VERT_SPV vert "${SHADER_DIR}/viewer_gauss_resolve.vert")
compile_glsl(VIEWER_GAUSS_RESOLVE_FRAG_SPV frag "${SHADER_DIR}/viewer_gauss_resolve.frag")
add_custom_target(fuzzydr_viewer_shaders DEPENDS ${SHADER_OUTPUTS})

# ----------------------------------
# nanobind module
# ----------------------------------
nanobind_add_module(_core
  src/viewer.h
  src/viewer.cpp
  src/camera.h
  src/offline/offline.h
  src/offline/offline.cpp
  src/main.cpp
)

add_dependencies(_core fuzzydr_viewer_shaders)

target_include_directories(_core PRIVATE src)
# Vulkan and glfw arrive through imgui_vk, which links them PUBLIC.
target_link_libraries(_core PRIVATE imgui_vk)

if (MSVC)
  target_compile_options(_core PRIVATE /utf-8)
endif()

# Put SPV files next to the Python extension inside the package (same as fuzzydr)
install(FILES ${SHADER_OUTPUTS} DESTINATION fuzzydr_viewer)

install(TARGETS _core
  LIBRARY DESTINATION fuzzydr_viewer
  RUNTIME DESTINATION fuzzydr_viewer
  ARCHIVE DESTINATION fuzzydr_viewer
)

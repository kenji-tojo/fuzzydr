cmake_minimum_required(VERSION 3.20)
project(fuzzydr VERSION 0.1.0 LANGUAGES CXX)

# Default to Release if no build type specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ----------------------------------
# C++ standard
# ----------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Good default for Python extension modules
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ----------------------------------
# Python
# ----------------------------------
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# ----------------------------------
# nanobind (from pip)
# ----------------------------------
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
# Optional CUDA
# ----------------------------------
option(FUZZYDR_ENABLE_CUDA "Enable CUDA backend if CUDA is available" ON)

set(FUZZYDR_WITH_CUDA 0)

if (FUZZYDR_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)

  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)

    # Match the CUDA standard to C++17.
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Position-independent code for CUDA objects in the shared module.
    set(CMAKE_CUDA_POSITION_INDEPENDENT_CODE ON)

    set(FUZZYDR_WITH_CUDA 1)

    message(STATUS "CUDA detected: building with CUDA support (FUZZYDR_WITH_CUDA=1)")
  else()
    message(STATUS "CUDA not available: building CPU-only backend (FUZZYDR_WITH_CUDA=0)")
  endif()
else()
  message(STATUS "FUZZYDR_ENABLE_CUDA=OFF: building CPU-only")
endif()

# ----------------------------------
# Shader compile (glslangValidator from Vulkan SDK)
# ----------------------------------
find_program(GLSLANG_VALIDATOR glslangValidator
  HINTS ENV VULKAN_SDK
  PATH_SUFFIXES Bin bin
)

if (NOT GLSLANG_VALIDATOR)
  message(FATAL_ERROR "glslangValidator not found. Install Vulkan SDK and ensure it's on PATH.")
endif()

# ----------------------------------
# Shader compile helper
# ----------------------------------
set(SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(SPIRV_DIR  "${CMAKE_CURRENT_BINARY_DIR}/spv")
file(MAKE_DIRECTORY "${SPIRV_DIR}")

set(SHADER_OUTPUTS)

function(compile_glsl OUT_VAR STAGE INPUT_FILE)
  get_filename_component(_in_abs "${INPUT_FILE}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

  # Keep full filename and just append ".spv"
  # rasterize.vert -> rasterize.vert.spv
  # edge_grad.comp -> edge_grad.comp.spv
  get_filename_component(_name "${_in_abs}" NAME)
  set(_out "${SPIRV_DIR}/${_name}.spv")

  add_custom_command(
    OUTPUT "${_out}"
    COMMAND "${GLSLANG_VALIDATOR}" -V -S "${STAGE}" "${_in_abs}" -o "${_out}"
    DEPENDS "${_in_abs}" "${GLSLANG_VALIDATOR}"
    VERBATIM
  )

  set(${OUT_VAR} "${_out}" PARENT_SCOPE)
  set(SHADER_OUTPUTS "${SHADER_OUTPUTS};${_out}" PARENT_SCOPE)
endfunction()

# ----------------------------------
# Define shaders
# ----------------------------------
compile_glsl(RASTERIZE_VERT_SPV             vert "${SHADER_DIR}/rasterize.vert")
compile_glsl(RASTERIZE_LINE_VERT_QUADS_SPV  vert "${SHADER_DIR}/rasterize_line_quads.vert")
compile_glsl(RASTERIZE_LINE_VERT_BRESEN_SPV vert "${SHADER_DIR}/rasterize_line_bresen.vert")
compile_glsl(RASTERIZE_POINT_VERT_SPV       vert "${SHADER_DIR}/rasterize_point.vert")
compile_glsl(RASTERIZE_FRAG_SPV             frag "${SHADER_DIR}/rasterize.frag")
compile_glsl(OPACITY_GRAD_SPV               frag "${SHADER_DIR}/opacity_grad.frag")
compile_glsl(EDGE_GRAD_SPV                  comp "${SHADER_DIR}/edge_grad.comp")
compile_glsl(MSAA_RESOLVE_FWD_SPV           comp "${SHADER_DIR}/msaa_resolve_fwd.comp")
compile_glsl(MSAA_RESOLVE_BWD_SPV           comp "${SHADER_DIR}/msaa_resolve_bwd.comp")

add_custom_target(shaders DEPENDS ${SHADER_OUTPUTS})

# ----------------------------------
# Build nanobind module
# ----------------------------------
nanobind_add_module(_core
  src/vk_common.h
  src/vk_common.cpp
  src/rasterize.h
  src/rasterize.cpp
  src/filter.h
  src/filter.cpp
  src/main.cpp
)

if (FUZZYDR_WITH_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(_core PRIVATE CUDA::cuda_driver CUDA::cudart)
endif()

target_compile_definitions(_core PRIVATE FUZZYDR_WITH_CUDA=${FUZZYDR_WITH_CUDA})

add_dependencies(_core shaders)

target_include_directories(_core PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_link_libraries(_core PRIVATE Vulkan::Vulkan)

# Put SPIR-V next to the Python extension inside the package
install(FILES ${SHADER_OUTPUTS} DESTINATION fuzzydr)

# ----------------------------------
# Install into your Python package
# ----------------------------------
install(TARGETS _core
  LIBRARY DESTINATION fuzzydr
  RUNTIME DESTINATION fuzzydr
  ARCHIVE DESTINATION fuzzydr
)

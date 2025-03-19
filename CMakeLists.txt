cmake_minimum_required(VERSION 3.10)
project(FastMultipoleMethod CUDA CXX)

# Find packages
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# Set CUDA flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 -arch=sm_52)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

# Add CUDA sources
cuda_add_executable(fmm_nbody
    main.cu
    fastMultipoleCuda.cu
    fastMultipole_kernel.cu
)

# Link libraries
target_link_libraries(fmm_nbody
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
) 
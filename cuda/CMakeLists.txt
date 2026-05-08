cmake_minimum_required(VERSION 3.20)
project(MyProject LANGUAGES CXX CUDA)

file(GLOB SOURCE_FILES "../src/QComputations/*.cpp" "../src/QComputations/*.cu")

find_package(CUDA 13 REQUIRED)
# Set Intel OneAPI installation path
#set(LIB_ROOT "/usr/lib/x86_64-linux-gnu/lib")
message(STATUS "linux home dir: $ENV{PATH}")
# Set Intel MKL variables
#set(MKL_INCLUDE_DIRS "$ENV{ONEAPI_ROOT}/mkl/latest/include")
#set(MKL_LIBRARIES "$ENV{ONEAPI_ROOT}/mkl/latest/lib/intel64")

# Set Intel Python variables
set(INTEL_PYTHON_INCLUDE_DIRS "$ENV{ONEAPI_ROOT}/intelpython/latest/include/python3.9" "$ENV{ONEAPI_ROOT}/intelpython/python3.9/lib/python3.9/site-packages/numpy/core/include")
set(CUDA_DIRECTORIES "/usr/local/cuda-13/include")
set(INTEL_PYTHON_LIBRARIES "$ENV{ONEAPI_ROOT}/intelpython/latest/lib")
set(BUILD_SHARED_LIBS ON)

# Function to create custom build targets for the given options
function(create_build_target TARGET_NAME PYTHON_FLAG)
    # Create target
    add_library(${TARGET_NAME} SHARED ${SOURCE_FILES})

    # Set flags
    #target_compile_options(${TARGET_NAME} PRIVATE --compiler-options -fPIC)
    target_compile_options(${TARGET_NAME} PRIVATE -DENABLE_CUDA)
    #target_include_directories(${TARGET_NAME} PRIVATE ${CUDA_DIRECTORIES})
    set_target_properties(${TARGET_NAME} PROPERTIES
        CXX_STANDARD 17  # Установите стандарт C++ при необходимости
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
    )

   #target_link_libraries(${TARGET_NAME} PRIVATE cublas)

    #target_compile_options(${TARGET_NAME} PRIVATE)

    # Link libraries
    target_link_libraries(${TARGET_NAME} cublas cusolver cusparse)
    #target_link_libraries(${TARGET_NAME} /lib/x86_64-linux-gnu/libm.so.6)
    #target_link_libraries(${TARGET_NAME} ${MKL_LIBRARIES}/libmkl_intel_lp64.so)
    #target_link_libraries(${TARGET_NAME} ${MKL_LIBRARIES}/libmkl_intel_thread.so)
    #target_link_libraries(${TARGET_NAME} ${MKL_LIBRARIES}/libmkl_core.so)
    #target_link_libraries(${TARGET_NAME} ${SCALAPACK_LIBRARIES})
    #target_link_libraries(${TARGET_NAME} ${BLACS_LIBRARIES})

    if (PYTHON_FLAG)
        #target_link_libraries(${TARGET_NAME} ${INTEL_PYTHON_LIBRARIES}/libiomp5.so)
        target_link_libraries(${TARGET_NAME} ${INTEL_PYTHON_LIBRARIES}/libpython3.9.so)
        target_compile_options(${TARGET_NAME} PRIVATE -DENABLE_MATPLOTLIB)
        target_include_directories(${TARGET_NAME} PRIVATE ${INTEL_PYTHON_INCLUDE_DIRS})
        #target_link_libraries(${TARGET_NAME} ${INTEL_PYTHON_LIBRARIES})
    endif()

    #target_include_directories(${TARGET_NAME} PRIVATE ${MKL_INCLUDE_DIRS})
    #target_link_options(${TARGET_NAME} PRIVATE -qmkl=parallel)
endfunction()

# Create build targets
#create_build_target(QComputations_CUDA ON)
create_build_target(QComputations_CUDA_NO_PLOTS OFF)

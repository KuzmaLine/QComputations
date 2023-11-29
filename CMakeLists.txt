cmake_minimum_required(VERSION 3.12)
project(MyProject)

# Set Intel OneAPI installation path
set(ONEAPI_ROOT "/opt/intel/oneapi/")

# Set Intel MKL variables
set(MKL_INCLUDE_DIRS "${ONEAPI_ROOT}/mkl/latest/include")
set(MKL_LIBRARIES "${ONEAPI_ROOT}/mkl/latest/lib")

# Set BLACS and ScaLAPACK variables
set(SCALAPACK_INCLUDE_DIRS "${ONEAPI_ROOT}/mkl/latest/include")
set(SCALAPACK_LIBRARIES "${ONEAPI_ROOT}/mkl/latest/lib/intel64/libmkl_scalapack_lp64.so")
set(BLACS_LIBRARIES "${ONEAPI_ROOT}/mkl/latest/lib/intel64/libmkl_blacs_intelmpi_lp64.so")

# Set Intel Python variables
set(INTEL_PYTHON_INCLUDE_DIRS "${ONEAPI_ROOT}/intelpython/latest/include")
set(INTEL_PYTHON_LIBRARIES "${ONEAPI_ROOT}/intelpython/latest/lib")

# Set Intel MPI variables
set(MPI_INCLUDE_DIRS "${ONEAPI_ROOT}/mpi/latest/include")
set(MPI_LIBRARIES "${ONEAPI_ROOT}/mpi/latest/lib/release/libmpi.so")

# Add source files to be compiled
file(GLOB SOURCE_FILES "src/*.cpp")

# Function to create custom build targets for the given options
function(create_build_target TARGET_NAME MKL_FLAG BLACS_FLAG PYTHON_FLAG MPI_FLAG)
    # Create target
    add_library(${TARGET_NAME} SHARED ${SOURCE_FILES})

    # Set compiler and flags
    target_compile_options(${TARGET_NAME} PRIVATE "-std=c++11")

    # Link libraries
    target_include_directories(${TARGET_NAME} PRIVATE ${MKL_INCLUDE_DIRS})
    target_include_directories(${TARGET_NAME} PRIVATE ${SCALAPACK_INCLUDE_DIRS})

    target_link_libraries(${TARGET_NAME} ${MKL_LIBRARIES})
    target_link_libraries(${TARGET_NAME} ${SCALAPACK_LIBRARIES})
    target_link_libraries(${TARGET_NAME} ${BLACS_LIBRARIES})

    if (PYTHON_FLAG)
        target_include_directories(${TARGET_NAME} PRIVATE ${INTEL_PYTHON_INCLUDE_DIRS})
        target_link_libraries(${TARGET_NAME} ${INTEL_PYTHON_LIBRARIES})
    endif()

    if (MPI_FLAG)
        target_include_directories(${TARGET_NAME} PRIVATE ${MPI_INCLUDE_DIRS})
        target_link_libraries(${TARGET_NAME} ${MPI_LIBRARIES})
    endif()
endfunction()

# Create build targets
create_build_target(QComputations_CPU_CLUSTER ON ON ON ON)
create_build_target(QComputations_CPU_CLUSTER_NO_PLOTS ON ON OFF ON)
create_build_target(QComputations_SINGLE ON ON ON OFF)
create_build_target(QComputations_SINGLE_NO_PLOTS ON ON OFF OFF)
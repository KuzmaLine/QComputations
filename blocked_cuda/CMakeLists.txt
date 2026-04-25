cmake_minimum_required(VERSION 3.20)
project(MyProject LANGUAGES CXX CUDA)

file(GLOB SOURCE_FILES "../src/QComputations/*.cpp" "../src/QComputations/*.cu")

# Поиск пакетов
find_package(CUDA 13 REQUIRED)
find_package(MPI REQUIRED)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(MPI_CXX_COMPILE_FLAGS "")
    set(MPI_CXX_LINK_FLAGS "")
endif()

# Поиск библиотек
find_library(CUBLAS_LIBRARY cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
find_library(CUSOLVER_LIBRARY cusolver HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
find_library(CUBLASMP_LIBRARY cublasmp HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
find_library(NCCL_LIBRARY nccl HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# Функция создания цели
function(create_build_target TARGET_NAME)
    add_library(${TARGET_NAME} SHARED ${SOURCE_FILES})
    set_target_properties(${TARGET_NAME} PROPERTIES
        CXX_STANDARD 17
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
    target_include_directories(${TARGET_NAME} PRIVATE
        ${CUDA_INCLUDE_DIRS}
        ${MPI_CXX_INCLUDE_PATH}
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
    )
    target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_CUDA ENABLE_MPI ENABLE_CLUSTER)
    target_link_libraries(${TARGET_NAME} PRIVATE
        ${CUBLAS_LIBRARY} ${CUSOLVER_LIBRARY} ${CUBLASMP_LIBRARY} ${NCCL_LIBRARY}
        ${MPI_CXX_LIBRARIES}
    )
    target_compile_options(${TARGET_NAME} PRIVATE ${MPI_CXX_COMPILE_FLAGS})
    target_link_options(${TARGET_NAME} PRIVATE ${MPI_CXX_LINK_FLAGS})
endfunction()

# Сборка цели
create_build_target(QComputations_BLOCKED_CUDA)
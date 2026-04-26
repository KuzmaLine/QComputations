cmake_minimum_required(VERSION 3.20)
project(TestBlockedCuda LANGUAGES CXX CUDA)

# Поиск пакетов
find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)

# Убедитесь, что переменная CUDA_TOOLKIT_ROOT_DIR заполнена (после find_package она есть)
if(NOT CUDA_TOOLKIT_ROOT_DIR)
    message(FATAL_ERROR "CUDA toolkit not found")
endif()

# Путь к заголовкам вашей библиотеки
set(BLOCKED_CUDA_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../../../src/QComputations")

# Исполняемый файл
add_executable(qme_blocked_cuda qme_blocked_cuda.cu)
set_target_properties(qme_blocked_cuda PROPERTIES
    CXX_STANDARD 17
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(qme_blocked_cuda PRIVATE
    ${BLOCKED_CUDA_INCLUDE_DIR}
    ${MPI_CXX_INCLUDE_PATH}
    ${CUDA_INCLUDE_DIRS}
)

target_compile_definitions(qme_blocked_cuda PRIVATE ENABLE_CUDA ENABLE_MPI)

# Поиск библиотек CUDA – указываем HINTS на lib64 в установке CUDA
find_library(CUBLAS_LIB cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
find_library(CUSOLVER_LIB cusolver HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
find_library(CUBLASMP_LIB NAMES cublasMp cublasmp HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
find_library(NCCL_LIB nccl HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# Поиск вашей библиотеки QComputations_BLOCKED_CUDA (подставьте нужный путь)
find_library(QC_LIB
    NAMES QComputations_BLOCKED_CUDA
    HINTS
        ${CMAKE_BINARY_DIR}/..
        "${CMAKE_SOURCE_DIR}/../build"
        /usr/local/lib
        /usr/lib
        "${CMAKE_SOURCE_DIR}/../../qme_blocked_cuda/build"   # пример вашего пути
)
if(NOT QC_LIB)
    message(FATAL_ERROR "Library QComputations_BLOCKED_CUDA not found. Specify with -DQC_LIB=/path/to/lib")
endif()

# Проверяем, что все библиотеки найдены
foreach(lib CUBLAS_LIB CUSOLVER_LIB CUBLASMP_LIB NCCL_LIB)
    if(NOT ${lib})
        message(FATAL_ERROR "${lib} not found. Check CUDA installation.")
    endif()
endforeach()

# Линковка
target_link_libraries(qme_blocked_cuda
    ${QC_LIB}
    ${CUBLAS_LIB}
    ${CUSOLVER_LIB}
    ${CUBLASMP_LIB}
    ${NCCL_LIB}
    ${MPI_CXX_LIBRARIES}
)
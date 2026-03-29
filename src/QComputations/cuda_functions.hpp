#ifdef __CUDACC__
#pragma once
#include <iostream>
#include <typeinfo>
#include <cublas_v2.h>
#include <cuda.h>
#include <cstdio>
#include <cusolverDn.h>

#include "matrix.hpp"
#include "cuda_kernels.hpp"

#define CSC(call)  									                \
do {											                    \
    cudaError_t res = call;							                \
    if (res != cudaSuccess) {							            \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
                __FILE__, __LINE__, cudaGetErrorString(res));		\
        exit(0);								                    \
    }										                        \
} while(0)

#define CUBLASSC(x) do { if((x)!=CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "Error %s at %s:%d\n", cublasGetStatusString(x), __FILE__, __LINE__);\
    exit(EXIT_FAILURE);}} while(0)

#define CUSOLVERSC(call)                                                        \
do {                                                                          \
    cusolverStatus_t res = call;                                              \
    if (res != CUSOLVER_STATUS_SUCCESS) {                                     \
        const char* err_str = cusolverGetErrorString(res);                    \
        fprintf(stderr, "ERROR in %s:%d. CuSolver error %d: %s\n",            \
                __FILE__, __LINE__, (int)res, err_str ? err_str : "Unknown"); \
        exit(0);                                                              \
    }                                                                         \
} while(0)

// For CUSOLVER errors
static const char* cusolverGetErrorString(cusolverStatus_t status) {
    switch(status) {
        case CUSOLVER_STATUS_SUCCESS: return "CUSOLVER_STATUS_SUCCESS";
        case CUSOLVER_STATUS_NOT_INITIALIZED: return "CUSOLVER_STATUS_NOT_INITIALIZED";
        case CUSOLVER_STATUS_ALLOC_FAILED: return "CUSOLVER_STATUS_ALLOC_FAILED";
        case CUSOLVER_STATUS_INVALID_VALUE: return "CUSOLVER_STATUS_INVALID_VALUE";
        case CUSOLVER_STATUS_ARCH_MISMATCH: return "CUSOLVER_STATUS_ARCH_MISMATCH";
        case CUSOLVER_STATUS_MAPPING_ERROR: return "CUSOLVER_STATUS_MAPPING_ERROR";
        case CUSOLVER_STATUS_EXECUTION_FAILED: return "CUSOLVER_STATUS_EXECUTION_FAILED";
        case CUSOLVER_STATUS_INTERNAL_ERROR: return "CUSOLVER_STATUS_INTERNAL_ERROR";
        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        case CUSOLVER_STATUS_NOT_SUPPORTED: return "CUSOLVER_STATUS_NOT_SUPPORTED";
        case CUSOLVER_STATUS_ZERO_PIVOT: return "CUSOLVER_STATUS_ZERO_PIVOT";
        case CUSOLVER_STATUS_INVALID_LICENSE: return "CUSOLVER_STATUS_INVALID_LICENSE";
        default: return NULL;
    }
}

namespace QComputations {

namespace CUDA {
    __host__ void cudaMalloc(void** dev_ptr, size_t dev_size);
    __host__ void cudaMemcpy(void* dst, const void* src, size_t size, enum ::cudaMemcpyKind kind);
    __host__ void cudaFree(void* dst);
/* -------------------------------------------------- CUBLAS ----------------------------------------------------------------------- */
    __host__ void cublasSetMatrix(ILP_TYPE n, ILP_TYPE m, ILP_TYPE elem_size, const void* A, ILP_TYPE LDA, void* B, ILP_TYPE LDB);
    __host__ void cublasGetMatrix(ILP_TYPE n, ILP_TYPE m, ILP_TYPE elem_size, const void* A, ILP_TYPE LDA, void* B, ILP_TYPE LDB);


    __host__ void cublasDGEMM(cublasHandle_t handle, cublasOperation_t is_trans_A, cublasOperation_t is_trans_B, ILP_TYPE n, ILP_TYPE m, ILP_TYPE k, const double* A, ILP_TYPE lda, const double* B, ILP_TYPE ldb, double* C, ILP_TYPE ldc, double alpha = 1, double beta = 0);
    __host__ void cublasZGEMM(cublasHandle_t handle, cublasOperation_t is_trans_A, cublasOperation_t is_trans_B, ILP_TYPE n, ILP_TYPE m, ILP_TYPE k, const cuDoubleComplex* A, ILP_TYPE lda, const cuDoubleComplex* B, ILP_TYPE ldb, cuDoubleComplex* C, ILP_TYPE ldc, cuDoubleComplex alpha = cuDoubleComplex{1, 0}, cuDoubleComplex beta = {0, 0});

    __host__ void cublasDGEMV(cublasHandle_t handle, cublasOperation_t is_trans, ILP_TYPE n, ILP_TYPE m, const double* A, ILP_TYPE lda, const double* x, double* y, double alpha = 1, double beta = 0);
    __host__ void cublasZGEMV(cublasHandle_t handle, cublasOperation_t is_trans, ILP_TYPE n, ILP_TYPE m, const cuDoubleComplex* A, ILP_TYPE lda, const cuDoubleComplex* x, cuDoubleComplex* y, cuDoubleComplex alpha = cuDoubleComplex{1, 0}, cuDoubleComplex beta = {0, 0});

    __host__ void cublasDADD(cublasHandle_t handle, int size, const double* A, double* B, double alpha = 1);
    __host__ void cublasZADD(cublasHandle_t handle, int size, const cuDoubleComplex* A, cuDoubleComplex* B, cuDoubleComplex alpha = cuDoubleComplex{1, 0});

    __host__ void cublasDSCAL(cublasHandle_t handle, int size, double* A, double alpha);
    __host__ void cublasZSCAL(cublasHandle_t handle, int size, cuDoubleComplex* A, COMPLEX alpha);

    __host__ void cublasDGEAM(cublasHandle_t handle, cublasOperation_t is_trans_A, cublasOperation_t is_trans_B, ILP_TYPE n, ILP_TYPE m, const double* A, ILP_TYPE lda, const double* B, ILP_TYPE ldb, double* C, ILP_TYPE ldc, double alpha = 1, double beta = 1);
    __host__ void cublasZGEAM(cublasHandle_t handle, cublasOperation_t is_trans_A, cublasOperation_t is_trans_B, ILP_TYPE n, ILP_TYPE m, const cuDoubleComplex* A, ILP_TYPE lda, const cuDoubleComplex* B, ILP_TYPE ldb, cuDoubleComplex* C, ILP_TYPE ldc, cuDoubleComplex alpha = cuDoubleComplex{1, 0}, cuDoubleComplex beta = {1, 0});

/* -------------------------------------------------- CUSOLVER DENSE ----------------------------------------------------------------- */
    __host__ ILP_TYPE cusolverDnZHEEVD_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, ILP_TYPE n, const cuDoubleComplex* A, ILP_TYPE lda, const double* W);
    __host__ void cusolverDnZHEEVD(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, ILP_TYPE n, cuDoubleComplex* A, ILP_TYPE lda, double* W, cuDoubleComplex* work, ILP_TYPE lwork, ILP_TYPE* devInfo);

/* --------------------------------------------------- HELPER KERNELS ---------------------------------------------------------------- */
    __global__ void set_default_value_double(double* data, size_t n, size_t m, double value);
    __global__ void set_default_value_complex(cuDoubleComplex* data, size_t n, size_t m, cuDoubleComplex value);
}

}

#endif
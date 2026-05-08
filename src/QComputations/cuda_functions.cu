#ifdef __CUDACC__

#include "cuda_functions.hpp"

namespace QComputations {

namespace CUDA {
    __host__ void cudaMalloc(void** dev_ptr, size_t dev_size) {
        CSC(::cudaMalloc(dev_ptr, dev_size));
    }

    __host__ void cudaMemcpy(void* dst, const void* src, size_t size, enum ::cudaMemcpyKind kind) {
        CSC(::cudaMemcpy(dst, src, size, kind));
    }

    __host__ void cudaFree(void* dst) {
        if (dst == nullptr) {
            return;
        }

        CSC(::cudaFree(dst));
    }

/* -------------------------------------------------- CUBLAS ----------------------------------------------------------------------- */

    __host__ void cublasSetMatrix(ILP_TYPE n, ILP_TYPE m, ILP_TYPE elem_size, const void* A, ILP_TYPE LDA, void* B, ILP_TYPE LDB) {
        // std::cout << n << " " << m << " " << elem_size << " " << LDA << " " << LDB << std::endl;
        CUBLASSC(::cublasSetMatrix(n, m, elem_size, A, LDA, B, LDB));
    }

    __host__ void cublasGetMatrix(ILP_TYPE n, ILP_TYPE m, ILP_TYPE elem_size, const void* A, ILP_TYPE LDA, void* B, ILP_TYPE LDB) {
        CUBLASSC(::cublasGetMatrix(n, m, elem_size, A, LDA, B, LDB));
    }


    __host__ void cublasDGEMM(cublasHandle_t handle, cublasOperation_t is_trans_A, cublasOperation_t is_trans_B, ILP_TYPE n, ILP_TYPE m, ILP_TYPE k, const double* A, ILP_TYPE lda, const double* B, ILP_TYPE ldb, double* C, ILP_TYPE ldc, double alpha, double beta) {
        CUBLASSC(::cublasDgemm(handle, is_trans_A, is_trans_B, n, m, k, &alpha, A, lda, B, ldb, &beta, C, ldc));
    }

    __host__ void cublasZGEMM(cublasHandle_t handle, cublasOperation_t is_trans_A, cublasOperation_t is_trans_B, ILP_TYPE n, ILP_TYPE m, ILP_TYPE k, const cuDoubleComplex* A, ILP_TYPE lda, const cuDoubleComplex* B, ILP_TYPE ldb, cuDoubleComplex* C, ILP_TYPE ldc, cuDoubleComplex alpha, cuDoubleComplex beta) {
        CUBLASSC(::cublasZgemm(handle, is_trans_A, is_trans_B, n, m, k, &alpha, A, lda, B, ldb, &beta, C, ldc));
    }


    __host__ void cublasDGEMV(cublasHandle_t handle, cublasOperation_t is_trans, ILP_TYPE n, ILP_TYPE m, const double* A, ILP_TYPE lda, const double* x, double* y, double alpha, double beta) {
        CUBLASSC(::cublasDgemv(handle, is_trans, n, m, &alpha, A, lda, x, 1, &beta, y, 1));
    }

    __host__ void cublasZGEMV(cublasHandle_t handle, cublasOperation_t is_trans, ILP_TYPE n, ILP_TYPE m, const cuDoubleComplex* A, ILP_TYPE lda, const cuDoubleComplex* x, cuDoubleComplex* y, cuDoubleComplex alpha, cuDoubleComplex beta) {
        CUBLASSC(::cublasZgemv(handle, is_trans, n, m, &alpha, A, lda, x, 1, &beta, y, 1));
    }

    __host__ void cublasDADD(cublasHandle_t handle, int size, const double* A, double* B, double alpha) {
        CUBLASSC(::cublasDaxpy(handle, size, &alpha, A, 1, B, 1));
    }
    __host__ void cublasZADD(cublasHandle_t handle, int size, const cuDoubleComplex* A, cuDoubleComplex* B, cuDoubleComplex alpha) {
        CUBLASSC(::cublasZaxpy(handle, size, &alpha, A, 1, B, 1));
    }

    __host__ void cublasDGEAM(cublasHandle_t handle, cublasOperation_t trans_A, cublasOperation_t trans_B, ILP_TYPE n, ILP_TYPE m, const double* A, ILP_TYPE lda, const double* B, ILP_TYPE ldb, double* C, ILP_TYPE ldc, double alpha, double beta) {
        CUBLASSC(::cublasDgeam(handle, trans_A, trans_B, n, m, &alpha, A, lda, &beta, B, ldb, C, ldc));
    }

    __host__ void cublasZGEAM(cublasHandle_t handle, cublasOperation_t trans_A, cublasOperation_t trans_B, ILP_TYPE n, ILP_TYPE m, const cuDoubleComplex* A, ILP_TYPE lda, const cuDoubleComplex* B, ILP_TYPE ldb, cuDoubleComplex* C, ILP_TYPE ldc, cuDoubleComplex alpha, cuDoubleComplex beta) {
        CUBLASSC(::cublasZgeam(handle, trans_A, trans_B, n, m, &alpha, A, lda, &beta, B, ldb, C, ldc));
    }

    __host__ void cublasDSCAL(cublasHandle_t handle, int size, double* A, double alpha) {
        CUBLASSC(::cublasDscal(handle, size, &alpha, A, 1));
    }
    __host__ void cublasZSCAL(cublasHandle_t handle, int size, cuDoubleComplex* A, COMPLEX alpha) {
        cuDoubleComplex* _alpha = reinterpret_cast<cuDoubleComplex*>(&alpha);
        CUBLASSC(::cublasZscal(handle, size, _alpha, A, 1));
    }

/* -------------------------------------------------- CUSOLVER DENSE ----------------------------------------------------------------- */

    __host__ ILP_TYPE cusolverDnZHEEVD_bufferSize(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, ILP_TYPE n, const cuDoubleComplex* A, ILP_TYPE lda, const double* W) {
        ILP_TYPE lwork;
        CUSOLVERSC(::cusolverDnZheevd_bufferSize(handle, jobz, uplo, n, A, lda, W, &lwork));
        return lwork;
    }

    __host__ void cusolverDnZHEEVD(cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, ILP_TYPE n, cuDoubleComplex* A, ILP_TYPE lda, double* W, cuDoubleComplex* work, ILP_TYPE lwork, ILP_TYPE* devInfo) {
        CUSOLVERSC(::cusolverDnZheevd(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo));
    }

/* --------------------------------------------------- HELPER KERNELS ---------------------------------------------------------------- */

    __global__ void set_default_value_double(double* data, size_t n, size_t m, double value) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int grid_size = blockDim.x * gridDim.x;
        size_t size = n * m;

        for (size_t i = idx; i < size; i += grid_size) {
            data[i] = value;
        }
    }

    __global__ void set_default_value_complex(cuDoubleComplex* data, size_t n, size_t m, cuDoubleComplex value) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        int grid_size = blockDim.x * gridDim.x;
        size_t size = n * m;

        for (size_t i = idx; i < size; i += grid_size) {
            data[i] = value;
        }
    }
}

}

#endif
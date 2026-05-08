#include "QComputations_CUDA_NO_PLOTS.hpp"

using namespace QComputations;

COMPLEX func(int i, int j) {
    return 2 * i + j;
}

// Only hermitian
COMPLEX func_2(int i, int j) {
    return COMPLEX(i, 0) + COMPLEX(j, 0) + (i >= j ? COMPLEX(0, 1) : COMPLEX(0, -1)) - (i == j ? COMPLEX(0, 1) : 0);
}

int main() {
    int n = 4;
    int m = 4;

    cublasHandle_t handle;
    cusparseHandle_t sparse_handle;
    cublasCreate(&handle);
    cusparseCreate(&sparse_handle);

    CUDA_Matrix<COMPLEX> A(handle, n, m, func);
    CUDA_CSR_Matrix<COMPLEX> CSR_A(sparse_handle, n, m, func);

    A.show();

    std::cout << std::endl;

    CSR_A.show();

    std::cout << std::endl;

    auto B = A * A;

    B.show();

    std::cout << std::endl;

    CUDA_Matrix<COMPLEX> C(handle, B.n(), B.m());

    optimized_multiply(A, A, C, cuDoubleComplex{1, 0}, cuDoubleComplex{0, 0});

    C.show();

    std::cout << std::endl;

    CUDA_CSR_Matrix<COMPLEX> CSR_C(sparse_handle, n, m);

    optimized_multiply(CSR_A, CSR_A, CSR_C, cuDoubleComplex{1, 0}, cuDoubleComplex{0, 0});

    CSR_C.show();

    auto new_C = A;

    optimized_multiply(CSR_A, A, new_C, cuDoubleComplex{1, 0}, cuDoubleComplex{0, 0});

    std::cout << std::endl;

    new_C.show();

    CUDA_Matrix<COMPLEX> left_C(handle, n, m);

    optimized_multiply(A, CSR_A, left_C, cuDoubleComplex{1, 0}, cuDoubleComplex{0, 0});

    std::cout << std::endl;

    left_C.show();

    cublasDestroy(handle);
    cusparseDestroy(sparse_handle);

    return 0;
}
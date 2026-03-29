#include "QComputations_CUDA_NO_PLOTS.hpp"

double func(int i, int j) {
    return i + j;
}

double func_b(int i, int j) {
    return i - j;
}

int main(int argc, char** argv) {
    using namespace QComputations;
    cublasHandle_t handle;
    cublasCreate(&handle);
    QConfig::instance().set_width(5);
    Matrix<double> m(FORTRAN_STYLE, 4, 4, double(1));
    m(0, 0) = 2;
    m(1, 0) = 3;
    CUDA_Matrix<double> cuda_m(handle, m);
    cuda_m.show();

    Matrix<double> a(FORTRAN_STYLE, 3, 3, func);
    Matrix<double> b(FORTRAN_STYLE, 3, 3, func_b);

    a.show();
    CUDA_Matrix<double> A(handle, a);
    A.show();
    CUDA_Matrix<double> B(handle, b);
    B.show();
    CUDA_Matrix<double> res(handle, 3, 3);

    optimized_multiply(A, B, res, double(1), double(0), CUBLAS_OP_N, CUBLAS_OP_T);
    res.show();

    cublasDestroy(handle);

    return 0;
}
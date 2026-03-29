#ifdef __CUDACC__

#include "cuda_matrix.hpp"

namespace QComputations {

namespace {
    constexpr cublasOperation_t NO_TRANS = CUBLAS_OP_N;
}

template<>
CUDA_Matrix<double> CUDA_Matrix<double>::operator*(const CUDA_Matrix<double>& other) const {
    CUDA_Matrix<double> res(this->handle_, this->n(), other.m());
    if (other.n() == 1 || other.m() == 1) {
        CUDA::cublasDGEMV(handle_, NO_TRANS, this->n(), this->m(), this->data(), this->ld(), other.data(), res.data());
    } else {
        CUDA::cublasDGEMM(handle_, NO_TRANS, NO_TRANS, this->n(), other.m(), this->m(), this->data(), this->ld(), other.data(), other.ld(), res.data(), res.ld());
    }

    return res;
}

template<>
CUDA_Matrix<COMPLEX> CUDA_Matrix<COMPLEX>::operator*(const CUDA_Matrix<COMPLEX>& other) const {
    CUDA_Matrix<COMPLEX> res(this->handle_, this->n(), other.m());
    if (other.n() == 1 || other.m() == 1) {
        CUDA::cublasZGEMV(handle_, NO_TRANS, this->n(), this->m(), this->data(), this->ld(), other.data(), res.data());
    } else {
        CUDA::cublasZGEMM(handle_, NO_TRANS, NO_TRANS, this->n(), other.m(), this->m(), this->data(), this->ld(), other.data(), other.ld(), res.data(), res.ld());
    }

    return res;
}

template<>
void optimized_multiply(const CUDA_Matrix<double>& A, const CUDA_Matrix<double>& B, CUDA_Matrix<double>& C,
                        double alpha, double betta, cublasOperation_t trans_A, cublasOperation_t trans_B) {
    CUDA::cublasDGEMM(A.handle(), trans_A, trans_B, A.n(), B.m(), A.m(), A.data(), A.ld(), B.data(), B.ld(), C.data(), C.ld());
}

template<>
void optimized_multiply(const CUDA_Matrix<COMPLEX>& A, const CUDA_Matrix<COMPLEX>& B, CUDA_Matrix<COMPLEX>& C,
                        cuDoubleComplex alpha, cuDoubleComplex betta, cublasOperation_t trans_A, cublasOperation_t trans_B) {
    CUDA::cublasZGEMM(A.handle(), trans_A, trans_B, A.n(), B.m(), A.m(), A.data(), A.ld(), B.data(), B.ld(), C.data(), C.ld(), alpha, betta);
}

template<>
void optimized_add(CUDA_Matrix<double>& A, const CUDA_Matrix<double>& B, CUDA_Matrix<double>& C,
                        double alpha, double betta, cublasOperation_t trans_A, cublasOperation_t trans_B) {
    CUDA::cublasDGEAM(A.handle(), trans_A, trans_B, A.n(), B.m(), A.data(), A.ld(), B.data(), B.ld(), C.data(), C.ld(), alpha, betta);
}

template<>
void optimized_add(CUDA_Matrix<COMPLEX>& A, const CUDA_Matrix<COMPLEX>& B, CUDA_Matrix<COMPLEX>& C,
                        cuDoubleComplex alpha, cuDoubleComplex betta, cublasOperation_t trans_A, cublasOperation_t trans_B) {
    CUDA::cublasZGEAM(A.handle(), trans_A, trans_B, A.n(), B.m(), A.data(), A.ld(), B.data(), B.ld(), C.data(), C.ld(), alpha, betta);
}

template<>
CUDA_Matrix<double> CUDA_Matrix<double>::operator+(const CUDA_Matrix<double>& other) const {
    assert(this->n() == other.n() && this->m() == other.m());
    CUDA_Matrix<double> res(other);
    CUDA::cublasDADD(handle_, this->n() * this->m(), this->data(), res.data());

    return res;
}

template<>
CUDA_Matrix<COMPLEX> CUDA_Matrix<COMPLEX>::operator+(const CUDA_Matrix<COMPLEX>& other) const {
    assert(this->n() == other.n() && this->m() == other.m());
    CUDA_Matrix<COMPLEX> res(other);
    CUDA::cublasZADD(handle_, this->n() * this->m(), this->data(), res.data());

    return res;
}

template<>
CUDA_Matrix<double> CUDA_Matrix<double>::operator*(double num) const {
    CUDA_Matrix<double> res(*this);
    CUDA::cublasDSCAL(handle_, this->n() * this->m(), res.data(), num);

    return res;
}

template<>
CUDA_Matrix<COMPLEX> CUDA_Matrix<COMPLEX>::operator*(COMPLEX num) const {
    CUDA_Matrix<COMPLEX> res(*this);
    CUDA::cublasZSCAL(handle_, this->n() * this->m(), res.data(), num);

    return res;
}

template<>
CUDA_Matrix<double>& CUDA_Matrix<double>::operator+=(const CUDA_Matrix<double>& A) {
    CUDA::cublasDADD(handle_, this->n() * this->m(), A.data(), this->data());
    return *this;
}

template<>
CUDA_Matrix<COMPLEX>& CUDA_Matrix<COMPLEX>::operator+=(const CUDA_Matrix<COMPLEX>& A) {
    CUDA::cublasZADD(handle_, this->n() * this->m(), A.data(), this->data());
    return *this;
}

std::vector<CUDA_Matrix<COMPLEX>> CUDA_QME_OPT_Runge_Kutt_4(const std::vector<double>& x,
                                        const CUDA_Matrix<COMPLEX>& y0,
                                        std::function<void(double, const CUDA_Matrix<COMPLEX>&, CUDA_Matrix<COMPLEX>&)> f) {
    size_t len = x.size();
    size_t dim = y0.n();
    std::vector<CUDA_Matrix<COMPLEX>> y(len, CUDA_Matrix<COMPLEX>(y0.handle(), dim, dim));
    y[0] = y0;

    CUDA_Matrix<COMPLEX> k1(y0.handle(), dim, dim);
    CUDA_Matrix<COMPLEX> k2(y0.handle(), dim, dim);
    CUDA_Matrix<COMPLEX> k3(y0.handle(), dim, dim);

    for (size_t i = 0; i < len - 1; i++) {
        //if (i % (len / 100) == 0) std::cout << i << " " << len << std::endl;
        //std::cout << i << " " << y[i] << " ";
        double h = x[i + 1] - x[i];

        f(x[i], y[i], k1);
        optimized_add(y[i], k1, k1, cuDoubleComplex{1, 0}, cuDoubleComplex{h / 2.0, 0});
        f(x[i] + h / 2.0, k1, k2);
        optimized_add(y[i], k2, k2, cuDoubleComplex{1, 0}, cuDoubleComplex{h / 2.0, 0});
        f(x[i] + h / 2.0, k2, k3);
        optimized_add(y[i], k3, k3, cuDoubleComplex{1, 0}, cuDoubleComplex{h, 0});
        f(x[i] + h, k3, y[i + 1]);
        optimized_add(y[i], k1, k1, cuDoubleComplex{double(-2)/h, 0}, cuDoubleComplex{double(2) / h, 0});
        optimized_add(y[i], k2, k2, cuDoubleComplex{double(-2)/h, 0}, cuDoubleComplex{double(2) / h, 0});
        optimized_add(y[i], k3, k3, cuDoubleComplex{double(-1)/h, 0}, cuDoubleComplex{double(1) / h, 0});

        optimized_add(k3, y[i + 1], y[i + 1], cuDoubleComplex{(h / 3.0), 0}, cuDoubleComplex{(h / 6.0), 0});
        optimized_add(k2, y[i + 1], y[i + 1], cuDoubleComplex{(h / 3.0), 0}, cuDoubleComplex{1, 0});
        optimized_add(k1, y[i + 1], y[i + 1], cuDoubleComplex{(h / 6.0), 0}, cuDoubleComplex{1, 0});
        optimized_add(y[i], y[i + 1], y[i + 1], cuDoubleComplex{1, 0}, cuDoubleComplex{1, 0});
        //y[i + 1] = y[i]  + (k1 + (k2 + k3) * 2 + k4) * (h / 6.0);
        //std::cout << h << " " << y[i + 1] << " " << 2 * x[i + 1] << std::endl;
    }

    return y;
}

std::vector<CUDA_Matrix<COMPLEX>> CUDA_QME_OPT_Runge_Kutt_2(const std::vector<double>& x,
                                        const CUDA_Matrix<COMPLEX>& y0,
                                        std::function<void(double, const CUDA_Matrix<COMPLEX>&, CUDA_Matrix<COMPLEX>&)> f) {
    size_t len = x.size();
    size_t dim = y0.n();
    std::vector<CUDA_Matrix<COMPLEX>> y(len, CUDA_Matrix<COMPLEX>(y0.handle(), dim, dim));
    y[0] = y0;

    CUDA_Matrix<COMPLEX> k1(y0.handle(), dim, dim);

    for (size_t i = 0; i < len - 1; i++) {
        double h = x[i + 1] - x[i];

        f(x[i], y[i], k1);
        optimized_add(y[i], k1, k1, cuDoubleComplex{1, 0}, cuDoubleComplex{h, 0});
        f(x[i] + h, k1, y[i + 1]);
        optimized_add(y[i], k1, k1, cuDoubleComplex{double(-1)/h, 0}, cuDoubleComplex{double(1) / h, 0});
        optimized_add(k1, y[i + 1], y[i + 1], cuDoubleComplex{h / 2.0, 0}, cuDoubleComplex{h / 2.0, 0});
        optimized_add(y[i], y[i + 1], y[i + 1], cuDoubleComplex{1, 0}, cuDoubleComplex{1, 0});
        //y[i + 1] = y[i]  + (k1 + k2) * (h / 2.0);
        //std::cout << h << " " << y[i + 1] << " " << 2 * x[i + 1] << std::endl;
    }

    return y;
}

std::pair<double*, CUDA_Matrix<COMPLEX>> HermitEigen(const CUDA_Matrix<COMPLEX>& A) {
    if (A.n() != A.m()) {
        throw std::invalid_argument("Matrix must be square for eigenvalue decomposition");
    }

    cusolverDnHandle_t cusolver_handle;
    cusolverStatus_t cusolver_status = cusolverDnCreate(&cusolver_handle);

    double* d_W;
    cudaMalloc((void**)&d_W, sizeof(double) * A.n());

    CUDA_Matrix<COMPLEX> A_copy(A);

    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

    auto lwork = CUDA::cusolverDnZHEEVD_bufferSize(
        cusolver_handle, jobz, uplo, A.n(), 
        // reinterpret_cast<const cuDoubleComplex*>(A_copy.data()),
        A_copy.data(), 
        A_copy.ld(), d_W
    );

    cuDoubleComplex* d_work = nullptr;
    cudaMalloc((void**)&d_work, sizeof(cuDoubleComplex) * lwork);

    ILP_TYPE* d_info = nullptr;
    cudaMalloc((void**)&d_info, sizeof(ILP_TYPE));

    CUDA::cusolverDnZHEEVD(
        cusolver_handle, jobz, uplo, A.n(),
        A_copy.data(), A_copy.ld(),
        d_W, d_work, lwork, d_info
    );

    cudaFree(d_info);
    cudaFree(d_work);
    cusolverDnDestroy(cusolver_handle);

    return std::make_pair(d_W, std::move(A_copy));
}

}

#endif
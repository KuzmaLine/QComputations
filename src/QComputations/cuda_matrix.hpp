#ifdef __CUDACC__

#pragma once
#include <iostream>
#include "cuda_functions.hpp"
#include <typeinfo>

namespace QComputations {

template <typename T, typename GPU_T = typename std::conditional_t<
    std::is_same<T, COMPLEX>::value, cuDoubleComplex,
    typename std::conditional_t<std::is_same<T, double>::value, double, void>>
>
class CUDA_Matrix {
    public:
        explicit CUDA_Matrix() = default;
        explicit CUDA_Matrix(cublasHandle_t handle, ILP_TYPE n, ILP_TYPE m): handle_(handle), n_(n), m_(m) {
            CUDA::cudaMalloc(reinterpret_cast<void**>(&dev_mass_), sizeof(GPU_T) * n_ * m_);
        }

        explicit CUDA_Matrix(cublasHandle_t handle, ILP_TYPE n, ILP_TYPE m, const std::vector<T>& matrix): handle_(handle), n_(n), m_(m) {
            CUDA::cudaMalloc(reinterpret_cast<void**>(&dev_mass_), sizeof(GPU_T) * n_ * m_);
            CUDA::cublasSetMatrix(n_, m_, sizeof(T), matrix.data(), n_, dev_mass_, m_);
        }

        explicit CUDA_Matrix(cublasHandle_t handle, ILP_TYPE n, ILP_TYPE m, T default_value): handle_(handle), n_(n), m_(m) {
            CUDA::cudaMalloc(reinterpret_cast<void**>(&dev_mass_), sizeof(GPU_T)*n_*m_);
        }

        explicit CUDA_Matrix(cublasHandle_t handle, const Matrix<T>& matrix): handle_(handle), n_(matrix.n()), m_(matrix.m()) {
            assert(matrix.matrix_style() == FORTRAN_STYLE);
            CUDA::cudaMalloc(reinterpret_cast<void**>(&dev_mass_), sizeof(GPU_T)*n_*m_);
            CUDA::cublasSetMatrix(n_, m_, sizeof(T), matrix.data(), matrix.LD(), dev_mass_, m_);
        }

        CUDA_Matrix(const CUDA_Matrix<T>& other): handle_(other.handle_), n_(other.n_), m_(other.m_) {
            CUDA::cudaMalloc(reinterpret_cast<void**>(&(this->dev_mass_)), sizeof(GPU_T)*n_*m_);
            CUDA::cudaMemcpy(this->dev_mass_, other.dev_mass_, sizeof(GPU_T)*n_*m_, cudaMemcpyDeviceToDevice);
        }

        CUDA_Matrix(cublasHandle_t handle, ILP_TYPE n, ILP_TYPE m, std::function<T(ILP_TYPE, ILP_TYPE)> func):handle_(handle), n_(n), m_(m) {
            CUDA::cudaMalloc(reinterpret_cast<void**>(&dev_mass_), sizeof(GPU_T) * n_ * m_);
            Matrix<T> matrix(FORTRAN_STYLE, n, m, func);
            CUDA::cublasSetMatrix(n_, m_, sizeof(T), matrix.data(), matrix.LD(), dev_mass_, m_);
        }

        CUDA_Matrix(CUDA_Matrix<T>&& other) noexcept: handle_(other.handle_), n_(other.n_), m_(other.m_), dev_mass_(other.dev_mass_) {
            other.dev_mass_ = nullptr;
        }

        CUDA_Matrix<T>& operator=(const CUDA_Matrix<T>& A) {
            handle_ = A.handle_;
            n_ = A.n_;
            m_ = A.m_;
            CUDA::cudaMalloc(reinterpret_cast<void**>(&(dev_mass_)), sizeof(GPU_T)*n_*m_);
            CUDA::cudaMemcpy(dev_mass_, A.dev_mass_, sizeof(GPU_T)*n_*m_, cudaMemcpyDeviceToDevice);
            return *this;
        }

        CUDA_Matrix<T> operator*(const CUDA_Matrix<T>& other) const;
        CUDA_Matrix<T> operator+(const CUDA_Matrix<T>& other) const;
        CUDA_Matrix<T> operator-(const CUDA_Matrix<T>& other) const;
        CUDA_Matrix<T>& operator*=(const CUDA_Matrix<T>& other);
        CUDA_Matrix<T>& operator+=(const CUDA_Matrix<T>& other);
        CUDA_Matrix<T>& operator-=(const CUDA_Matrix<T>& other);

        CUDA_Matrix<T> operator*(T num) const;
        void operator*=(T num);

        ~CUDA_Matrix() {
            CUDA::cudaFree(dev_mass_);
            //cublasDestroy(&handle_);
        }

        void show(size_t width = QConfig::instance().width()) const {
            Matrix<T> tmp(FORTRAN_STYLE, n_, m_);
            CUDA::cublasGetMatrix(n_, m_, sizeof(GPU_T), dev_mass_, this->ld(), tmp.data(), tmp.LD());
            tmp.show(width);
        }
        
        Matrix<T> to_cpu() const {
            Matrix<T> tmp(FORTRAN_STYLE, n_, m_);
            CUDA::cublasGetMatrix(n_, m_, sizeof(GPU_T), dev_mass_, this->ld(), tmp.data(), tmp.LD());
            return tmp;
        }

        ILP_TYPE n() const { return n_; }
        ILP_TYPE m() const { return m_; }
        ILP_TYPE ld() const { return n_; } // FORTRAN STYLE
        GPU_T* data() { return dev_mass_; }
        const GPU_T* data() const { return dev_mass_; }
        cublasHandle_t handle() const { return handle_; }

        CUDA_Matrix<T> copy(const CUDA_Matrix<T>& A) const;

        void write_to_csv_file(const std::string& filename) const { this->to_cpu().write_to_csv_file(filename); }
    private:
        cublasHandle_t handle_;
        ILP_TYPE n_;
        ILP_TYPE m_;
        GPU_T* dev_mass_ = nullptr;
};

template<>
CUDA_Matrix<double> CUDA_Matrix<double>::operator*(const CUDA_Matrix<double>& other) const;

template<>
CUDA_Matrix<COMPLEX> CUDA_Matrix<COMPLEX>::operator*(const CUDA_Matrix<COMPLEX>& other) const;

template<>
CUDA_Matrix<double> CUDA_Matrix<double>::operator+(const CUDA_Matrix<double>& other) const;

template<>
CUDA_Matrix<COMPLEX> CUDA_Matrix<COMPLEX>::operator+(const CUDA_Matrix<COMPLEX>& other) const;

template<>
CUDA_Matrix<double> CUDA_Matrix<double>::operator*(double num) const;

template<>
CUDA_Matrix<COMPLEX> CUDA_Matrix<COMPLEX>::operator*(COMPLEX num) const;

template <typename T, typename GPU_T = typename std::conditional_t<
    std::is_same<T, COMPLEX>::value, cuDoubleComplex,
    typename std::conditional_t<std::is_same<T, double>::value, double, void>>
>
void optimized_add(CUDA_Matrix<T>& A, const CUDA_Matrix<T>& B, CUDA_Matrix<T>& C,
                        GPU_T alpha, GPU_T betta, cublasOperation_t trans_A = CUBLAS_OP_N, cublasOperation_t trans_B = CUBLAS_OP_N);

void optimized_add(CUDA_Matrix<COMPLEX>& A, const CUDA_Matrix<COMPLEX>& B, CUDA_Matrix<COMPLEX>& C,
                        cuDoubleComplex alpha, cuDoubleComplex betta, cublasOperation_t trans_A, cublasOperation_t trans_B);

template<>
void optimized_add(CUDA_Matrix<double>& A, const CUDA_Matrix<double>& B, CUDA_Matrix<double>& C,
                        double alpha, double betta, cublasOperation_t trans_A, cublasOperation_t trans_B);


template <typename T, typename GPU_T = typename std::conditional_t<
    std::is_same<T, COMPLEX>::value, cuDoubleComplex,
    typename std::conditional_t<std::is_same<T, double>::value, double, void>>
>
void optimized_multiply(const CUDA_Matrix<T>& A, const CUDA_Matrix<T>& B, CUDA_Matrix<T>& C,
                        GPU_T alpha, GPU_T betta, cublasOperation_t trans_A = CUBLAS_OP_N, cublasOperation_t trans_B = CUBLAS_OP_N);

template<>
void optimized_multiply(const CUDA_Matrix<COMPLEX>& A, const CUDA_Matrix<COMPLEX>& B, CUDA_Matrix<COMPLEX>& C,
                        cuDoubleComplex alpha, cuDoubleComplex betta, cublasOperation_t trans_A, cublasOperation_t trans_B);

template<>
void optimized_multiply(const CUDA_Matrix<double>& A, const CUDA_Matrix<double>& B, CUDA_Matrix<double>& C,
                        double alpha, double betta, cublasOperation_t trans_A, cublasOperation_t trans_B);

std::vector<CUDA_Matrix<COMPLEX>> CUDA_QME_OPT_Runge_Kutt_4(const std::vector<double>& x,
                                        const CUDA_Matrix<COMPLEX>& y0,
                                        std::function<void(double, const CUDA_Matrix<COMPLEX>&, CUDA_Matrix<COMPLEX>&)> f);

std::vector<CUDA_Matrix<COMPLEX>> CUDA_QME_OPT_Runge_Kutt_2(const std::vector<double>& x,
                                        const CUDA_Matrix<COMPLEX>& y0,
                                        std::function<void(double, const CUDA_Matrix<COMPLEX>&, CUDA_Matrix<COMPLEX>&)> f);

std::pair<double*, CUDA_Matrix<COMPLEX>> HermitEigen(const CUDA_Matrix<COMPLEX>& A);
}

#endif
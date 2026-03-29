#pragma once
#include <mkl_sparse_handle.h>
#include <mkl_spblas.h>

#include <complex>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include "additional_operators.hpp"
#include "config.hpp"
#include "matrix.hpp"

namespace QComputations {

    namespace {
        template <typename T>
        bool is_close(T a, T b) {
            return (std::abs(a - b) < QConfig::instance().eps());
        }

        ILP_TYPE get_index_value(ILP_TYPE row_index, const MKL_INT* ia) { return ia[row_index]; }
    }  // namespace

    // WRITE TO ALL TEMPLATE VERSIONS
    template <typename T>
    class CSR_Matrix {
       public:
        explicit CSR_Matrix() = default;
        explicit CSR_Matrix(const Matrix<T>& A, T default_value = T(0));
        explicit CSR_Matrix(std::function<T(ILP_TYPE, ILP_TYPE)> func, T default_value = T(0));
        explicit CSR_Matrix(const CSR_Matrix<T>& A);
        explicit CSR_Matrix(ILP_TYPE n,
                            ILP_TYPE m,
                            const std::vector<T>& vals,
                            const std::vector<MKL_INT>& ia,
                            const std::vector<MKL_INT>& ja);
        explicit CSR_Matrix(
            const std::function<void(std::vector<MKL_INT>& ia, std::vector<MKL_INT>& ja, std::vector<T>& vals)>&,
            T default_value = T(0));

        ~CSR_Matrix() {
            if (mkl_matrix_ != nullptr) {
                mkl_sparse_destroy(mkl_matrix_);
            }
        }

        void insert_value(ILP_TYPE i, ILP_TYPE j, T value);
        const T operator()(ILP_TYPE i, ILP_TYPE j) const;
        T operator()(ILP_TYPE i, ILP_TYPE j);
        void delete_value(ILP_TYPE i, ILP_TYPE j);
        bool is_contain(ILP_TYPE i, ILP_TYPE j) const;

        const CSR_Matrix<T> operator*(const CSR_Matrix<T>& A) const;
        void operator*=(const CSR_Matrix<T>& A);
        const CSR_Matrix<T> operator+(const CSR_Matrix<T>& A) const;
        void operator+=(const CSR_Matrix<T>& A);

        const std::vector<T> operator*(const std::vector<T>& x) const;

        void show() const;

        sparse_matrix_t mkl_matrix() const { return mkl_matrix_; }

       private:
        bool is_exported_ = true;
        ILP_TYPE n_ = 0;
        ILP_TYPE m_ = 0;
        T default_value_ = T(0);
        std::unique_ptr<T[]> vals_;
        std::unique_ptr<MKL_INT[]> ia_;
        std::unique_ptr<MKL_INT[]> ja_;
        sparse_matrix_t mkl_matrix_ = nullptr;
        // TEMP
        sparse_matrix_type_t matrix_type_ = SPARSE_MATRIX_TYPE_GENERAL;
    };

    // NAIVE VERSION for not double and complex double
    template <typename T>
    CSR_Matrix<T>::CSR_Matrix(const Matrix<T>& A, T default_value)
        : n_(A.n()), m_(A.m()), default_value_(default_value) {
        std::vector<MKL_INT> ia_vec(n_ + 1, 0);
        std::vector<MKL_INT> ja_vec;
        std::vector<T> vals_vec;

        for (ILP_TYPE i = 0; i < n_; i++) {
            ia_vec[i + 1] = ia_vec[i];
            for (ILP_TYPE j = 0; j < m_; j++) {
                if (!is_close(A.elem(i, j), default_value)) {
                    ia_vec[i + 1]++;
                    vals_vec.emplace_back(A.elem(i, j));
                    ja_vec.emplace_back(j);
                }
            }
        }

        ia_ = std::make_unique<MKL_INT[]>(n_ + 1);
        std::copy(ia_vec.begin(), ia_vec.end(), ia_.get());

        if (!ja_vec.empty()) {
            ja_ = std::make_unique<MKL_INT[]>(ja_vec.size());
            std::copy(ja_vec.begin(), ja_vec.end(), ja_.get());

            vals_ = std::make_unique<T[]>(vals_vec.size());
            std::copy(vals_vec.begin(), vals_vec.end(), vals_.get());
        }
    }

    template <typename T>
    CSR_Matrix<T>::CSR_Matrix(ILP_TYPE n,
                              ILP_TYPE m,
                              const std::vector<T>& vals,
                              const std::vector<MKL_INT>& ia,
                              const std::vector<MKL_INT>& ja)
        : n_(n), m_(m) {
        ia_ = std::make_unique<MKL_INT[]>(n_ + 1);
        std::copy(ia.begin(), ia.end(), ia_.get());

        if (!ja.empty()) {
            ja_ = std::make_unique<MKL_INT[]>(ja.size());
            std::copy(ja.begin(), ja.end(), ja_.get());

            vals_ = std::make_unique<T[]>(vals.size());
            std::copy(vals.begin(), vals.end(), vals_.get());
        }
    }

    template <typename T>
    CSR_Matrix<T>::CSR_Matrix(const CSR_Matrix<T>& A) : n_(A.n_), m_(A.m_), default_value_(A.default_value_) {
        ia_ = std::make_unique<MKL_INT[]>(n_ + 1);
        std::copy(A.ia_.get(), A.ia_.get() + n_ + 1, ia_.get());

        size_t nnz = A.ia_[n_];
        if (nnz > 0) {
            ja_ = std::make_unique<MKL_INT[]>(nnz);
            std::copy(A.ja_.get(), A.ja_.get() + nnz, ja_.get());

            vals_ = std::make_unique<T[]>(nnz);
            std::copy(A.vals_.get(), A.vals_.get() + nnz, vals_.get());
        }
    }

    template <typename T>
    const T CSR_Matrix<T>::operator()(ILP_TYPE i, ILP_TYPE j) const {
        ILP_TYPE index_value = ia_[i];

        for (ILP_TYPE k = 0; k < ia_[i + 1] - ia_[i]; k++) {
            if (ja_[index_value] == j) {
                return vals_[index_value];
            }
            index_value++;
        }

        return default_value_;
    }

    template <typename T>
    void CSR_Matrix<T>::show() const {
        for (ILP_TYPE i = 0; i < n_; i++) {
            for (ILP_TYPE j = 0; j < m_; j++) {
                std::cout << std::setw(QConfig::instance().width()) << (*this)(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }

    // ---------------------------------------------- FUNCTIONS -----------------------------------------------------

    inline void sparse_spmm(const sparse_matrix_t A, const sparse_matrix_t B, sparse_matrix_t* C, char op = 'N');

    void sparse_add(const sparse_matrix_t A, const sparse_matrix_t B, sparse_matrix_t* C, double alpha, char op = 'N');
    void sparse_add(const sparse_matrix_t A, const sparse_matrix_t B, sparse_matrix_t* C, COMPLEX alpha, char op = 'N');

    template <typename T>
    void optimized_multiply(const CSR_Matrix<T>& A, const CSR_Matrix<T>& B, CSR_Matrix<T>& C, char op = 'N');

    template <>
    void optimized_multiply(const CSR_Matrix<double>& A, const CSR_Matrix<double>& B, CSR_Matrix<double>& C, char op);

    template <>
    void optimized_multiply(const CSR_Matrix<COMPLEX>& A,
                            const CSR_Matrix<COMPLEX>& B,
                            CSR_Matrix<COMPLEX>& C,
                            char op);

    template <typename T>
    void optimized_multiply(const CSR_Matrix<T>& A,
                            const Matrix<T>& B,
                            Matrix<T>& C,
                            T alpha = T(1),
                            T betta = T(0),
                            char op = 'N');

    template <>
    void optimized_multiply(const CSR_Matrix<double>& A,
                            const Matrix<double>& B,
                            Matrix<double>& C,
                            double alpha,
                            double betta,
                            char op);

    template <>
    void optimized_multiply(const CSR_Matrix<COMPLEX>& A,
                            const Matrix<COMPLEX>& B,
                            Matrix<COMPLEX>& C,
                            COMPLEX alpha,
                            COMPLEX betta,
                            char op);

    template <typename T>
    void optimized_add(const CSR_Matrix<T>& A, const CSR_Matrix<T>& B, CSR_Matrix<T>& C);

    template <>
    void optimized_add(const CSR_Matrix<double>& A, const CSR_Matrix<double>& B, CSR_Matrix<double>& C);

    template <>
    void optimized_add(const CSR_Matrix<COMPLEX>& A, const CSR_Matrix<COMPLEX>& B, CSR_Matrix<COMPLEX>& C);

    template <typename T>
    void optimized_add(const CSR_Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);

    template <>
    void optimized_add(const CSR_Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C);

    template <>
    void optimized_add(const CSR_Matrix<COMPLEX>& A, const Matrix<COMPLEX>& B, Matrix<COMPLEX>& C);

}  // namespace QComputations

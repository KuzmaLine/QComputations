#pragma once
#include <iostream>
#include <complex>
#include <vector>
#include "matrix.hpp"

#include <functional>
#include <iomanip>
#include "config.hpp"
#include "additional_operators.hpp"

#include <mkl_spblas.h>
#include <mkl_sparse_handle.h>

// rewrite to return results of optimized_multiply

namespace QComputations {

namespace {
    template <typename T>
    bool is_close(T a, T b) {
        return (std::abs(a - b) < QConfig::instance().eps());
    }

    ILP_TYPE get_index_value(ILP_TYPE row_index, const std::vector<ILP_TYPE>& ia) {
        ILP_TYPE index_value = 0;
        for (ILP_TYPE k = 0; k < row_index; k++) {
            index_value = ia[k + 1];
        }

        return index_value;
    }

    template <typename T>
    void insert_value_csr(ILP_TYPE i, ILP_TYPE j, T value,
                          std::vector<T>& values, std::vector<ILP_TYPE>& ja,
                          std::vector<ILP_TYPE>& ia) {
        auto index_value = get_index_value(i, ia);
        
        //std::cout << index_value << std::endl;
        if (index_value == values.size()) {
            values.emplace_back(value);
            ja.emplace_back(j);
        } else {
            for (ILP_TYPE k = 0; k < ia[i + 1] - ia[i]; k++) {
                if (ja[index_value] < j) {
                    index_value++;
                } else {
                    break;
                }
            }

            values.insert(std::next(values.begin(), index_value), value);
            ja.insert(std::next(ja.begin(), index_value), j);
        }

        //std::cout << "HERE\n";

        for (ILP_TYPE k = i + 1; k < ia.size(); k++) {
            ia[k]++;
        }
    }
}


// WRITE TO ALL TEMPLATE VERSIONS
template<typename T> 
class CSR_Matrix {
    public:
        explicit CSR_Matrix() = default;
        explicit CSR_Matrix(const Matrix<T>& A, T default_value = T(0));
        explicit CSR_Matrix(std::function<T(ILP_TYPE, ILP_TYPE)> func, T default_value = T(0));
        explicit CSR_Matrix(const CSR_Matrix<T>& A);
        explicit CSR_Matrix(const std::vector<T>& vals, const std::vector<ILP_TYPE>& ia, const std::vector<ILP_TYPE>& ja);
        explicit CSR_Matrix(const std::function<void(std::vector<ILP_TYPE>& ia, std::vector<ILP_TYPE>& ja, std::vector<T>& vals)>&, T default_value = T(0));


        ~CSR_Matrix() {
            mkl_sparse_destroy(mkl_matrix_);
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
    private:
        bool is_exported_ = true;
        ILP_TYPE n_;
        ILP_TYPE m_;
        T* vals_;
        MKL_INT* ia_;
        MKL_INT* ja_;
        sparse_matrix_t mkl_matrix_ = nullptr;
        // TEMP
        sparse_matrix_type_t matrix_type_ = SPARSE_MATRIX_TYPE_GENERAL;
};


// NAIVE VERSION for not double and complex double
template<typename T>
CSR_Matrix<T>::CSR_Matrix(const Matrix<T>& A, T default_value) : n_(A.n()), m_(A.m()), ia_(n_ + 1, 0), default_value_(default_value) {
    for (ILP_TYPE i = 0; i < n_; i++) {
        for (ILP_TYPE j = 0; j < m_; j++) {
            if (!is_close(A.elem(i, j), default_value)) {
                //std::cout << "NEW_ELEM: " << i << " " << j << " " << A.elem(i, j) << std::endl;
                //std::cout << vals_ << std::endl << ja_ << std::endl << ia_ << std::endl;
                insert_value_csr(i, j, A.elem(i, j), vals_, ja_, ia_);
            }
        }
    }
}

template<typename T>
CSR_Matrix<T>::CSR_Matrix(ILP_TYPE n, ILP_TYPE m, const std::vector<T>& vals, const std::vector<ILP_TYPE>& ia, const std::vector<ILP_TYPE>& ja): n_(n), m_(m), vals_(vals), ia_(ia), ja_(ja), default_value_(default_value) {}

template<typename T>
CSR_Matrix<T>::CSR_Matrix(const CSR_Matrix<T>& A): n_(A.n_), m_(A.m_), ia_(A.ia_), ja_(A.ja_), default_value_(A.default_value_) {}

template<typename T>
const T CSR_Matrix<T>::operator()(ILP_TYPE i, ILP_TYPE j) const {
    ILP_TYPE index_value = get_index_value(i, ia_);

    for (ILP_TYPE k = 0; k < ia_[i + 1] - ia_[i]; k++) {
        if (ja_[index_value] == j) { 
            return vals_[index_value];
        }

        index_value++;
    }

    return default_value_;
}

template<typename T>
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

template<typename T>
void optimized_multiply(const CSR_Matrix<T>& A, const CSR_Matrix<T>& B, CSR_Matrix<T>& C, char op = 'N');

template<>
void optimized_multiply(const CSR_Matrix<double>& A, const CSR_Matrix<double>& B, CSR_Matrix<double>& C, char op = 'N');

template<>
void optimized_multiply(const CSR_Matrix<COMPLEX>& A, const CSR_Matrix<COMPLEX>& B, CSR_Matrix<COMPLEX>& C, char op = 'N');

template<typename T>
void optimized_multiply(const CSR_Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C, T alpha = T(1), T betta = T(0), char op = 'N');

template<>
void optimized_multiply(const CSR_Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C, double alpha = 1, double betta = 0,  char op = 'N');

template<>
void optimized_multiply(const CSR_Matrix<COMPLEX>& A, const Matrix<COMPLEX>& B, Matrix<COMPLEX>& C, COMPLEX alpha = COMPLEX(1, 0), COMPLEX betta = COMPLEX(0, 0),  char op = 'N');

template<typename T>
void optimized_add(const CSR_Matrix<T>& A, const CSR_Matrix<T>& B, CSR_Matrix<T>& C);

template<>
void optimized_add(const CSR_Matrix<double>& A, const CSR_Matrix<double>& B, CSR_Matrix<double>& C);

template<>
void optimized_add(const CSR_Matrix<COMPLEX>& A, const CSR_Matrix<COMPLEX>& B, CSR_Matrix<COMPLEX>& C);

template<typename T>
void optimized_add(const CSR_Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);

template<>
void optimized_add(const CSR_Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C);

template<>
void optimized_add(const CSR_Matrix<COMPLEX>& A, const Matrix<COMPLEX>& B, Matrix<COMPLEX>& C);

} // namespace QComputations
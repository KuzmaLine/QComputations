#ifdef ENABLE_ONEAPI
#pragma once
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
        ILP_TYPE get_index_value(ILP_TYPE row_index, const MKL_INT* ia) { return ia[row_index]; }

        void to_mkl_sparse(ILP_TYPE n, ILP_TYPE m, ILP_TYPE* ia, ILP_TYPE* ja, double* data, sparse_matrix_t* A) {
            sparse_status_t status = mkl_sparse_d_create_csr(A, SPARSE_INDEX_BASE_ZERO, n, m, ia, ia + 1, ja, data);
            if (status != SPARSE_STATUS_SUCCESS) {
                throw std::runtime_error("to_mkl_sparse: failed to create MKL handle, status = " + std::to_string(status));
            }
        }

        void to_mkl_sparse(ILP_TYPE n, ILP_TYPE m, ILP_TYPE* ia, ILP_TYPE* ja, COMPLEX* data, sparse_matrix_t* A) {
            sparse_status_t status = mkl_sparse_z_create_csr(
                A, SPARSE_INDEX_BASE_ZERO, n, m, ia, ia + 1, ja, reinterpret_cast<MKL_Complex16*>(data));

            if (status != SPARSE_STATUS_SUCCESS) {
                throw std::runtime_error("to_mkl_sparse: failed to create MKL handle, status = " + std::to_string(status));
            }
        }

        sparse_operation_t get_sparse_operation(char op) {
            if (op == 'T') return SPARSE_OPERATION_TRANSPOSE;
            if (op == 'C') return SPARSE_OPERATION_CONJUGATE_TRANSPOSE;
            return SPARSE_OPERATION_NON_TRANSPOSE;
        }
    }  // namespace



    // WRITE TO ALL TEMPLATE VERSIONS
    template <typename T>
    class CSR_Matrix {
        public:
            explicit CSR_Matrix() = default;
            // explicit CSR_Matrix(const Matrix<T>& A, T default_value = T(0));
            CSR_Matrix(const CSR_Matrix& A)
                : n_(A.n_), m_(A.m_), nnz_(A.nnz_), default_value_(A.default_value_),
                owns_arrays_(true), matrix_type_(A.matrix_type_)
            {
                if (A.mkl_matrix_) {
                    sparse_matrix_t copy = nullptr;
                    struct matrix_descr descr;
                    descr.type = matrix_type_;
                    descr.mode = SPARSE_FILL_MODE_FULL;
                    descr.diag = SPARSE_DIAG_NON_UNIT;
                    sparse_status_t status = mkl_sparse_copy(A.mkl_matrix_, descr, &copy);
                    if (status != SPARSE_STATUS_SUCCESS) {
                        throw std::runtime_error("mkl_sparse_copy failed in copy constructor");
                    }

                    sparse_index_base_t indexing;
                    ILP_TYPE rows, cols;
                    ILP_TYPE *ia_begin, *ia_end, *ja;
                    if constexpr (std::is_same_v<T, double>) {
                        double* vals;
                        mkl_sparse_d_export_csr(copy, &indexing, &rows, &cols, &ia_begin, &ia_end, &ja, &vals);
                        vals_ = vals;
                    } else if constexpr (std::is_same_v<T, COMPLEX>) {
                        MKL_Complex16* vals;
                        mkl_sparse_z_export_csr(copy, &indexing, &rows, &cols, &ia_begin, &ia_end, &ja, &vals);
                        vals_ = reinterpret_cast<COMPLEX*>(vals);
                    } else {
                        throw std::logic_error("CSR_Matrix copy constructor only supports double and COMPLEX");
                    }
                    ia_ = ia_begin;
                    ja_ = ja;
                    mkl_matrix_ = copy;
                    owns_arrays_ = false;
                    nnz_ = ia_[n_];
                }
                else if (A.ia_ != nullptr) {
                    ia_ = new ILP_TYPE[n_ + 1];
                    std::copy(A.ia_, A.ia_ + n_ + 1, ia_);
                    if (nnz_ > 0) {
                        ja_ = new ILP_TYPE[nnz_];
                        std::copy(A.ja_, A.ja_ + nnz_, ja_);
                        vals_ = new T[nnz_];
                        std::copy(A.vals_, A.vals_ + nnz_, vals_);
                    } else {
                        ja_ = nullptr;
                        vals_ = nullptr;
                    }
                    to_mkl_sparse(n_, m_, ia_, ja_, vals_, &mkl_matrix_);
                    owns_arrays_ = true;
                }
                else {
                    ia_ = new ILP_TYPE[n_ + 1];
                    std::fill_n(ia_, n_ + 1, 0);
                    ja_ = nullptr;
                    vals_ = nullptr;
                    to_mkl_sparse(n_, m_, ia_, ja_, vals_, &mkl_matrix_);
                    owns_arrays_ = true;
                    nnz_ = 0;
                }
            }
            explicit CSR_Matrix(ILP_TYPE n, ILP_TYPE m, T default_value = T(0));
            explicit CSR_Matrix(ILP_TYPE n,
                                ILP_TYPE m,
                                const std::vector<T>& vals,
                                const std::vector<MKL_INT>& ia,
                                const std::vector<MKL_INT>& ja);
            explicit CSR_Matrix(
                const std::function<void(std::vector<MKL_INT>& ia, std::vector<MKL_INT>& ja, std::vector<T>& vals)>&,
                T default_value = T(0));
            explicit CSR_Matrix(ILP_TYPE n, ILP_TYPE m,
                    std::function<T(ILP_TYPE, ILP_TYPE)> func,
                    T default_value = T(0));

            ~CSR_Matrix() {
                if (mkl_matrix_) {
                    mkl_sparse_destroy(mkl_matrix_);
                }
                if (owns_arrays_) {
                    delete[] ia_;
                    delete[] ja_;
                    delete[] vals_;
                }
            }


            inline ILP_TYPE n() const { return n_; }
            inline ILP_TYPE m() const { return m_; }
            inline T* vals() { return vals_; }
            inline const T* vals() const { return vals_; }
            inline ILP_TYPE* ia() { return ia_; }
            inline const ILP_TYPE* ia() const { return ia_; }
            inline ILP_TYPE* ja() { return ja_; }
            inline const ILP_TYPE* ja() const { return ja_; }
            inline ILP_TYPE nnz() const { return nnz_; }


            CSR_Matrix(CSR_Matrix&& other) noexcept;
            explicit CSR_Matrix(sparse_matrix_t mkl_matrix);

            CSR_Matrix& operator=(const CSR_Matrix& other) {
                std::cerr << "CSR_Matrix copy constructor called for " << typeid(T).name() << std::endl;
                if (this != &other) {
                    // Освобождаем текущие ресурсы
                    if (mkl_matrix_) {
                        mkl_sparse_destroy(mkl_matrix_);
                    }
                    if (owns_arrays_) {
                        delete[] ia_;
                        delete[] ja_;
                        delete[] vals_;
                    }

                    // Копируем метаданные
                    n_ = other.n_;
                    m_ = other.m_;
                    nnz_ = other.nnz_;
                    default_value_ = other.default_value_;
                    matrix_type_ = other.matrix_type_;
                    owns_arrays_ = true;

                    if (other.mkl_matrix_) {
                        // Глубокое копирование через MKL
                        sparse_matrix_t copy = nullptr;
                        struct matrix_descr descr;
                        descr.type = matrix_type_;
                        descr.mode = SPARSE_FILL_MODE_FULL;
                        descr.diag = SPARSE_DIAG_NON_UNIT;
                        sparse_status_t status = mkl_sparse_copy(other.mkl_matrix_, descr, &copy);
                        if (status != SPARSE_STATUS_SUCCESS) {
                            throw std::runtime_error("mkl_sparse_copy failed in operator=");
                        }

                        sparse_index_base_t indexing;
                        ILP_TYPE rows, cols;
                        ILP_TYPE *ia_begin, *ia_end, *ja;
                        if constexpr (std::is_same_v<T, double>) {
                            double* vals;
                            mkl_sparse_d_export_csr(copy, &indexing, &rows, &cols, &ia_begin, &ia_end, &ja, &vals);
                            vals_ = vals;
                        } else if constexpr (std::is_same_v<T, COMPLEX>) {
                            MKL_Complex16* vals;
                            mkl_sparse_z_export_csr(copy, &indexing, &rows, &cols, &ia_begin, &ia_end, &ja, &vals);
                            vals_ = reinterpret_cast<COMPLEX*>(vals);
                        } else {
                            throw std::logic_error("operator= only supported for double and COMPLEX");
                        }
                        ia_ = ia_begin;
                        ja_ = ja;
                        mkl_matrix_ = copy;
                        owns_arrays_ = false;
                        nnz_ = ia_[n_];
                    }
                    else if (other.ia_ != nullptr) {
                        // Ручное копирование массивов
                        ia_ = new ILP_TYPE[n_ + 1];
                        std::copy(other.ia_, other.ia_ + n_ + 1, ia_);
                        ja_ = new ILP_TYPE[nnz_];
                        std::copy(other.ja_, other.ja_ + nnz_, ja_);
                        vals_ = new T[nnz_];
                        std::copy(other.vals_, other.vals_ + nnz_, vals_);
                        to_mkl_sparse(n_, m_, ia_, ja_, vals_, &mkl_matrix_);
                        owns_arrays_ = true;
                    }
                    else {
                        // Пустая матрица
                        ia_ = new ILP_TYPE[n_ + 1];
                        std::fill_n(ia_, n_ + 1, 0);
                        ja_ = nullptr;
                        vals_ = nullptr;
                        to_mkl_sparse(n_, m_, ia_, ja_, vals_, &mkl_matrix_);
                        owns_arrays_ = true;
                        nnz_ = 0;
                    }
                }
                return *this;
            }
            CSR_Matrix& operator=(CSR_Matrix&& other) noexcept;

            void sort_ja();

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
            bool owns_arrays_ = false;
            ILP_TYPE n_ = 0;
            ILP_TYPE m_ = 0;
            ILP_TYPE nnz_ = 0;
            T default_value_ = T(0);
            T* vals_ = nullptr;
            ILP_TYPE* ia_ = nullptr;
            ILP_TYPE* ja_ = nullptr;
            sparse_matrix_t mkl_matrix_ = nullptr;
            sparse_matrix_type_t matrix_type_ = SPARSE_MATRIX_TYPE_GENERAL;
    };

    // // NAIVE VERSION for not double and complex double
    // template <typename T>
    // CSR_Matrix<T>::CSR_Matrix(const Matrix<T>& A, T default_value)
    //     : n_(A.n()), m_(A.m()), default_value_(default_value) {
    //     std::vector<MKL_INT> ia_vec(n_ + 1, 0);
    //     std::vector<MKL_INT> ja_vec;
    //     std::vector<T> vals_vec;

    //     for (ILP_TYPE i = 0; i < n_; i++) {
    //         ia_vec[i + 1] = ia_vec[i];
    //         for (ILP_TYPE j = 0; j < m_; j++) {
    //             if (!is_close(A.elem(i, j), default_value)) {
    //                 ia_vec[i + 1]++;
    //                 vals_vec.emplace_back(A.elem(i, j));
    //                 ja_vec.emplace_back(j);
    //             }
    //         }
    //     }

    //     ia_ = std::make_unique<MKL_INT[]>(n_ + 1);
    //     std::copy(ia_vec.begin(), ia_vec.end(), ia_.get());

    //     if (!ja_vec.empty()) {
    //         ja_ = std::make_unique<MKL_INT[]>(ja_vec.size());
    //         std::copy(ja_vec.begin(), ja_vec.end(), ja_.get());

    //         vals_ = std::make_unique<T[]>(vals_vec.size());
    //         std::copy(vals_vec.begin(), vals_vec.end(), vals_.get());
    //     }
    // }

    template <typename T>
    CSR_Matrix<T>::CSR_Matrix(ILP_TYPE n, ILP_TYPE m, T default_value)
        : n_(n), m_(m), default_value_(default_value), owns_arrays_(true)
    {
        ia_ = new ILP_TYPE[n_ + 1];
        std::fill_n(ia_, n_ + 1, 0);

        ja_ = nullptr;
        vals_ = nullptr;
     
        to_mkl_sparse(n_, m_, ia_, ja_, vals_, &mkl_matrix_);
    }

    template <typename T>
    CSR_Matrix<T>::CSR_Matrix(ILP_TYPE n, ILP_TYPE m,
                            std::function<T(ILP_TYPE, ILP_TYPE)> func,
                            T default_value)
        : n_(n), m_(m), default_value_(default_value), owns_arrays_(true)
    {
        std::vector<ILP_TYPE> ia_vec(n_ + 1, 0);
        std::vector<ILP_TYPE> ja_vec;
        std::vector<T> vals_vec;

        ja_vec.reserve(n_ * m_ / 10);
        vals_vec.reserve(n_ * m_ / 10);

        for (ILP_TYPE i = 0; i < n_; ++i) {
            ia_vec[i] = vals_vec.size();
            for (ILP_TYPE j = 0; j < m_; ++j) {
                T val = func(i, j);
                if (!is_close(val, default_value_)) {
                    vals_vec.push_back(val);
                    ja_vec.push_back(j);
                }
            }
        }
        ia_vec[n_] = vals_vec.size();

        ia_ = new ILP_TYPE[n_ + 1];
        std::copy(ia_vec.begin(), ia_vec.end(), ia_);

        nnz_ = vals_vec.size();
        if (nnz_ > 0) {
            ja_ = new ILP_TYPE[nnz_];
            std::copy(ja_vec.begin(), ja_vec.end(), ja_);

            vals_ = new T[nnz_];
            std::copy(vals_vec.begin(), vals_vec.end(), vals_);
        } else {
            ja_ = nullptr;
            vals_ = nullptr;
        }

        to_mkl_sparse(n_, m_, ia_, ja_, vals_, &mkl_matrix_);
    }

    template <typename T>
    CSR_Matrix<T>::CSR_Matrix(ILP_TYPE n,
                            ILP_TYPE m,
                            const std::vector<T>& vals,
                            const std::vector<ILP_TYPE>& ia,
                            const std::vector<ILP_TYPE>& ja)
        : n_(n), m_(m), default_value_(T(0)), owns_arrays_(true)
    {
        assert(ia.size() == static_cast<size_t>(n_ + 1));
        nnz_ = static_cast<ILP_TYPE>(vals.size());
        assert(ja.size() == static_cast<size_t>(nnz_));

        std::vector<ILP_TYPE> sorted_ia = ia;
        std::vector<ILP_TYPE> sorted_ja = ja;
        std::vector<T> sorted_vals = vals;

        ia_ = new ILP_TYPE[n_ + 1];
        std::copy(sorted_ia.begin(), sorted_ia.end(), ia_);

        if (nnz_ > 0) {
            ja_ = new ILP_TYPE[nnz_];
            std::copy(sorted_ja.begin(), sorted_ja.end(), ja_);

            vals_ = new T[nnz_];
            std::copy(sorted_vals.begin(), sorted_vals.end(), vals_);
        } else {
            ja_ = nullptr;
            vals_ = nullptr;
        }

        to_mkl_sparse(n_, m_, ia_, ja_, vals_, &mkl_matrix_);
        if (!mkl_matrix_) {
            throw std::runtime_error("CSR_Matrix(vals, ia, ja): MKL handle is nullptr after creation");
        }
    }

    template <typename T>
    CSR_Matrix<T>::CSR_Matrix(CSR_Matrix<T>&& other) noexcept
        : n_(std::exchange(other.n_, 0))
        , m_(std::exchange(other.m_, 0))
        , default_value_(std::exchange(other.default_value_, T(0)))
        , vals_(std::exchange(other.vals_, nullptr))
        , ia_(std::exchange(other.ia_, nullptr))
        , ja_(std::exchange(other.ja_, nullptr))
        , mkl_matrix_(std::exchange(other.mkl_matrix_, nullptr))
        , matrix_type_(std::exchange(other.matrix_type_, SPARSE_MATRIX_TYPE_GENERAL))
        , owns_arrays_(std::exchange(other.owns_arrays_, false))
        , nnz_(std::exchange(other.nnz_, 0))
    {}

    template <typename T>
    CSR_Matrix<T>& CSR_Matrix<T>::operator=(CSR_Matrix<T>&& other) noexcept {
        if (this != &other) {
            if (mkl_matrix_) {
                mkl_sparse_destroy(mkl_matrix_);
            }
            if (owns_arrays_) {
                delete[] ia_;
                delete[] ja_;
                delete[] vals_;
            }

            n_ = other.n_;
            m_ = other.m_;
            default_value_ = other.default_value_;
            vals_ = other.vals_;
            ia_ = other.ia_;
            ja_ = other.ja_;
            mkl_matrix_ = other.mkl_matrix_;
            matrix_type_ = other.matrix_type_;
            owns_arrays_ = other.owns_arrays_;
            nnz_ = other.nnz_;

            other.n_ = other.m_ = 0;
            other.default_value_ = T(0);
            other.vals_ = nullptr;
            other.ia_ = nullptr;
            other.ja_ = nullptr;
            other.mkl_matrix_ = nullptr;
            other.owns_arrays_ = false;
            other.matrix_type_ = SPARSE_MATRIX_TYPE_GENERAL;
            other.nnz_ = 0;
        }
        return *this;
    }

    // template <typename T>
    // CSR_Matrix<T>::CSR_Matrix(const CSR_Matrix<T>& A) : n_(A.n_), m_(A.m_), default_value_(A.default_value_) {
    //     ia_ = std::make_unique<MKL_INT[]>(n_ + 1);
    //     std::copy(A.ia_.get(), A.ia_.get() + n_ + 1, ia_.get());

    //     size_t nnz = A.ia_[n_];
    //     if (nnz > 0) {
    //         ja_ = std::make_unique<MKL_INT[]>(nnz);
    //         std::copy(A.ja_.get(), A.ja_.get() + nnz, ja_.get());

    //         vals_ = std::make_unique<T[]>(nnz);
    //         std::copy(A.vals_.get(), A.vals_.get() + nnz, vals_.get());
    //     }
    // }

    template <typename T>
    void CSR_Matrix<T>::sort_ja() {
        if (mkl_matrix_) {
            sparse_status_t status = mkl_sparse_order(mkl_matrix_);
            if (status != SPARSE_STATUS_SUCCESS) {
                std::cerr << "mkl_sparse_order failed with status: " << status << std::endl;
            }
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

    template<typename T>
    CSR_Matrix<T> sparse_spmm(const CSR_Matrix<T>& A, const CSR_Matrix<T>& B, char op) {
        if (!A.mkl_matrix() || !B.mkl_matrix())
            throw std::runtime_error("sparse_spmm: input matrix has no MKL handle");
        sparse_matrix_t C = nullptr;
        sparse_status_t st = mkl_sparse_spmm(get_sparse_operation(op), A.mkl_matrix(), B.mkl_matrix(), &C);
        if (st != SPARSE_STATUS_SUCCESS)
            throw std::runtime_error("mkl_sparse_spmm failed");
        return CSR_Matrix<T>(C);
    }

    void sparse_syrd(const CSR_Matrix<COMPLEX>& A, const Matrix<COMPLEX>& B, Matrix<COMPLEX>& C, COMPLEX alpha, COMPLEX betta, char op = 'N');

    void sparse_add(const sparse_matrix_t A, const sparse_matrix_t B, sparse_matrix_t* C, double alpha, char op = 'N');
    void sparse_add(const sparse_matrix_t A, const sparse_matrix_t B, sparse_matrix_t* C, COMPLEX alpha, char op = 'N');

    template<typename T>
    CSR_Matrix<T> sparse_add(const CSR_Matrix<T>& A, const CSR_Matrix<T>& B, T alpha, char op = 'N');

    template<>
    CSR_Matrix<COMPLEX> sparse_add(const CSR_Matrix<COMPLEX>& A, const CSR_Matrix<COMPLEX>& B, COMPLEX alpha, char op);

    template <typename T>
    void optimized_multiply(const CSR_Matrix<T>& A, const CSR_Matrix<T>& B, CSR_Matrix<T>& C, T alpha, T betta, char op = 'N');

    template <>
    void optimized_multiply(const CSR_Matrix<double>& A, const CSR_Matrix<double>& B, CSR_Matrix<double>& C, double alpha, double betta, char op);

    template <>
    void optimized_multiply(const CSR_Matrix<COMPLEX>& A,
                            const CSR_Matrix<COMPLEX>& B,
                            CSR_Matrix<COMPLEX>& C,
                            COMPLEX alpha,
                            COMPLEX betta,
                            char op);

    template <typename T>
    void optimized_multiply(const Matrix<T>& A,
                            const CSR_Matrix<T>& B,
                            Matrix<T>& C,
                            T alpha = T(1),
                            T betta = T(0),
                            char op = 'N');
    
    template <>
    void optimized_multiply(const Matrix<COMPLEX>& A,
                            const CSR_Matrix<COMPLEX>& B,
                            Matrix<COMPLEX>& C,
                            COMPLEX alpha,
                            COMPLEX betta,
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
#endif
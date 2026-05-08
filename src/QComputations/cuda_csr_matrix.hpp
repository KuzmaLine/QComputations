#ifdef __CUDACC__
#pragma once

#include <complex>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>
#include <cusparse.h>

#include "additional_operators.hpp"
#include "config.hpp"
#include "cuda_matrix.hpp"
#include "functions.hpp"

namespace QComputations {
    namespace {
        using IndexType = int;


    }  // namespace



    // WRITE TO ALL TEMPLATE VERSIONS
template <typename T, typename GPU_T = typename std::conditional_t<
    std::is_same<T, COMPLEX>::value, cuDoubleComplex,
    typename std::conditional_t<std::is_same<T, double>::value, double, void>>
>
    class CUDA_CSR_Matrix {
        public:
            CUDA_CSR_Matrix() = delete;
            CUDA_CSR_Matrix(cusparseHandle_t handle) : handle_(handle), n_(0), m_(0), nnz_(0), descr_(nullptr) {}
            CUDA_CSR_Matrix(cusparseHandle_t handle, IndexType rows, IndexType cols, IndexType nnz_count,
                            const std::vector<IndexType>& row_offsets,
                            const std::vector<IndexType>& col_indices,
                            const std::vector<T>& values) : handle_(handle), n_(rows), m_(cols), nnz_(nnz_count) {
                cudaMalloc(&d_row_offsets_, (rows + 1) * sizeof(IndexType));
                cudaMalloc(&d_col_indices_, nnz_ * sizeof(IndexType));
                cudaMalloc(&d_values_, nnz_ * sizeof(GPU_T));

                cudaMemcpy(d_row_offsets_, row_offsets.data(), (rows + 1) * sizeof(IndexType), cudaMemcpyHostToDevice);
                cudaMemcpy(d_col_indices_, col_indices.data(), nnz_ * sizeof(IndexType), cudaMemcpyHostToDevice);
                cudaMemcpy(d_values_, values.data(), nnz_ * sizeof(GPU_T), cudaMemcpyHostToDevice);

                if constexpr (std::is_same_v<T, double>) {
                    cusparseCreateCsr(&descr_, rows, cols, nnz_,
                                    d_row_offsets_, d_col_indices_, d_values_,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
                } else { 
                    cusparseCreateCsr(&descr_, rows, cols, nnz_,
                                    d_row_offsets_, d_col_indices_, d_values_,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F);
                }
            }

            CUDA_CSR_Matrix(cusparseHandle_t handle, IndexType n, IndexType m,
                            std::function<T(IndexType, IndexType)> func);

            // CUDA_CSR_Matrix(cusparseHandle_t handle, cusparseSpMatDescr_t descr, IndexType rows, IndexType cols, IndexType nnz)
            //     : handle_(handle), n_(rows), m_(cols), nnz_(nnz), descr_(descr) {}

            CUDA_CSR_Matrix(cusparseHandle_t handle, IndexType rows, IndexType cols)
                : handle_(handle), n_(rows), m_(cols), nnz_(0) {
                // Выбираем тип значений во время компиляции
                constexpr cudaDataType valueType = std::is_same_v<T, COMPLEX> 
                                                ? CUDA_C_64F 
                                                : CUDA_R_64F;
                cudaMalloc((void**)&d_row_offsets_, (rows + 1) * sizeof(IndexType));
                cudaMemset(d_row_offsets_, 0, (rows + 1) * sizeof(IndexType));
                cusparseCreateCsr(&descr_, rows, cols, 0,
                                d_row_offsets_, nullptr, nullptr,
                                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                CUSPARSE_INDEX_BASE_ZERO, valueType);
            }

            ~CUDA_CSR_Matrix() {
                if (descr_) cusparseDestroySpMat(descr_);
                if (d_row_offsets_) cudaFree(d_row_offsets_);
                if (d_col_indices_) cudaFree(d_col_indices_);
                if (d_values_) cudaFree(d_values_);
            }

            CUDA_CSR_Matrix(const CUDA_CSR_Matrix&) = delete;
            CUDA_CSR_Matrix& operator=(const CUDA_CSR_Matrix&) = delete;

            CUDA_CSR_Matrix(CUDA_CSR_Matrix&& other) noexcept
                : handle_(other.handle_), n_(other.n_), m_(other.m_), nnz_(other.nnz_),
                  d_row_offsets_(other.d_row_offsets_), d_col_indices_(other.d_col_indices_),
                  d_values_(other.d_values_), descr_(other.descr_) {
                other.d_row_offsets_ = nullptr;
                other.d_col_indices_ = nullptr;
                other.d_values_ = nullptr;
                other.descr_ = nullptr;
            }

            CUDA_CSR_Matrix& operator=(CUDA_CSR_Matrix&& other) noexcept {
                if (this != &other) {
                    handle_ = other.handle_;
                    if (descr_) cusparseDestroySpMat(descr_);
                    if (d_row_offsets_) cudaFree(d_row_offsets_);
                    if (d_col_indices_) cudaFree(d_col_indices_);
                    if (d_values_) cudaFree(d_values_);
                    m_ = other.m_; n_ = other.n_; nnz_ = other.nnz_;
                    descr_ = other.descr_;
                    d_col_indices_ = other.d_col_indices_;
                    d_row_offsets_ = other.d_row_offsets_;
                    d_values_ = other.d_values_;
                    other.d_row_offsets_ = nullptr;
                    other.d_col_indices_ = nullptr;
                    other.d_values_ = nullptr;
                    other.descr_ = nullptr;
                }
                return *this;
            }

            cusparseHandle_t handle() const { return handle_; }
            IndexType n() const { return n_; }
            IndexType m() const { return m_; }
            IndexType nnz() const { return nnz_; }
            void set_nnz(IndexType nnz) { nnz_ = nnz; }
            IndexType* d_ia() { return d_row_offsets_; }
            IndexType* d_ja() { return d_col_indices_; }
            GPU_T* vals() { return d_values_; }
            cusparseSpMatDescr_t& descr() { return descr_; }
            cusparseSpMatDescr_t descr() const { return descr_;}

            void show(size_t width = QConfig::instance().width());
            void show_data(size_t width = QConfig::instance().width());

            void sort_ja();

        private:
            cusparseHandle_t handle_;
            IndexType n_, m_, nnz_;
            IndexType *d_row_offsets_ = nullptr, *d_col_indices_ = nullptr;
            GPU_T* d_values_ = nullptr;
            cusparseSpMatDescr_t descr_;
    };

template <typename T, typename GPU_T>
CUDA_CSR_Matrix<T, GPU_T>::CUDA_CSR_Matrix(cusparseHandle_t handle, IndexType rows, IndexType cols,
                std::function<T(IndexType, IndexType)> func)
    : handle_(handle), n_(rows), m_(cols) {
    std::vector<IndexType> row_off(rows + 1, 0);
    std::vector<IndexType> col_ind;
    std::vector<T> vals;

    for (IndexType i = 0; i < rows; ++i) {
        row_off[i] = static_cast<IndexType>(col_ind.size());
        for (IndexType j = 0; j < cols; ++j) {
            T val = func(i, j);
            if (val != T(0)) {               // работаем напрямую с результатом func
                col_ind.push_back(j);
                vals.push_back(val);
            }
        }
    }
    row_off[rows] = static_cast<IndexType>(col_ind.size());
    nnz_ = static_cast<IndexType>(vals.size());

    constexpr cudaDataType valueType = std::is_same_v<T, COMPLEX>
                                       ? CUDA_C_64F : CUDA_R_64F;

    cudaMalloc((void**)&d_row_offsets_, (rows + 1) * sizeof(IndexType));
    cudaMalloc((void**)&d_col_indices_, nnz_ * sizeof(IndexType));
    cudaMalloc((void**)&d_values_, nnz_ * sizeof(T));

    cudaMemcpy(d_row_offsets_, row_off.data(), (rows + 1) * sizeof(IndexType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices_, col_ind.data(), nnz_ * sizeof(IndexType), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values_, vals.data(), nnz_ * sizeof(T), cudaMemcpyHostToDevice);

    cusparseCreateCsr(&descr_, rows, cols, nnz_,
                      d_row_offsets_, d_col_indices_, d_values_,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, valueType);
}

// template <typename T, typename GPU_T>
// void CUDA_CSR_Matrix<T, GPU_T>::sort_ja() {
//     if (!descr_ || nnz_ == 0) return;

//     int64_t rows, cols, nnz;
//     void *d_offsets = nullptr, *d_cols = nullptr, *d_vals = nullptr;
//     cusparseIndexType_t off_type, col_type;
//     cusparseIndexBase_t idx_base;
//     cudaDataType val_type;
//     CUSPARSESC(cusparseCsrGet(descr_, &rows, &cols, &nnz,
//                               &d_offsets, &d_cols, &d_vals,
//                               &off_type, &col_type, &idx_base, &val_type));

//     // Временный дескриптор с теми же offsets, но без col/val (они будут перезаписаны)
//     cusparseSpMatDescr_t sorted_descr;
//     CUSPARSESC(cusparseCreateCsr(&sorted_descr, rows, cols, 0,
//                                  (int*)d_offsets, nullptr, nullptr,
//                                  off_type, col_type, idx_base, val_type));

//     size_t bufferSize = 0;
//     CUSPARSESC(cusparseCsrSort_bufferSize(handle_, rows, cols, nnz,
//                                           d_offsets, d_cols, d_vals,
//                                           &bufferSize));
//     void *dBuffer;
//     cudaMalloc(&dBuffer, bufferSize);

//     // Сортировка на месте: указатели col/val после вызова будут указывать на упорядоченные массивы
//     CUSPARSESC(cusparseCsrSort(handle_, rows, cols, nnz, d_offsets, d_cols, d_vals,
//                                sorted_descr, dBuffer));

//     // Удаляем старый дескриптор (память не освобождается, так как она управляется пользователем)
//     cusparseDestroySpMat(descr_);
//     descr_ = sorted_descr;

//     cudaFree(dBuffer);
// }

template <typename T, typename GPU_T>
    void CUDA_CSR_Matrix<T, GPU_T>::show(size_t width) {
        int64_t m64 = n_, n64 = m_, nnz64 = nnz_;
        void *d_row_off = nullptr, *d_col_ind = nullptr, *d_vals = nullptr;
        cusparseIndexType_t rowOffType, colIndType;
        cusparseIndexBase_t idxBase;
        cudaDataType valueType;
        cusparseCsrGet(descr_, &m64, &n64, &nnz64,
                    &d_row_off, &d_col_ind, &d_vals,
                    &rowOffType, &colIndType, &idxBase, &valueType);

        // Копируем на хост
        std::vector<int> row_off(n_ + 1);
        std::vector<int> col_ind(nnz_);
        std::vector<T> vals(nnz_);

        cudaMemcpy(row_off.data(), d_row_off, (n_ + 1) * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(col_ind.data(), d_col_ind, nnz_ * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(vals.data(), d_vals, nnz_ * sizeof(T), cudaMemcpyDeviceToHost);

        // Строим плотную матрицу для вывода
        std::vector<std::string> dense(m_ * n_);
        for (int i = 0; i < n_; ++i) {
            for (int j = row_off[i]; j < row_off[i + 1]; ++j) {
                std::ostringstream oss;
                if constexpr (std::is_same_v<T, COMPLEX>) {
                    oss << "(" << vals[j].real() << "," << vals[j].imag() << ")";
                } else {
                    oss << vals[j];
                }
                dense[i * m_ + col_ind[j]] = oss.str();
            }
        }
        std::cout << "Sparse matrix (" << n_ << " x " << m_ << "):\n";
        for (int i = 0; i < n_; ++i) {
            for (int j = 0; j < m_; ++j) {
                if (dense[i * m_ + j].empty()) {
                    std::cout << std::setw(width) << "0";
                } else {
                    std::cout << std::setw(width) << dense[i * m_ + j];
                }
                std::cout << " ";
            }
            std::cout << "\n";
        }
    }

    template <typename T, typename GPU_T>
    void CUDA_CSR_Matrix<T, GPU_T>::show_data(size_t width) {
        int64_t m64 = n_, n64 = m_, nnz64 = nnz_;
        void *d_row_off = nullptr, *d_col_ind = nullptr, *d_vals = nullptr;
        cusparseIndexType_t rowOffType, colIndType;
        cusparseIndexBase_t idxBase;
        cudaDataType valueType;
        cusparseCsrGet(descr_, &m64, &n64, &nnz64,
                    &d_row_off, &d_col_ind, &d_vals,
                    &rowOffType, &colIndType, &idxBase, &valueType);

        // Копируем на хост
        std::vector<int> row_off(n_ + 1);
        std::vector<int> col_ind(nnz_);
        std::vector<T> vals(nnz_);

        cudaMemcpy(row_off.data(), d_row_off, (n_ + 1) * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(col_ind.data(), d_col_ind, nnz_ * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(vals.data(), d_vals, nnz_ * sizeof(T), cudaMemcpyDeviceToHost);

        std::cout << n_ << " " << m_ << " " << nnz_ << std::endl;

        for (size_t i = 0; i < row_off.size(); i++) {
            std::cout << row_off[i] << " ";
        }

        std::cout << std::endl;

        for (size_t i = 0; i < col_ind.size(); i++) {
            std::cout << col_ind[i] << " ";
        }

        std::cout << std::endl;

        for (size_t i = 0; i < vals.size(); i++) {
            std::cout << vals[i] << " ";
        }

        std::cout << std::endl;
    }

// ---------------------------------------------- FUNCTIONS -----------------------------------------------------


template <typename T, typename GPU_T = typename std::conditional_t<
    std::is_same<T, COMPLEX>::value, cuDoubleComplex,
    typename std::conditional_t<std::is_same<T, double>::value, double, void>>
>
    void optimized_multiply(const CUDA_CSR_Matrix<T>& A, const CUDA_CSR_Matrix<T>& B, CUDA_CSR_Matrix<T>& C, GPU_T alpha, GPU_T betta, char opA = 'N', char opB = 'N');

    // template <>
    // void optimized_multiply(const CUDA_CSR_Matrix<double>& A, const CUDA_CSR_Matrix<double>& B, CUDA_CSR_Matrix<double>& C, double alpha, double betta, char opA, char opB);

    template <>
    void optimized_multiply(const CUDA_CSR_Matrix<COMPLEX>& A,
                            const CUDA_CSR_Matrix<COMPLEX>& B,
                            CUDA_CSR_Matrix<COMPLEX>& C,
                            cuDoubleComplex alpha,
                            cuDoubleComplex betta,
                            char opA, char opB);

template <typename T, typename GPU_T = typename std::conditional_t<
    std::is_same<T, COMPLEX>::value, cuDoubleComplex,
    typename std::conditional_t<std::is_same<T, double>::value, double, void>>
>
    void optimized_multiply(const CUDA_Matrix<T>& A,
                            const CUDA_CSR_Matrix<T>& B,
                            CUDA_Matrix<T>& C,
                            GPU_T alpha,
                            GPU_T betta,
                            char op = 'N');
    
    template <>
    void optimized_multiply(const CUDA_Matrix<COMPLEX>& A,
                            const CUDA_CSR_Matrix<COMPLEX>& B,
                            CUDA_Matrix<COMPLEX>& C,
                            cuDoubleComplex alpha,
                            cuDoubleComplex betta,
                            char op);

template <typename T, typename GPU_T = typename std::conditional_t<
    std::is_same<T, COMPLEX>::value, cuDoubleComplex,
    typename std::conditional_t<std::is_same<T, double>::value, double, void>>
>
    void optimized_multiply(const CUDA_CSR_Matrix<T>& A,
                            const CUDA_Matrix<T>& B,
                            CUDA_Matrix<T>& C,
                            GPU_T alpha,
                            GPU_T betta,
                            char op = 'N');

    // template <>
    // void optimized_multiply(const CUDA_CSR_Matrix<double>& A,
    //                         const CUDA_Matrix<double>& B,
    //                         CUDA_Matrix<double>& C,
    //                         double alpha,
    //                         double betta,
    //                         char op);

    template <>
    void optimized_multiply(const CUDA_CSR_Matrix<COMPLEX>& A,
                            const CUDA_Matrix<COMPLEX>& B,
                            CUDA_Matrix<COMPLEX>& C,
                            cuDoubleComplex alpha,
                            cuDoubleComplex betta,
                            char op);

    // template <typename T>
    // void optimized_add(const CSR_Matrix<T>& A, const CSR_Matrix<T>& B, CSR_Matrix<T>& C);

    // template <>
    // void optimized_add(const CSR_Matrix<double>& A, const CSR_Matrix<double>& B, CSR_Matrix<double>& C);

    // template <>
    // void optimized_add(const CSR_Matrix<COMPLEX>& A, const CSR_Matrix<COMPLEX>& B, CSR_Matrix<COMPLEX>& C);

    // template <typename T>
    // void optimized_add(const CSR_Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);

    // template <>
    // void optimized_add(const CSR_Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C);

    // template <>
    // void optimized_add(const CSR_Matrix<COMPLEX>& A, const Matrix<COMPLEX>& B, Matrix<COMPLEX>& C);

}  // namespace QComputations
#endif
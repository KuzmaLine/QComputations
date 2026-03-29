#include "csr_matrix.hpp"

namespace QComputations {

namespace {
    void to_mkl_sparse(ILP_TYPE n, ILP_TYPE m, ILP_TYPE* ia, ILP_TYPE* ja, double* data, sparse_matrix_t* A) {
        mkl_sparse_d_create_csr(A, SPARSE_INDEX_BASE_ZERO, n, m, ia, ia + 1, ja, data);
    }

    void to_mkl_sparse(ILP_TYPE n, ILP_TYPE m, ILP_TYPE* ia, ILP_TYPE* ja, COMPLEX* data, sparse_matrix_t* A) {
        mkl_sparse_z_create_csr(A, SPARSE_INDEX_BASE_ZERO, n, m, ia, ia + 1, ja, data);
    }

    template<typename T>
    void to_csr(const Matrix<T>& A, std::vector<ILP_TYPE>& ia, std::vector<ILP_TYPE>& ja, std::vector<T>& vals) {
        for (ILP_TYPE i = 0; i < n_; i++) {
            for (ILP_TYPE j = 0; j < m_; j++) {
                auto num = A.elem(i, j);
                if (!is_close(A.elem(i, j), T(0))) {
                    ia_[i + 1]++;
                    vals_.emplace_back(num);
                    ja_.emplace_back(j);
                }
            }
        }
    }
}

template<>
CSR_Matrix<double>::CSR_Matrix(ILP_TYPE n, ILP_TYPE m, const std::vector<double>& vals, const std::vector<ILP_TYPE>& ia, const std::vector<ILP_TYPE>& ja): n_(n), m_(m) {
    to_mkl_sparse(n_, m_, ia_, ja_, vals_ &mkl_matrix_);
}

template<>
CSR_Matrix<COMPLEX>::CSR_Matrix(ILP_TYPE n, ILP_TYPE m, const std::vector<COMPLEX>& vals, const std::vector<ILP_TYPE>& ia, const std::vector<ILP_TYPE>& ja): n_(n), m_(m), vals_(vals), ia_(ia), ja_(ja), default_value_(default_value) {
    to_mkl_sparse(n_, m_, ia_.data(), ja_.data(), vals_.data(), &mkl_matrix_);
}

template<>
CSR_Matrix<double>::CSR_Matrix(const Matrix<double>& A, double default_value) : n_(A.n()), m_(A.m()), ia_(n_ + 1, 0), default_value_(default_value) {
    to_csr(A, ia_, ja_, vals_);
    to_mkl_sparse(n_, m_, ia_.data(), ja_.data(), vals_.data(), &mkl_matrix_);
}

template<>
CSR_Matrix<COMPLEX>::CSR_Matrix(const Matrix<COMPLEX>& A, double default_value) : n_(A.n()), m_(A.m()), ia_(n_ + 1, 0), default_value_(default_value) {
    to_csr(A, ia_, ja_, vals_);
    to_mkl_sparse(n_, m_, ia_.data(), ja_.data(), vals_.data(), &mkl_matrix_);
}

inline void sparse_spmm(const sparse_matrix_t A, const sparse_matrix_t B, sparse_matrix_t* C, char op) {
    sparse_operation_t tmp = QConfig::instance().sparse_operation_t(op);
    mkl_sparse_spmm(tmp, A, B, C);
}

void sparse_add(const sparse_matrix_t A, const sparse_matrix_t B, sparse_matrix_t* C, double alpha, char op) {
    sparse_operation_t tmp = QConfig::instance().sparse_operation_t(op);
    mkl_sparse_d_add(tmp, A, alpha, B, C);
}

void sparse_add(const sparse_matrix_t A, const sparse_matrix_t B, sparse_matrix_t* C, COMPLEX alpha, char op) {
    sparse_operation_t tmp = QConfig::instance().sparse_operation_t(op);
    mkl_sparse_z_add(tmp, A, alpha, B, C);
}

// // copy code
// template<>
// void optimized_multiply(const CSR_Matrix<double>& A, const CSR_Matrix<double>& B, CSR_Matrix<double>& C, char op) {
//     if (C.mkl_matrix_ != nullptr) {
//         mkl_sparse_destroy(C.mkl_matrix_);
//     }
//     sparse_spmm(A.mkl_matrix_, B.mkl_matrix_, &C.mkl_matrix_, op);

//     sparse_index_base_t indexing;
//     MKL_INT rows, cols;
//     MKL_INT* rows_start = nullptr;
//     MKL_INT* rows_end = nullptr;
//     MKL_INT* col_indx = nullptr;
//     double* values = nullptr;
//     mkl_sparse_d_export_csr(C_mkl, &indexing, &rows, &cols, &rows_start, &rows_end, &col_indx, &values);

//     C.n_ = rows;
//     C.m_ = cols;
//     C.ia_.assign(rows_start, rows_start + rows + 1);
//     C.ja_.assign(col_indx, col_indx + rows_end[rows-1] - indexing);
//     C.vals_.assign(values, values + (rows_end[rows-1] - indexing));

//     mkl_sparse_d_create_csr(&C.mkl_matrix_, indexing, rows, cols,
//                             C.ia_.data(), C.ia_.data() + 1,
//                             C.ja_.data(), C.vals_.data());
// }

// // copy code
// template<>
// void optimized_multiply(const CSR_Matrix<COMPLEX>& A, const CSR_Matrix<COMPLEX>& B, CSR_Matrix<COMPLEX>& C, char op) {
//     if (C.mkl_matrix_ != nullptr) {
//         mkl_sparse_destroy(C.mkl_matrix_);
//     }
//     sparse_spmm(A.mkl_matrix_, B.mkl_matrix_, &C.mkl_matrix_, op);

//     sparse_index_base_t indexing;
//     MKL_INT rows, cols;
//     MKL_INT* rows_start = nullptr;
//     MKL_INT* rows_end = nullptr;
//     MKL_INT* col_indx = nullptr;
//     double* values = nullptr;
//     mkl_sparse_d_export_csr(C_mkl, &indexing, &rows, &cols, &rows_start, &rows_end, &col_indx, &values);

//     C.n_ = rows;
//     C.m_ = cols;
//     C.ia_.assign(rows_start, rows_start + rows + 1);
//     C.ja_.assign(col_indx, col_indx + rows_end[rows-1] - indexing);
//     C.vals_.assign(values, values + (rows_end[rows-1] - indexing));

//     mkl_sparse_d_create_csr(&C.mkl_matrix_, indexing, rows, cols,
//                             C.ia_.data(), C.ia_.data() + 1,
//                             C.ja_.data(), C.vals_.data());
// }

template<>
void optimized_multiply(const CSR_Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C, double alpha, double betta, char op) {
    sparse_operation_t tmp = QConfig::instance().sparse_operation_t(op);
    sparse_layout_t layout = QConfig::instance().sparse_layout(B.matrix_style());
    mkl_sparse_d_mm(tmp, alpha, A.mkl_matrix_, descr, layout, B.data(), B.ld(), betta, C.data(), C.ld());
}

template<>
void optimized_multiply(const CSR_Matrix<COMPLEX>& A, const Matrix<COMPLEX>& B, Matrix<COMPLEX>& C, COMPLEX alpha, COMPLEX betta, char op) {
    sparse_operation_t tmp = QConfig::instance().sparse_operation_t(op);
    sparse_layout_t layout = QConfig::instance().sparse_layout(B.matrix_style());
    mkl_sparse_z_mm(tmp, alpha, A.mkl_matrix_, descr, layout, B.data(), B.ld(), betta, C.data(), C.ld());
}

} // namespace QComputations

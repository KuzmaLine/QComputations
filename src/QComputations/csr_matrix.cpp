#include "csr_matrix.hpp"

namespace QComputations {

    namespace {
        void to_mkl_sparse(ILP_TYPE n, ILP_TYPE m, MKL_INT* ia, MKL_INT* ja, double* data, sparse_matrix_t* A) {
            mkl_sparse_d_create_csr(A, SPARSE_INDEX_BASE_ZERO, n, m, ia, ia + 1, ja, data);
        }

        void to_mkl_sparse(ILP_TYPE n, ILP_TYPE m, MKL_INT* ia, MKL_INT* ja, COMPLEX* data, sparse_matrix_t* A) {
            mkl_sparse_z_create_csr(
                A, SPARSE_INDEX_BASE_ZERO, n, m, ia, ia + 1, ja, reinterpret_cast<MKL_Complex16*>(data));
        }

        template <typename T>
        void to_csr(const Matrix<T>& A, std::vector<MKL_INT>& ia, std::vector<MKL_INT>& ja, std::vector<T>& vals) {
            ia.assign(A.n() + 1, 0);
            for (ILP_TYPE i = 0; i < A.n(); i++) {
                ia[i + 1] = ia[i];
                for (ILP_TYPE j = 0; j < A.m(); j++) {
                    auto num = A.elem(i, j);
                    if (!is_close(num, T(0))) {
                        ia[i + 1]++;
                        vals.emplace_back(num);
                        ja.emplace_back(j);
                    }
                }
            }
        }

        sparse_operation_t get_sparse_operation(char op) {
            if (op == 'T') return SPARSE_OPERATION_TRANSPOSE;
            if (op == 'C') return SPARSE_OPERATION_CONJUGATE_TRANSPOSE;
            return SPARSE_OPERATION_NON_TRANSPOSE;
        }
    }  // namespace

    template <>
    CSR_Matrix<double>::CSR_Matrix(ILP_TYPE n,
                                   ILP_TYPE m,
                                   const std::vector<double>& vals,
                                   const std::vector<MKL_INT>& ia,
                                   const std::vector<MKL_INT>& ja)
        : n_(n), m_(m) {
        ia_ = std::make_unique<MKL_INT[]>(n_ + 1);
        std::copy(ia.begin(), ia.end(), ia_.get());

        if (!ja.empty()) {
            ja_ = std::make_unique<MKL_INT[]>(ja.size());
            std::copy(ja.begin(), ja.end(), ja_.get());

            vals_ = std::make_unique<double[]>(vals.size());
            std::copy(vals.begin(), vals.end(), vals_.get());
        }
        to_mkl_sparse(n_, m_, ia_.get(), ja_.get(), vals_.get(), &mkl_matrix_);
    }

    template <>
    CSR_Matrix<COMPLEX>::CSR_Matrix(ILP_TYPE n,
                                    ILP_TYPE m,
                                    const std::vector<COMPLEX>& vals,
                                    const std::vector<MKL_INT>& ia,
                                    const std::vector<MKL_INT>& ja)
        : n_(n), m_(m) {
        ia_ = std::make_unique<MKL_INT[]>(n_ + 1);
        std::copy(ia.begin(), ia.end(), ia_.get());

        if (!ja.empty()) {
            ja_ = std::make_unique<MKL_INT[]>(ja.size());
            std::copy(ja.begin(), ja.end(), ja_.get());

            vals_ = std::make_unique<COMPLEX[]>(vals.size());
            std::copy(vals.begin(), vals.end(), vals_.get());
        }
        to_mkl_sparse(n_, m_, ia_.get(), ja_.get(), vals_.get(), &mkl_matrix_);
    }

    template <>
    CSR_Matrix<double>::CSR_Matrix(const Matrix<double>& A, double default_value)
        : n_(A.n()), m_(A.m()), default_value_(default_value) {
        std::vector<MKL_INT> ia_vec;
        std::vector<MKL_INT> ja_vec;
        std::vector<double> vals_vec;
        to_csr(A, ia_vec, ja_vec, vals_vec);

        ia_ = std::make_unique<MKL_INT[]>(n_ + 1);
        std::copy(ia_vec.begin(), ia_vec.end(), ia_.get());

        if (!ja_vec.empty()) {
            ja_ = std::make_unique<MKL_INT[]>(ja_vec.size());
            std::copy(ja_vec.begin(), ja_vec.end(), ja_.get());

            vals_ = std::make_unique<double[]>(vals_vec.size());
            std::copy(vals_vec.begin(), vals_vec.end(), vals_.get());
        }
        to_mkl_sparse(n_, m_, ia_.get(), ja_.get(), vals_.get(), &mkl_matrix_);
    }

    template <>
    CSR_Matrix<COMPLEX>::CSR_Matrix(const Matrix<COMPLEX>& A, COMPLEX default_value)
        : n_(A.n()), m_(A.m()), default_value_(default_value) {
        std::vector<MKL_INT> ia_vec;
        std::vector<MKL_INT> ja_vec;
        std::vector<COMPLEX> vals_vec;
        to_csr(A, ia_vec, ja_vec, vals_vec);

        ia_ = std::make_unique<MKL_INT[]>(n_ + 1);
        std::copy(ia_vec.begin(), ia_vec.end(), ia_.get());

        if (!ja_vec.empty()) {
            ja_ = std::make_unique<MKL_INT[]>(ja_vec.size());
            std::copy(ja_vec.begin(), ja_vec.end(), ja_.get());

            vals_ = std::make_unique<COMPLEX[]>(vals_vec.size());
            std::copy(vals_vec.begin(), vals_vec.end(), vals_.get());
        }
        to_mkl_sparse(n_, m_, ia_.get(), ja_.get(), vals_.get(), &mkl_matrix_);
    }

    inline void sparse_spmm(const sparse_matrix_t A, const sparse_matrix_t B, sparse_matrix_t* C, char op) {
        sparse_operation_t tmp = get_sparse_operation(op);
        mkl_sparse_spmm(tmp, A, B, C);
    }

    void sparse_add(const sparse_matrix_t A, const sparse_matrix_t B, sparse_matrix_t* C, double alpha, char op) {
        sparse_operation_t tmp = get_sparse_operation(op);
        mkl_sparse_d_add(tmp, A, alpha, B, C);
    }

    void sparse_add(const sparse_matrix_t A, const sparse_matrix_t B, sparse_matrix_t* C, COMPLEX alpha, char op) {
        sparse_operation_t tmp = get_sparse_operation(op);
        MKL_Complex16 mkl_alpha = {alpha.real(), alpha.imag()};
        mkl_sparse_z_add(tmp, A, mkl_alpha, B, C);
    }

    template <>
    void optimized_multiply(const CSR_Matrix<double>& A,
                            const Matrix<double>& B,
                            Matrix<double>& C,
                            double alpha,
                            double betta,
                            char op) {
        sparse_operation_t tmp = get_sparse_operation(op);
        sparse_layout_t layout = (B.matrix_style() == C_STYLE) ? SPARSE_LAYOUT_ROW_MAJOR : SPARSE_LAYOUT_COLUMN_MAJOR;
        struct matrix_descr descr;
        descr.type = SPARSE_MATRIX_TYPE_GENERAL;
        mkl_sparse_d_mm(tmp,
                        alpha,
                        A.mkl_matrix(),
                        descr,
                        layout,
                        const_cast<double*>(B.data()),
                        B.m(),
                        B.LD(),
                        betta,
                        C.data(),
                        C.LD());
    }

    template <>
    void optimized_multiply(const CSR_Matrix<COMPLEX>& A,
                            const Matrix<COMPLEX>& B,
                            Matrix<COMPLEX>& C,
                            COMPLEX alpha,
                            COMPLEX betta,
                            char op) {
        sparse_operation_t tmp = get_sparse_operation(op);
        sparse_layout_t layout = (B.matrix_style() == C_STYLE) ? SPARSE_LAYOUT_ROW_MAJOR : SPARSE_LAYOUT_COLUMN_MAJOR;
        struct matrix_descr descr;
        descr.type = SPARSE_MATRIX_TYPE_GENERAL;
        MKL_Complex16 mkl_alpha = {alpha.real(), alpha.imag()};
        MKL_Complex16 mkl_betta = {betta.real(), betta.imag()};
        mkl_sparse_z_mm(tmp,
                        mkl_alpha,
                        A.mkl_matrix(),
                        descr,
                        layout,
                        reinterpret_cast<MKL_Complex16*>(const_cast<COMPLEX*>(B.data())),
                        B.m(),
                        B.LD(),
                        mkl_betta,
                        reinterpret_cast<MKL_Complex16*>(C.data()),
                        C.LD());
    }

}  // namespace QComputations

#include "csr_matrix.hpp"

namespace QComputations {

    namespace {
        template <typename T>
        void to_csr(const Matrix<T>& A, std::vector<ILP_TYPE>& ia, std::vector<ILP_TYPE>& ja, std::vector<T>& vals) {
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
    }  // namespace

    // template <>
    // CSR_Matrix<double>::CSR_Matrix(ILP_TYPE n,
    //                                ILP_TYPE m,
    //                                const std::vector<double>& vals,
    //                                const std::vector<MKL_INT>& ia,
    //                                const std::vector<MKL_INT>& ja)
    //     : n_(n), m_(m) {
    //     ia_ = std::make_unique<MKL_INT[]>(n_ + 1);
    //     std::copy(ia.begin(), ia.end(), ia_.get());

    //     if (!ja.empty()) {
    //         ja_ = std::make_unique<MKL_INT[]>(ja.size());
    //         std::copy(ja.begin(), ja.end(), ja_.get());

    //         vals_ = std::make_unique<double[]>(vals.size());
    //         std::copy(vals.begin(), vals.end(), vals_.get());
    //     }
    //     to_mkl_sparse(n_, m_, ia_.get(), ja_.get(), vals_.get(), &mkl_matrix_);
    // }

    template <>
    CSR_Matrix<double>::CSR_Matrix(const CSR_Matrix<double>& A)
        : n_(A.n_), m_(A.m_), default_value_(A.default_value_) {
        if (A.mkl_matrix_) {
            sparse_matrix_t copy = nullptr;
            struct matrix_descr descr;
            descr.type = SPARSE_MATRIX_TYPE_GENERAL;
            descr.mode = SPARSE_FILL_MODE_FULL;
            descr.diag = SPARSE_DIAG_NON_UNIT;
            mkl_sparse_copy(A.mkl_matrix_, descr, &copy);

            sparse_index_base_t indexing;
            ILP_TYPE rows, cols;
            ILP_TYPE *ia_begin, *ia_end, *ja;
            double* vals;
            mkl_sparse_d_export_csr(copy, &indexing, &rows, &cols, &ia_begin, &ia_end, &ja, &vals);
            vals_ = vals;
            ia_ = ia_begin;
            ja_ = ja;
            mkl_matrix_ = copy;
            owns_arrays_ = false;
        }
    }

    template <>
    CSR_Matrix<COMPLEX>::CSR_Matrix(const CSR_Matrix<COMPLEX>& A)
        : n_(A.n_), m_(A.m_), default_value_(A.default_value_) {
        if (A.mkl_matrix_) {
            sparse_matrix_t copy = nullptr;
            struct matrix_descr descr;
            descr.type = SPARSE_MATRIX_TYPE_GENERAL;
            descr.mode = SPARSE_FILL_MODE_FULL;
            descr.diag = SPARSE_DIAG_NON_UNIT;
            mkl_sparse_copy(A.mkl_matrix_, descr, &copy);

            sparse_index_base_t indexing;
            ILP_TYPE rows, cols;
            ILP_TYPE *ia_begin, *ia_end, *ja;
            MKL_Complex16* vals;
            mkl_sparse_z_export_csr(copy, &indexing, &rows, &cols, &ia_begin, &ia_end, &ja, &vals);
            vals_ = reinterpret_cast<COMPLEX*>(vals);
            ia_ = ia_begin;
            ja_ = ja;
            mkl_matrix_ = copy;
            owns_arrays_ = false;
        }
    }

    // template <>
    // CSR_Matrix<COMPLEX>::CSR_Matrix(ILP_TYPE n,
    //                                 ILP_TYPE m,
    //                                 const std::vector<COMPLEX>& vals,
    //                                 const std::vector<ILP_TYPE>& ia,
    //                                 const std::vector<ILP_TYPE>& ja)
    //     : n_(n), m_(m) {
    //     ia_ = std::make_unique<ILP_TYPE[]>(n_ + 1);
    //     std::copy(ia.begin(), ia.end(), ia_.get());

    //     if (!ja.empty()) {
    //         ja_ = std::make_unique<ILP_TYPE[]>(ja.size());
    //         std::copy(ja.begin(), ja.end(), ja_.get());

    //         vals_ = std::make_unique<COMPLEX[]>(vals.size());
    //         std::copy(vals.begin(), vals.end(), vals_.get());
    //     }
    //     to_mkl_sparse(n_, m_, ia_.get(), ja_.get(), vals_.get(), &mkl_matrix_);
    // }

    template <>
    CSR_Matrix<double>::CSR_Matrix(sparse_matrix_t mkl_matrix) {
        sparse_index_base_t indexing;
        ILP_TYPE rows, cols;
        ILP_TYPE *ia_begin, *ia_end, *ja;
        double *vals;

        mkl_sparse_d_export_csr(mkl_matrix, &indexing, &rows, &cols,
                                &ia_begin, &ia_end, &ja, &vals);

        n_ = rows;
        m_ = cols;
        ia_ = ia_begin;
        ja_ = ja;
        vals_ = vals;
        mkl_matrix_ = mkl_matrix;
        owns_arrays_ = false;
    }

    template <>
    CSR_Matrix<COMPLEX>::CSR_Matrix(sparse_matrix_t mkl_matrix) {
        sparse_index_base_t indexing;
        ILP_TYPE rows, cols;
        ILP_TYPE *ia_begin, *ia_end, *ja;
        MKL_Complex16 *vals;

        mkl_sparse_z_export_csr(mkl_matrix, &indexing, &rows, &cols,
                                &ia_begin, &ia_end, &ja, &vals);

        n_ = rows;
        m_ = cols;
        ia_ = ia_begin;
        ja_ = ja;
        vals_ = reinterpret_cast<COMPLEX*>(vals);
        mkl_matrix_ = mkl_matrix;
        owns_arrays_ = false;
    }

    // template <>
    // CSR_Matrix<double>::CSR_Matrix(const Matrix<double>& A, double default_value)
    //     : n_(A.n()), m_(A.m()), default_value_(default_value) {
    //     std::vector<MKL_INT> ia_vec;
    //     std::vector<MKL_INT> ja_vec;
    //     std::vector<double> vals_vec;
    //     to_csr(A, ia_vec, ja_vec, vals_vec);

    //     ia_ = std::make_unique<MKL_INT[]>(n_ + 1);
    //     std::copy(ia_vec.begin(), ia_vec.end(), ia_.get());

    //     if (!ja_vec.empty()) {
    //         ja_ = std::make_unique<MKL_INT[]>(ja_vec.size());
    //         std::copy(ja_vec.begin(), ja_vec.end(), ja_.get());

    //         vals_ = std::make_unique<double[]>(vals_vec.size());
    //         std::copy(vals_vec.begin(), vals_vec.end(), vals_.get());
    //     }
    //     to_mkl_sparse(n_, m_, ia_.get(), ja_.get(), vals_.get(), &mkl_matrix_);
    // }

    // template <>
    // CSR_Matrix<COMPLEX>::CSR_Matrix(const Matrix<COMPLEX>& A, COMPLEX default_value)
    //     : n_(A.n()), m_(A.m()), default_value_(default_value) {
    //     std::vector<MKL_INT> ia_vec;
    //     std::vector<MKL_INT> ja_vec;
    //     std::vector<COMPLEX> vals_vec;
    //     to_csr(A, ia_vec, ja_vec, vals_vec);

    //     ia_ = std::make_unique<MKL_INT[]>(n_ + 1);
    //     std::copy(ia_vec.begin(), ia_vec.end(), ia_.get());

    //     if (!ja_vec.empty()) {
    //         ja_ = std::make_unique<MKL_INT[]>(ja_vec.size());
    //         std::copy(ja_vec.begin(), ja_vec.end(), ja_.get());

    //         vals_ = std::make_unique<COMPLEX[]>(vals_vec.size());
    //         std::copy(vals_vec.begin(), vals_vec.end(), vals_.get());
    //     }
    //     to_mkl_sparse(n_, m_, ia_.get(), ja_.get(), vals_.get(), &mkl_matrix_);
    // }

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

    template<>
    CSR_Matrix<COMPLEX> sparse_add(const CSR_Matrix<COMPLEX>& A, const CSR_Matrix<COMPLEX>& B, COMPLEX alpha, char op) {
        sparse_operation_t tmp = get_sparse_operation(op);
        sparse_matrix_t C_mkl = nullptr;
        MKL_Complex16 mkl_alpha = {alpha.real(), alpha.imag()};
        sparse_status_t status = mkl_sparse_z_add(tmp, A.mkl_matrix(), mkl_alpha, B.mkl_matrix(), &C_mkl);
        if (status != SPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("mkl_sparse_z_add failed in sparse_add");
        }
        return CSR_Matrix<COMPLEX>(C_mkl);
    }

    void sparse_syrd(const CSR_Matrix<COMPLEX>& A,
                    const Matrix<COMPLEX>& B,
                    Matrix<COMPLEX>& C,
                    COMPLEX alpha,
                    COMPLEX betta) {
        assert(A.n() == B.n());
        assert(B.n() == B.m());
        assert(C.n() == A.m() && C.m() == A.m());

        struct matrix_descr descrB;
        descrB.type = SPARSE_MATRIX_TYPE_HERMITIAN;
        descrB.mode = SPARSE_FILL_MODE_UPPER;
        descrB.diag = SPARSE_DIAG_NON_UNIT;

        sparse_layout_t layout = (B.matrix_style() == C_STYLE)
                                    ? SPARSE_LAYOUT_ROW_MAJOR
                                    : SPARSE_LAYOUT_COLUMN_MAJOR;

        COMPLEX mkl_alpha = {alpha.real(), alpha.imag()};
        COMPLEX mkl_beta  = {betta.real(), betta.imag()};

        struct matrix_descr descrC;
        descrC.type = SPARSE_MATRIX_TYPE_HERMITIAN;
        descrC.mode = SPARSE_FILL_MODE_UPPER;
        descrC.diag = SPARSE_DIAG_NON_UNIT;

        // C = alpha * A * B * A^H + beta * C
        sparse_status_t status = mkl_sparse_z_syprd(
            get_sparse_operation(op), 
            A.mkl_matrix(),
            descrB,
            layout,
            reinterpret_cast<const MKL_Complex16*>(B.data()),
            B.LD(),
            mkl_alpha,
            mkl_beta,
            descrC,
            reinterpret_cast<MKL_Complex16*>(C.data()),
            C.LD()
        );

        if (status != SPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("mkl_sparse_z_syprd failed with status " +
                                    std::to_string(status));
        }

        char matrix_storage = (C.matrix_style() == C_STYLE) ? 'R' : 'C';
        mkl_zomatcopy(
            matrix_storage,
            'C',
            C.n(),
            C.m(),
            COMPLEX(1.0, 0),
            reinterpret_cast<const MKL_Complex16*>(C.data()),
            C.LD(),
            reinterpret_cast<MKL_Complex16*>(C.data()),
            C.LD()
        );
    }

    template<>
    void optimized_multiply(const CSR_Matrix<COMPLEX>& A,
                            const CSR_Matrix<COMPLEX>& B,
                            CSR_Matrix<COMPLEX>& C,
                            COMPLEX alpha,
                            COMPLEX betta,
                            char op) {
        // 1. Вычисляем temp = op(A) * B   (sparse_spmm применяет op только к A)
        CSR_Matrix<COMPLEX> temp = sparse_spmm(A, B, op);

        // 2. Масштабируем temp на alpha
        if (alpha != COMPLEX(1.0, 0.0)) {
            ILP_TYPE nnz = temp.nnz();
            #pragma omp parallel for
            for (ILP_TYPE i = 0; i < nnz; ++i) {
                temp.vals()[i] *= alpha;
            }
        }

        // 3. Если betta == 0, результат — просто temp
        if (betta == COMPLEX(0.0, 0.0)) {
            C = std::move(temp);
            return;
        }

        // 4. betta != 0: C = temp + betta * C
        C = sparse_add(temp, C, betta, 'N');
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

#ifdef __CUDACC__

#include "cuda_csr_matrix.hpp"

namespace QComputations {

    template <>
    void optimized_multiply(const CUDA_CSR_Matrix<COMPLEX>& A,
                            const CUDA_CSR_Matrix<COMPLEX>& B,
                            CUDA_CSR_Matrix<COMPLEX>& C,
                            cuDoubleComplex alpha, cuDoubleComplex betta,
                            char opA, char opB) {
        cusparseHandle_t handle = A.handle();
        auto opA_cusparse = get_sparse_operation(opA);
        auto opB_cusparse = get_sparse_operation(opB);

        int64_t C_rows = A.n(), C_cols = B.m();


        // 1. Извлекаем старые указатели из дескриптора C
        int64_t old_rows, old_cols, old_nnz;
        void *d_old_offsets = nullptr, *d_old_cols = nullptr, *d_old_vals = nullptr;
        cusparseIndexType_t off_type, col_type;
        cusparseIndexBase_t idx_base;
        cudaDataType val_type;
        CUSPARSESC(cusparseCsrGet(C.descr(), &old_rows, &old_cols, &old_nnz,
                                &d_old_offsets, &d_old_cols, &d_old_vals,
                                &off_type, &col_type, &idx_base, &val_type));

        // 2. Обычная процедура SpGEMM (как в документации)
        cusparseSpGEMMDescr_t spgemmDesc;
        CUSPARSESC(cusparseSpGEMM_createDescr(&spgemmDesc));

        size_t bufferSize1 = 0;
        CUSPARSESC(cusparseSpGEMM_workEstimation(handle, opA_cusparse, opB_cusparse,
                                                &alpha, A.descr(), B.descr(), &betta,
                                                C.descr(), CUDA_C_64F, CUSPARSE_SPGEMM_DEFAULT,
                                                spgemmDesc, &bufferSize1, nullptr));
        void* dBuffer1;
        cudaMalloc(&dBuffer1, bufferSize1);

        CUSPARSESC(cusparseSpGEMM_workEstimation(handle, opA_cusparse, opB_cusparse,
                                                &alpha, A.descr(), B.descr(), &betta,
                                                C.descr(), CUDA_C_64F, CUSPARSE_SPGEMM_DEFAULT,
                                                spgemmDesc, &bufferSize1, dBuffer1));

        size_t bufferSize2 = 0;
        CUSPARSESC(cusparseSpGEMM_compute(handle, opA_cusparse, opB_cusparse,
                                        &alpha, A.descr(), B.descr(), &betta,
                                        C.descr(), CUDA_C_64F, CUSPARSE_SPGEMM_DEFAULT,
                                        spgemmDesc, &bufferSize2, nullptr));
        void* dBuffer2;
        cudaMalloc(&dBuffer2, bufferSize2);

        CUSPARSESC(cusparseSpGEMM_compute(handle, opA_cusparse, opB_cusparse,
                                        &alpha, A.descr(), B.descr(), &betta,
                                        C.descr(), CUDA_C_64F, CUSPARSE_SPGEMM_DEFAULT,
                                        spgemmDesc, &bufferSize2, dBuffer2));

        int64_t new_nnz;
        CUSPARSESC(cusparseSpMatGetSize(C.descr(), &C_rows, &C_cols, &new_nnz));

        // 3. Освобождаем старые массивы колонок и значений (offsets сохраняется)
        if (d_old_cols) cudaFree(d_old_cols);
        if (d_old_vals) cudaFree(d_old_vals);

        // 4. Выделяем новые col_indices и values
        int* d_new_cols = nullptr;
        cuDoubleComplex* d_new_vals = nullptr;
        if (new_nnz > 0) {
            cudaMalloc((void**)&d_new_cols, new_nnz * sizeof(int));
            cudaMalloc((void**)&d_new_vals, new_nnz * sizeof(cuDoubleComplex));
        }

        // 5. Устанавливаем новые указатели в дескриптор
        CUSPARSESC(cusparseCsrSetPointers(C.descr(), (int*)d_old_offsets,
                                        d_new_cols, d_new_vals));

        // 6. Копируем результат
        CUSPARSESC(cusparseSpGEMM_copy(handle, opA_cusparse, opB_cusparse,
                                    &alpha, A.descr(), B.descr(), &betta,
                                    C.descr(), CUDA_C_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc));

        // 7. Очистка временных буферов
        cudaFree(dBuffer1);
        cudaFree(dBuffer2);
        CUSPARSESC(cusparseSpGEMM_destroyDescr(spgemmDesc));

        C.set_nnz(static_cast<int>(new_nnz));
    }

    // ====================== sparse-dense (COMPLEX) ======================
    template <>
    void optimized_multiply(const CUDA_CSR_Matrix<COMPLEX>& A,
                            const CUDA_Matrix<COMPLEX>& B,
                            CUDA_Matrix<COMPLEX>& C,
                            cuDoubleComplex alpha, cuDoubleComplex betta,
                            char op) {                             // <-- один параметр op
        int A_rows = A.n(), A_cols = A.m();
        int B_rows = B.n(), B_cols = B.m();

        auto transA = get_cublas_operation(op);   // операция для разреженной A
        auto transB = CUBLAS_OP_N;                // плотная B не транспонируется

        int K = (transB == CUBLAS_OP_N) ? B_rows : B_cols;
        if (A_cols != K) throw std::invalid_argument("Inner dimensions mismatch");

        int C_rows = A_rows;
        int C_cols = (transB == CUBLAS_OP_N) ? B_cols : B_rows;

        if (C.n() != C_rows || C.m() != C_cols) {
            C = std::move(CUDA_Matrix<COMPLEX>(B.handle(), C_rows, C_cols));
        }

        cusparseHandle_t spHandle = A.handle();

        cusparseDnMatDescr_t B_dn, C_dn;
        cusparseCreateDnMat(&B_dn, B_rows, B_cols, B.ld(), (void*)B.data(),
                            CUDA_C_64F, CUSPARSE_ORDER_COL);
        cusparseCreateDnMat(&C_dn, C_rows, C_cols, C.ld(), (void*)C.data(),
                            CUDA_C_64F, CUSPARSE_ORDER_COL);

        auto spTransA = get_sparse_operation(op);
        auto spTransB = CUSPARSE_OPERATION_NON_TRANSPOSE;

        size_t bufferSize;
        cusparseSpMM_bufferSize(spHandle, spTransA, spTransB,
                                &alpha, A.descr(), B_dn, &betta, C_dn,
                                CUDA_C_64F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);
        void* dBuffer;
        cudaMalloc(&dBuffer, bufferSize);
        cusparseSpMM(spHandle, spTransA, spTransB,
                    &alpha, A.descr(), B_dn, &betta, C_dn,
                    CUDA_C_64F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);

        cudaFree(dBuffer);
        cusparseDestroyDnMat(B_dn);
        cusparseDestroyDnMat(C_dn);
    }

    // ====================== dense-sparse (COMPLEX) ======================
    template <>
    void optimized_multiply(const CUDA_Matrix<COMPLEX>& A,
                            const CUDA_CSR_Matrix<COMPLEX>& B,
                            CUDA_Matrix<COMPLEX>& C,
                            cuDoubleComplex alpha, cuDoubleComplex betta,
                            char op) {                             // <-- один параметр op
        int A_rows = A.n(), A_cols = A.m(); // column-major
        int B_rows = B.n(), B_cols = B.m();

        if (A_cols != B_rows) throw std::invalid_argument("Inner dimensions mismatch");

        int C_rows = A_rows;
        int C_cols = B_cols;

        if (C.n() != C_rows || C.m() != C_cols) {
            C = std::move(CUDA_Matrix<COMPLEX>(A.handle(), C_rows, C_cols));
        }

        cusparseHandle_t spHandle = B.handle();
        cublasHandle_t blasHandle = A.handle();

        // op применяется к разрежённой B: C = A * op(B)
        // Трюк: C^T = (op(B))^T * A^T
        // spOp для B в SpMM должно быть противоположно op, чтобы получить (op(B))^T
        cusparseOperation_t spOpB;
        if (op == 'N') spOpB = CUSPARSE_OPERATION_TRANSPOSE;
        else if (op == 'T') spOpB = CUSPARSE_OPERATION_NON_TRANSPOSE;
        else if (op == 'C') spOpB = CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE; // если нужно
        else throw std::invalid_argument("Unsupported op for dense-sparse");

        CUDA_Matrix<COMPLEX> Ct(blasHandle, B_cols, A_rows);
        cusparseDnMatDescr_t A_t_dn, Ct_dn;
        cusparseCreateDnMat(&A_t_dn, A_cols, A_rows, A.ld(), (void*)A.data(),
                            CUDA_C_64F, CUSPARSE_ORDER_COL);   // A^T (column-major, размеры переставлены)
        cusparseCreateDnMat(&Ct_dn, B_cols, A_rows, Ct.ld(), (void*)Ct.data(),
                            CUDA_C_64F, CUSPARSE_ORDER_COL);

        size_t bufferSize;
        cusparseSpMM_bufferSize(spHandle, spOpB, CUSPARSE_OPERATION_TRANSPOSE,
                                &alpha, B.descr(), A_t_dn, &betta, Ct_dn,
                                CUDA_C_64F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize);
        void* dBuffer;
        cudaMalloc(&dBuffer, bufferSize);
        cusparseSpMM(spHandle, spOpB, CUSPARSE_OPERATION_TRANSPOSE,
                    &alpha, B.descr(), A_t_dn, &betta, Ct_dn,
                    CUDA_C_64F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);
        cudaFree(dBuffer);
        cusparseDestroyDnMat(A_t_dn);
        cusparseDestroyDnMat(Ct_dn);

        // Транспонируем Ct -> C
        cuDoubleComplex cone = make_cuDoubleComplex(1.0, 0.0);
        cuDoubleComplex czero = make_cuDoubleComplex(0.0, 0.0);
        cublasZgeam(blasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                    A_rows, B_cols, &cone, Ct.data(), Ct.ld(),
                    &czero, nullptr, C.ld(), C.data(), C.ld());
    }

}

#endif
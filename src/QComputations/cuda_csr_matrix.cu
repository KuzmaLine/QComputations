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
        cudaStream_t stream = 0;

        static std::unordered_map<cusparseHandle_t, std::pair<void*, size_t>> buf1_cache, buf2_cache;

        auto get_cached_buffer = [&](auto& cache, size_t required) {
            auto& [ptr, size] = cache[handle];
            if (ptr && size < required) {
                std::cout << "HERE1 " << required << std::endl;
                cudaFree(ptr);
                ptr = nullptr;
                size = 0;
            }
            if (!ptr) {
                cudaMalloc(&ptr, required);
                size = required;
            }
            return ptr;
        };

        if (opA != 'N') {
            bool is_conjugate = (opA == 'C');

            A.get_transposed(is_conjugate);
        }

        if (opB != 'N') {
            bool is_conjugate = (opB == 'C');

            B.get_transposed(is_conjugate);
        }


        int64_t C_rows = A.n(), C_cols = B.m();
        // if (C.n() != C_rows || C.m() != C_cols) {
        //     // В графе нельзя создавать новые объекты хоста, поэтому изменение C вне графа
        //     // Вынесем за граф: если размеры не совпадают, пересоздаём C и перезахватываем
        //     C = CUDA_CSR_Matrix<COMPLEX>(handle, C_rows, C_cols);
        //     cudaStreamEndCapture(stream, nullptr); // отменить захват
        //     // Повторный вызов (с новым C)
        //     optimized_multiply(A, B, C, alpha, betta, opA, opB);
        //     return;
        // }

        // Извлечь старые указатели (до захвата)
        int64_t old_rows, old_cols, old_nnz;
        void *d_old_offsets = nullptr, *d_old_cols = nullptr, *d_old_vals = nullptr;
        cusparseIndexType_t off_type, col_type;
        cusparseIndexBase_t idx_base;
        cudaDataType val_type;
        CUSPARSESC(cusparseCsrGet(C.descr(), &old_rows, &old_cols, &old_nnz,
                                &d_old_offsets, &d_old_cols, &d_old_vals,
                                &off_type, &col_type, &idx_base, &val_type));

        cusparseSpGEMMDescr_t spgemmDesc;
        CUSPARSESC(cusparseSpGEMM_createDescr(&spgemmDesc));

        size_t bufferSize1 = 0;
        CUSPARSESC(cusparseSpGEMM_workEstimation(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                &alpha, A.descr(opA), B.descr(opB), &betta,
                                                C.descr(), CUDA_C_64F, CUSPARSE_SPGEMM_ALG1,
                                                spgemmDesc, &bufferSize1, nullptr));
        void* dBuffer1 = get_cached_buffer(buf1_cache, bufferSize1);

        CUSPARSESC(cusparseSpGEMM_workEstimation(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                &alpha, A.descr(opA), B.descr(opB), &betta,
                                                C.descr(), CUDA_C_64F, CUSPARSE_SPGEMM_ALG1,
                                                spgemmDesc, &bufferSize1, dBuffer1));

        size_t bufferSize2 = 0;
        CUSPARSESC(cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, A.descr(opA), B.descr(opB), &betta,
                                        C.descr(), CUDA_C_64F, CUSPARSE_SPGEMM_ALG1,
                                        spgemmDesc, &bufferSize2, nullptr));
        void* dBuffer2 = get_cached_buffer(buf2_cache, bufferSize2);

        CUSPARSESC(cusparseSpGEMM_compute(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, A.descr(opA), B.descr(opB), &betta,
                                        C.descr(), CUDA_C_64F, CUSPARSE_SPGEMM_ALG1,
                                        spgemmDesc, &bufferSize2, dBuffer2));

        int64_t new_nnz;
        CUSPARSESC(cusparseSpMatGetSize(C.descr(), &C_rows, &C_cols, &new_nnz));

        int *d_new_cols = nullptr;
        cuDoubleComplex *d_new_vals = nullptr;

        if (new_nnz == 0) {
            if (d_old_cols) cudaFree(d_old_cols);
            if (d_old_vals) cudaFree(d_old_vals);
        } else if (d_old_cols && d_old_vals && old_nnz >= new_nnz) {
            d_new_cols = (int*)d_old_cols;
            d_new_vals = (cuDoubleComplex*)d_old_vals;
        } else {
            if (d_old_cols) cudaFree(d_old_cols);
            if (d_old_vals) cudaFree(d_old_vals);
            cudaMalloc((void**)&d_new_cols, new_nnz * sizeof(int));
            cudaMalloc((void**)&d_new_vals, new_nnz * sizeof(cuDoubleComplex));
        }

        CUSPARSESC(cusparseCsrSetPointers(C.descr(), (int*)d_old_offsets,
                                        d_new_cols, d_new_vals));

        CUSPARSESC(cusparseSpGEMM_copy(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, A.descr(opA), B.descr(opB), &betta,
                                    C.descr(), CUDA_C_64F, CUSPARSE_SPGEMM_ALG1, spgemmDesc));

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
        cudaStream_t stream = 0;
        cusparseHandle_t handle = A.handle();
        static std::unordered_map<cusparseHandle_t, std::pair<void*, size_t>> bufcache;
        auto get_cached_buffer = [&](auto& cache, size_t required) {
            auto& [ptr, size] = cache[handle];
            if (ptr && size < required) {
                std::cout << "HERE2 " << required << std::endl;
                cudaFree(ptr);
                ptr = nullptr;
                size = 0;
            }
            if (!ptr) {
                cudaMalloc(&ptr, required);
                size = required;
            }
            return ptr;
        };
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
                                CUDA_C_64F, CUSPARSE_SPMM_CSR_ALG1, &bufferSize);
        void* dBuffer = get_cached_buffer(bufcache, bufferSize);
        cusparseSpMM(spHandle, spTransA, spTransB,
                    &alpha, A.descr(), B_dn, &betta, C_dn,
                    CUDA_C_64F, CUSPARSE_SPMM_CSR_ALG1, dBuffer);
        cusparseDestroyDnMat(B_dn);
        cusparseDestroyDnMat(C_dn);
    }

    // ====================== dense-sparse (COMPLEX) ======================
    template <>
    void optimized_multiply(const CUDA_Matrix<COMPLEX>& A,
                            const CUDA_CSR_Matrix<COMPLEX>& B,
                            CUDA_Matrix<COMPLEX>& C,
                            cuDoubleComplex alpha, cuDoubleComplex betta,
                            char opA, char opB) {                             // <-- один параметр op
        cudaStream_t stream = 0;
        cusparseHandle_t handle = B.handle();
        static std::unordered_map<cusparseHandle_t, std::pair<void*, size_t>> bufcache;
        auto get_cached_buffer = [&](auto& cache, size_t required) {
            auto& [ptr, size] = cache[handle];
            if (ptr && size < required) {
                std::cout << "HERE3 " << required << std::endl;
                cudaFree(ptr);
                ptr = nullptr;
                size = 0;
            }
            if (!ptr) {
                cudaMalloc(&ptr, required);
                size = required;
            }
            return ptr;
        };

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
        if (opB == 'N') spOpB = CUSPARSE_OPERATION_TRANSPOSE;
        else if (opB == 'T') spOpB = CUSPARSE_OPERATION_NON_TRANSPOSE;
        else if (opB == 'C') spOpB = CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE; // если нужно
        else throw std::invalid_argument("Unsupported op for dense-sparse");

        cusparseOperation_t spOpA;
        if (opA == 'N') spOpA = CUSPARSE_OPERATION_TRANSPOSE;
        else if (opA == 'T') spOpA = CUSPARSE_OPERATION_NON_TRANSPOSE;
        else if (opA == 'C') spOpA = CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE; // если нужно
        else throw std::invalid_argument("Unsupported op for dense-sparse");

        CUDA_Matrix<COMPLEX> Ct(blasHandle, B_cols, A_rows);
        cusparseDnMatDescr_t A_t_dn, Ct_dn;
        cusparseCreateDnMat(&A_t_dn, A_cols, A_rows, A.ld(), (void*)A.data(),
                            CUDA_C_64F, CUSPARSE_ORDER_COL);   // A^T (column-major, размеры переставлены)
        cusparseCreateDnMat(&Ct_dn, B_cols, A_rows, Ct.ld(), (void*)Ct.data(),
                            CUDA_C_64F, CUSPARSE_ORDER_COL);

        size_t bufferSize;
        cusparseSpMM_bufferSize(spHandle, spOpB, spOpA,
                                &alpha, B.descr(), A_t_dn, &betta, Ct_dn,
                                CUDA_C_64F, CUSPARSE_SPMM_CSR_ALG1, &bufferSize);
        void* dBuffer = get_cached_buffer(bufcache, bufferSize);
        cusparseSpMM(spHandle, spOpB, spOpA,
                    &alpha, B.descr(), A_t_dn, &betta, Ct_dn,
                    CUDA_C_64F, CUSPARSE_SPMM_CSR_ALG1, dBuffer);
        // cudaFree(dBuffer);
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
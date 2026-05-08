#ifdef __CUDACC__

#include "dynamic.hpp"

namespace QComputations {
    CUDA_Rho create_init_rho_cuda(cublasHandle_t handle, const std::vector<COMPLEX>& init_state) {
        size_t dim = init_state.size();
        Rho rho(FORTRAN_STYLE, dim, dim);
        for (size_t j = 0; j < dim; j++) {
            for (size_t i = 0; i < dim; i++) {
                rho(i, j) = init_state[i] * std::conj(init_state[j]);
            }
        }

        return CUDA_Rho(handle, rho);
    }

    Probs quantum_master_equation(const State<Basis_State>& init_state,
                                CUDA_Hamiltonian& H,
                                const std::vector<double>& time_vec,
                                bool is_full_rho) {
        
        size_t dim = H.size();
        std::vector<std::function<void(const CUDA_Rho& rho)>> lindblads;

        CUDA_Matrix<COMPLEX> T1(H.handle(), dim, dim);
        CUDA_Matrix<COMPLEX> T2(H.handle(), dim, dim);

        auto H_matrix = H.get_matrix();

        for (const auto& p: H.get_decoherence()) {
            auto gamma = p.first;
            //std::cout << "BEFORE: " << p.second.matrix_type() << std::endl;
            //BLOCKED_Matrix<COMPLEX> A(p.second);
            const CUDA_Matrix<COMPLEX>& A = p.second;
            //A.show();
            lindblads.push_back(std::function<void(const CUDA_Rho& rho)> {
                [A, &T1, &T2, gamma](const CUDA_Rho& rho) {
                    //std::cout << "HERE4\n";
                    //std::cout << A.matrix_type() << std::endl;
                    optimized_multiply(A, A, T1, cuDoubleComplex{1, 0}, cuDoubleComplex{0, 0}, CUBLAS_OP_C, CUBLAS_OP_N); // AconjA -> T1
                    //std::cout << "HERE5\n";
                    //std::cout << T1.matrix_type() << " " << rho.matrix_type() << std::endl;
                    optimized_multiply(T1, rho, T2, cuDoubleComplex{1, 0}, cuDoubleComplex{0, 0}); // AconjA*rho -> T2
                    //std::cout << "HERE6\n";
                    optimized_multiply(rho, T1, T2, cuDoubleComplex{1, 0}, cuDoubleComplex{1, 0}); // rho * AconjA + AconjA * rho
                    //std::cout << "HERE7\n";
                    optimized_multiply(A, rho, T1, cuDoubleComplex{1, 0}, cuDoubleComplex{0, 0}); // A*rho -> T1
                    //std::cout << "HERE8\n";
                    optimized_multiply(T1, A, T2, cuDoubleComplex{gamma, 0}, cuDoubleComplex{-0.5 * gamma, 0}, CUBLAS_OP_N, CUBLAS_OP_C); // res -> T2
                    //std::cout << "HERE9\n";
                    //auto Aconj = A.hermit();
                    //auto AconjA = Aconj * A;
                    //T2 = (A * rho * Aconj - (AconjA * rho + rho * AconjA) * COMPLEX(0.5, 0)) * gamma;
                }
            }
            );
        }

        
        std::function<void(double t, const CUDA_Rho&, CUDA_Matrix<COMPLEX>&)> equation 
        {[&H_matrix, &T1, &T2, &lindblads](double t, const CUDA_Rho& rho, CUDA_Matrix<COMPLEX>& res) {
            //std::cout << "HERE1\n";
            optimized_multiply(rho, H_matrix, res, cuDoubleComplex{1, 0}, cuDoubleComplex{0, 0}); // rho * H_matrix -> res
            //std::cout << "HERE2\n";
            optimized_multiply(H_matrix, rho, res, cuDoubleComplex{0, -1 / QConfig::instance().h()}, cuDoubleComplex{0, 1 / QConfig::instance().h()}); // result -> res
            //std::cout << "HERE3\n";

            for (const auto& lindblad: lindblads) {
                lindblad(rho);
                //std::cout << "HERE10\n";
                //optimized_add(T2, res, COMPLEX(1 / QConfig::instance().h(), 0), COMPLEX(1, 0));
                //T2 /= QConfig::instance().h();
                //res += T2;
                optimized_add(T2, res, res, cuDoubleComplex{1 / QConfig::instance().h(), 0}, cuDoubleComplex{1, 0});
                //std::cout << "HERE11\n";
            }

            //res = (H_matrix * rho - rho * H_matrix) * COMPLEX(0, -1/QConfig::instance().h());
            //for (const auto& lindblad: lindblads) {
            //    lindblad(rho);
            //    res += (T2 / QConfig::instance().h());
            //}
        }};

        CUDA_Rho rho_0(std::move(create_init_rho_cuda(H.handle(), init_state.fit_to_basis_state(H.get_basis()).get_vector())));
        //rho_0.show();
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        //std::cout << "HERE\n";
        std::vector<CUDA_Rho> rho_vec;
        CUDA_Probs probs(H.handle(), dim, time_vec.size(), 1);
        if (QConfig::instance().qme_algorithm() == RUNGE_KUTT_4) {
            //rho_vec = Runge_Kutt_4<double, Rho>(time_vec, rho_0, equation);
            rho_vec = CUDA_QME_OPT_Runge_Kutt_4(time_vec, rho_0, equation);
        } else if (QConfig::instance().qme_algorithm() == RUNGE_KUTT_2) {
            //rho_vec = Runge_Kutt_2<double, Rho>(time_vec, rho_0, equation);
            rho_vec = CUDA_QME_OPT_Runge_Kutt_2(time_vec, rho_0, equation);
        } else {
            assert(false); // Неизвестный алгоритм решения ОКУ
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float time;
        cudaEventElapsedTime(&time, start, stop);
        std::cout << time << std::endl;

        std::vector<cuDoubleComplex*> rho_pointers(time_vec.size());
        for (size_t t = 0; t < time_vec.size(); t++) {
            rho_pointers[t] = rho_vec[t].data();
        }

        cuDoubleComplex** rho_pointers_dev;
        CUDA::cudaMalloc(reinterpret_cast<void**>(&rho_pointers_dev), sizeof(cuDoubleComplex*) * time_vec.size());
        CUDA::cudaMemcpy(rho_pointers_dev, rho_pointers.data(), sizeof(cuDoubleComplex*) * time_vec.size(), cudaMemcpyHostToDevice);

        rho_to_probs<<<QConfig::instance().cuda_grid_size(), QConfig::instance().cuda_block_size()>>>(rho_pointers_dev, probs.data(), time_vec.size(), H.n());
        
        cudaFree(rho_pointers_dev);
        //cudaDeviceSynchronize();
        //std::cout << probs.n() << " " << probs.m() << " " << probs.ld() << "\n";
        auto probs_cpu = probs.to_cpu();
        //auto end_c = std::chrono::steady_clock::now();
        //std::cout << " c " << std::chrono::duration_cast<std::chrono::milliseconds>(end_c - begin_c).count() << std::endl;
        //std::cout << "HERE 2\n";

        /*
        for (size_t i = 0; i < dim; i++) {
            for (size_t t = 0; t < time_vec.size(); t++) {
                probs[i][t] = std::abs(rho_vec[t][i][i]);
            }
        }
        */

        /*
        for (size_t t = 0; t < time_vec.size(); t++) {
            double res = 0.0;
            for (size_t i = 0; i < dim; i++) {
                res += probs[i][t];
            }

            //std::cout << t << " " << res << std::endl;

            if (std::abs(res - 1) >= QConfig::instance().eps()) {
                //std::cout << t << " " << res << std::endl;
            }
        }
        */
        return probs_cpu;
    }

    Probs quantum_master_equation(const std::vector<COMPLEX>& init_state,
                                CUDA_Hamiltonian& H,
                                const std::vector<double>& time_vec,
                                bool is_full_rho) {
        
        size_t dim = H.size();
        std::vector<std::function<void(const CUDA_Rho& rho)>> lindblads;

        CUDA_Matrix<COMPLEX> T1(H.handle(), dim, dim);
        CUDA_Matrix<COMPLEX> T2(H.handle(), dim, dim);

        auto H_matrix = H.get_matrix();

        for (const auto& p: H.get_decoherence()) {
            auto gamma = p.first;
            //std::cout << "BEFORE: " << p.second.matrix_type() << std::endl;
            //BLOCKED_Matrix<COMPLEX> A(p.second);
            const CUDA_Matrix<COMPLEX>& A = p.second;
            //A.show();
            lindblads.push_back(std::function<void(const CUDA_Rho& rho)> {
                [A, &T1, &T2, gamma](const CUDA_Rho& rho) {
                    //std::cout << "HERE4\n";
                    //std::cout << A.matrix_type() << std::endl;
                    optimized_multiply(A, A, T1, cuDoubleComplex{1, 0}, cuDoubleComplex{0, 0}, CUBLAS_OP_C, CUBLAS_OP_N); // AconjA -> T1
                    //std::cout << "HERE5\n";
                    //std::cout << T1.matrix_type() << " " << rho.matrix_type() << std::endl;
                    optimized_multiply(T1, rho, T2, cuDoubleComplex{1, 0}, cuDoubleComplex{0, 0}); // AconjA*rho -> T2
                    //std::cout << "HERE6\n";
                    optimized_multiply(rho, T1, T2, cuDoubleComplex{1, 0}, cuDoubleComplex{1, 0}); // rho * AconjA + AconjA * rho
                    //std::cout << "HERE7\n";
                    optimized_multiply(A, rho, T1, cuDoubleComplex{1, 0}, cuDoubleComplex{0, 0}); // A*rho -> T1
                    //std::cout << "HERE8\n";
                    optimized_multiply(T1, A, T2, cuDoubleComplex{gamma, 0}, cuDoubleComplex{-0.5 * gamma, 0}, CUBLAS_OP_N, CUBLAS_OP_C); // res -> T2
                    //std::cout << "HERE9\n";
                    //auto Aconj = A.hermit();
                    //auto AconjA = Aconj * A;
                    //T2 = (A * rho * Aconj - (AconjA * rho + rho * AconjA) * COMPLEX(0.5, 0)) * gamma;
                }
            }
            );
        }

        
        std::function<void(double t, const CUDA_Rho&, CUDA_Matrix<COMPLEX>&)> equation 
        {[&H_matrix, &T1, &T2, &lindblads](double t, const CUDA_Rho& rho, CUDA_Matrix<COMPLEX>& res) {
            //std::cout << "HERE1\n";
            optimized_multiply(rho, H_matrix, res, cuDoubleComplex{1, 0}, cuDoubleComplex{0, 0}); // rho * H_matrix -> res
            //std::cout << "HERE2\n";
            optimized_multiply(H_matrix, rho, res, cuDoubleComplex{0, -1 / QConfig::instance().h()}, cuDoubleComplex{0, 1 / QConfig::instance().h()}); // result -> res
            //std::cout << "HERE3\n";

            for (const auto& lindblad: lindblads) {
                lindblad(rho);
                //std::cout << "HERE10\n";
                //optimized_add(T2, res, COMPLEX(1 / QConfig::instance().h(), 0), COMPLEX(1, 0));
                //T2 /= QConfig::instance().h();
                //res += T2;
                optimized_add(T2, res, res, cuDoubleComplex{1 / QConfig::instance().h(), 0}, cuDoubleComplex{1, 0});
                //std::cout << "HERE11\n";
            }

            //res = (H_matrix * rho - rho * H_matrix) * COMPLEX(0, -1/QConfig::instance().h());
            //for (const auto& lindblad: lindblads) {
            //    lindblad(rho);
            //    res += (T2 / QConfig::instance().h());
            //}
        }};

        CUDA_Rho rho_0(std::move(create_init_rho_cuda(H.handle(), init_state)));
        //rho_0.show();
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        //std::cout << "HERE\n";
        std::vector<CUDA_Rho> rho_vec;
        CUDA_Probs probs(H.handle(), dim, time_vec.size());
        if (QConfig::instance().qme_algorithm() == RUNGE_KUTT_4) {
            //rho_vec = Runge_Kutt_4<double, Rho>(time_vec, rho_0, equation);
            rho_vec = CUDA_QME_OPT_Runge_Kutt_4(time_vec, rho_0, equation);
        } else if (QConfig::instance().qme_algorithm() == RUNGE_KUTT_2) {
            //rho_vec = Runge_Kutt_2<double, Rho>(time_vec, rho_0, equation);
            rho_vec = CUDA_QME_OPT_Runge_Kutt_2(time_vec, rho_0, equation);
        } else {
            assert(false); // Неизвестный алгоритм решения ОКУ
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float time;
        cudaEventElapsedTime(&time, start, stop);
        std::cout << time << std::endl;

        std::vector<cuDoubleComplex*> rho_pointers(time_vec.size());
        for (size_t t = 0; t < time_vec.size(); t++) {
            rho_pointers[t] = rho_vec[t].data();
        }

        cuDoubleComplex** rho_pointers_dev;
        CUDA::cudaMalloc(reinterpret_cast<void**>(&rho_pointers_dev), sizeof(cuDoubleComplex*) * time_vec.size());
        CUDA::cudaMemcpy(rho_pointers_dev, rho_pointers.data(), sizeof(cuDoubleComplex*) * time_vec.size(), cudaMemcpyHostToDevice);

        rho_to_probs<<<QConfig::instance().cuda_grid_size(), QConfig::instance().cuda_block_size()>>>(rho_pointers_dev, probs.data(), time_vec.size(), H.n());
        cudaDeviceSynchronize();
        //std::cout << probs.n() << " " << probs.m() << " " << probs.ld() << "\n";
        auto probs_cpu = probs.to_cpu();
        //auto end_c = std::chrono::steady_clock::now();
        //std::cout << " c " << std::chrono::duration_cast<std::chrono::milliseconds>(end_c - begin_c).count() << std::endl;
        //std::cout << "HERE 2\n";

        /*
        for (size_t i = 0; i < dim; i++) {
            for (size_t t = 0; t < time_vec.size(); t++) {
                probs[i][t] = std::abs(rho_vec[t][i][i]);
            }
        }
        */

        /*
        for (size_t t = 0; t < time_vec.size(); t++) {
            double res = 0.0;
            for (size_t i = 0; i < dim; i++) {
                res += probs[i][t];
            }

            //std::cout << t << " " << res << std::endl;

            if (std::abs(res - 1) >= QConfig::instance().eps()) {
                //std::cout << t << " " << res << std::endl;
            }
        }
        */

        cudaFree(rho_pointers_dev);
        return probs_cpu;
    }

    Probs quantum_master_equation(const State<Basis_State>& init_state,
                            CUDA_CSR_Hamiltonian& H,
                            const std::vector<double>& time_vec) {
        return quantum_master_equation(init_state.fit_to_basis_state(H.get_basis()).get_vector(), H, time_vec);
    }

    Probs quantum_master_equation(const std::vector<COMPLEX>& init_state,
                                CUDA_CSR_Hamiltonian& H,
                                const std::vector<double>& time_vec) {
        
        size_t dim = H.size();
        std::vector<std::function<void(const CUDA_Matrix<COMPLEX>& rho)>> lindblads;
        cublasHandle_t handle;
        cublasCreate(&handle);

        CUDA_CSR_Matrix<COMPLEX> T1(H.handle(), dim, dim);
        CUDA_Matrix<COMPLEX> T2(handle, dim, dim);
        CUDA_Matrix<COMPLEX> T3(handle, dim, dim);

        for (const auto& p: H.get_decoherence()) {
            auto gamma = p.first;
            //std::cout << "BEFORE: " << p.second.matrix_type() << std::endl;
            //BLOCKED_Matrix<COMPLEX> A(p.second);
            const CUDA_CSR_Matrix<COMPLEX>& A = p.second;
            //A.show();
            lindblads.push_back(std::function<void(const CUDA_Matrix<COMPLEX>& rho)> {
                [&A, &T1, &T2, &T3, gamma](const CUDA_Matrix<COMPLEX>& rho) {
                    optimized_multiply(A, A, T1, cuDoubleComplex{1, 0}, cuDoubleComplex{0, 0}, 'C', 'N'); // AconjA -> T1
                    //std::cout << "HERE5\n";
                    //std::cout << T1.matrix_type() << " " << rho.matrix_type() << std::endl;
                    optimized_multiply(T1, rho, T2, cuDoubleComplex{1, 0}, cuDoubleComplex{0, 0}); // AconjA*rho -> T2
                    //std::cout << "HERE6\n";
                    optimized_multiply(rho, T1, T2, cuDoubleComplex{1, 0}, cuDoubleComplex{1, 0}); // rho * AconjA + AconjA * rho
                    //std::cout << "HERE7\n";
                    optimized_multiply(A, rho, T3, cuDoubleComplex{1, 0}, cuDoubleComplex{0, 0}); // A*rho -> T1
                    //std::cout << "HERE8\n";
                    optimized_multiply(T3, A, T2, cuDoubleComplex{gamma, 0}, cuDoubleComplex{-0.5 * gamma, 0}, 'N', 'C'); // res -> T2
                }
            }
            );
        }

        const CSR_CUDA_Matrix<COMPLEX>& H_matrix = H.get_matrix();
        std::function<void(double t, const CUDA_Matrix<COMPLEX>&, CUDA_Matrix<COMPLEX>&)> equation 
        {[&H_matrix, &T1, &T2, &lindblads](double t, const CUDA_Matrix<COMPLEX>& rho, CUDA_Matrix<COMPLEX>& res) {
            //std::cout << "HERE1\n";
            optimized_multiply(rho, H_matrix, res, cuDoubleComplex{1, 0}, cuDoubleComplex{0, 0}); // rho * H_matrix -> res
            //std::cout << "HERE2\n";
            optimized_multiply(H_matrix, rho, res, cuDoubleComplex{0, -1 / QConfig::instance().h()}, cuDoubleComplex{0, 1 / QConfig::instance().h()}); // result -> res
            //std::cout << "HERE3\n";

            for (const auto& lindblad: lindblads) {
                lindblad(rho);
                //std::cout << "HERE10\n";
                //optimized_add(T2, res, COMPLEX(1 / QConfig::instance().h(), 0), COMPLEX(1, 0));
                optimized_add(T2, res, res, cuDoubleComplex{1 / QConfig::instance().h(), 0}, cuDoubleComplex{1, 0});
                //std::cout << "HERE11\n";
            }
        }};

        auto rho_0 = create_init_rho(init_state);
        Probs probs(C_STYLE, dim, time_vec.size());
        if (QConfig::instance().qme_algorithm() == RUNGE_KUTT_4) {
            //rho_vec = Runge_Kutt_4<double, Rho>(time_vec, rho_0, equation);
            // QME_OPT_Runge_Kutt_4(time_vec, rho_0, equation, probs);
        } else if (QConfig::instance().qme_algorithm() == RUNGE_KUTT_2) {
            //rho_vec = Runge_Kutt_2<double, Rho>(time_vec, rho_0, equation);
            QME_OPT_Runge_Kutt_2(time_vec, rho_0, equation, probs);
        } else {
            assert(false); // Неизвестный алгоритм решения ОКУ
        }

        cublasDestroy(handle);
        return probs;
    }

    __global__ void rho_to_probs(cuDoubleComplex** rho_vec, double* probs, size_t time_length, size_t basis_size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int grid_size = blockDim.x * gridDim.x;

        for (size_t i = idx; i < time_length * basis_size; i += grid_size) {
            size_t j = i % basis_size;
            probs[i] = cuCabs(rho_vec[i / basis_size][j + basis_size * j]);
            //printf("%lf %llu %llu\n", cuCabs(rho_vec[i / basis_size][j + basis_size * j]), i / basis_size, j);
        }
    }

   Probs schrodinger(const State<Basis_State>& init_state, CUDA_Hamiltonian& H, const std::vector<double>& time_vec) {
        CUDA_Matrix<COMPLEX> eigenvectors = H.eigenvectors();
        double* eigenvalues = H.eigenvalues();
        
        const ILP_TYPE n = eigenvectors.n();
        const ILP_TYPE num_times = time_vec.size();
        const double hbar = QConfig::instance().h();

        auto init_state_vec = init_state.fit_to_basis_state(H.get_basis()).get_vector();
        std::vector<COMPLEX> psi0_host = init_state_vec;

        CUDA_Matrix<COMPLEX> psi0_device(eigenvectors.handle(), n, 1, init_state_vec);

        CUDA_Matrix<COMPLEX> lambda_device(eigenvectors.handle(), n, 1);
        
        cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
        cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
        
        // Zgemv: y = alpha * op(A) * x + beta * y
        // op(A) = CUBLAS_OP_C (сопряженная транспонированная)
        // cublasZgemv(
        //     eigenvectors.handle(),
        //     CUBLAS_OP_C,
        //     n, n,
        //     &alpha,
        //     reinterpret_cast<const cuDoubleComplex*>(eigenvectors.data()), eigenvectors.ld(),
        //     reinterpret_cast<const cuDoubleComplex*>(psi0_device.data()), 1,
        //     &beta,
        //     reinterpret_cast<cuDoubleComplex*>(lambda_device.data()), 1
        // );

        CUDA::cublasZGEMV(eigenvectors.handle(), CUBLAS_OP_C, n, n, eigenvectors.data(), eigenvectors.ld(), psi0_device.data(), lambda_device.data());

        double* d_time_vec = nullptr;
        cudaMalloc((void**)&d_time_vec, sizeof(double) * time_vec.size());
        cudaMemcpy(d_time_vec, time_vec.data(), sizeof(double) * time_vec.size(), cudaMemcpyHostToDevice);

        double* d_probs = nullptr;
        cudaMalloc((void**)&d_probs, sizeof(double) * n * time_vec.size());

        const size_t MAX_BLOCK_SIZE = 32;
        size_t block_x = std::min(size_t(n), MAX_BLOCK_SIZE);
        size_t block_y = std::min(time_vec.size(), MAX_BLOCK_SIZE);
        dim3 blockDim(block_x, block_y);
        size_t grid_x = std::min(std::max(size_t(n) / block_x, size_t(1)), QConfig::instance().cuda_grid_size());
        size_t grid_y = std::min(std::max(time_vec.size() / block_y, size_t(1)), QConfig::instance().cuda_grid_size());
        dim3 gridDim(grid_x, grid_y);
        CUDA::compute_psi_and_probs_kernel<<<gridDim, blockDim>>>(
            d_probs,
            reinterpret_cast<const cuDoubleComplex*>(eigenvectors.data()),
            reinterpret_cast<const cuDoubleComplex*>(lambda_device.data()),
            eigenvalues,
            d_time_vec,
            hbar,
            n,
            time_vec.size()
        );

        cudaDeviceSynchronize();

        cudaError_t sync_error = cudaGetLastError();
        if (sync_error != cudaSuccess) {
            std::cerr << "CUDA error after kernel execution: " << cudaGetErrorString(sync_error) << std::endl;
            std::cerr << "Configuration: grid(" << grid_x << "," << grid_y 
              << "), block(" << block_x << "," << block_y << ")" << std::endl;
        }

        std::vector<double> probs_host(n * time_vec.size());
        cudaMemcpy(probs_host.data(), d_probs, sizeof(double) * n * time_vec.size(), cudaMemcpyDeviceToHost);

        Probs probs(C_STYLE, probs_host, n, time_vec.size());

        cudaFree(d_time_vec);
        cudaFree(d_probs);
        
        return probs;
    }

#ifdef ENABLE_MPI
#ifdef ENABLE_CLUSTER

BLOCKED_CUDA_Matrix<COMPLEX, cuDoubleComplex> create_BLOCKED_CUDA_init_rho(
    MPI_Comm comm, ncclComm_t nccl_comm, cublasMpHandle_t handle, cublasMpGrid_t grid,
    const std::vector<COMPLEX>& init_state, int64_t NB = 0, int64_t MB = 0)
{
    int64_t dim = init_state.size();
    BLOCKED_CUDA_Matrix<COMPLEX, cuDoubleComplex> rho(comm, nccl_comm, handle, grid, dim, dim, NB, MB);

    std::vector<COMPLEX> psi(dim);
    std::copy(init_state.begin(), init_state.end(), psi.begin());

    int64_t lrows = rho.local_rows();
    int64_t lcols = rho.local_cols();
    std::vector<COMPLEX> local_rho(lrows * lcols, COMPLEX(0.0, 0.0));
    for (int64_t j = 0; j < lcols; ++j) {
        int64_t global_col = rho.get_global_col(j);
        COMPLEX psi_col = psi[global_col];
        for (int64_t i = 0; i < lrows; ++i) {
            int64_t global_row = rho.get_global_row(i);
            COMPLEX psi_row = psi[global_row];
            local_rho[i + j * lrows] = psi_row * std::conj(psi_col);
        }
    }
    cudaMemcpy(rho.local_data(), local_rho.data(), lrows * lcols * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    return rho;
}

void quantum_master_equation(const std::vector<COMPLEX>& init_state,
                                BLOCKED_CUDA_Hamiltonian& H,
                                const std::vector<double>& time_vec) {
    size_t dim = H.size();
    std::vector<std::function<void(const BLOCKED_CUDA_Matrix<COMPLEX>& rho)>> lindblads;

    // size_t NB = 64, MB = 64;

    BLOCKED_CUDA_Matrix<COMPLEX> T1(MPI_COMM_WORLD, H.nccl_comm(), H.handle(), H.grid(), dim, dim);
    BLOCKED_CUDA_Matrix<COMPLEX> T2(MPI_COMM_WORLD, H.nccl_comm(), H.handle(), H.grid(), dim, dim);
    const auto& H_matrix = H.get_blocked_matrix();

    // size_t hostSize = 0, devSize = 0, hostSizeAdd = 0, devSizeAdd = 0;

    // H.get_buffersize_gemm(hostSize, devSize);

    // void* d_work = nullptr;
    // void* h_work = nullptr;
    // if (devSize) CUDA_CHECK(cudaMalloc(&d_work, devSize));
    // if (hostSize) h_work = malloc(hostSize);

    for (const auto& p: H.get_decoherence()) {
        auto gamma = p.first;
        //std::cout << "BEFORE: " << p.second.matrix_type() << std::endl;
        //BLOCKED_Matrix<COMPLEX> A(p.second);
        const auto& A = p.second;
        //A.show();
        lindblads.push_back(std::function<void(const BLOCKED_CUDA_Matrix<COMPLEX>& rho)> {
            // [&H_matrix, &A, &T1, &T2, gamma, d_work, h_work, hostSize, devSize](const BLOCKED_CUDA_Matrix<COMPLEX>& rho) {
            [&H_matrix, &A, &T1, &T2, gamma](const BLOCKED_CUDA_Matrix<COMPLEX>& rho) {
                // std::cout << "HERE4\n";
                //std::cout << A.matrix_type() << std::endl;
                optimized_multiply(A, A, T1, COMPLEX(1, 0), COMPLEX(0, 0), CUBLAS_OP_C, CUBLAS_OP_N); // AconjA -> T1
                // std::cout << "HERE5\n";
                //std::cout << T1.matrix_type() << " " << rho.matrix_type() << std::endl;
                optimized_multiply(T1, rho, T2, COMPLEX(1, 0), COMPLEX(0, 0), CUBLAS_OP_N, CUBLAS_OP_N); // AconjA*rho -> T2
                // std::cout << "HERE6\n";
                optimized_multiply(rho, T1, T2, COMPLEX(1, 0), COMPLEX(1, 0), CUBLAS_OP_N, CUBLAS_OP_N); // rho * AconjA + AconjA * rho
                // std::cout << "HERE7\n";
                optimized_multiply(A, rho, T1, COMPLEX(1, 0), COMPLEX(0, 0), CUBLAS_OP_N, CUBLAS_OP_N); // A*rho -> T1
                // std::cout << "HERE8\n";
                optimized_multiply(T1, A, T2, COMPLEX(gamma, 0), COMPLEX(-0.5 * gamma, 0), CUBLAS_OP_N, CUBLAS_OP_C); // res -> T2
                // std::cout << "HERE9\n";

                // std::cout << "HERE4\n";
                // //std::cout << A.matrix_type() << std::endl;
                // optimized_multiply(A, A, T1, COMPLEX(1, 0), COMPLEX(0, 0), CUBLAS_OP_C, CUBLAS_OP_N, d_work, devSize, h_work, hostSize); // AconjA -> T1
                // std::cout << "HERE5\n";
                // //std::cout << T1.matrix_type() << " " << rho.matrix_type() << std::endl;
                // optimized_multiply(T1, rho, T2, COMPLEX(1, 0), COMPLEX(0, 0), CUBLAS_OP_N, CUBLAS_OP_N, d_work, devSize, h_work, hostSize); // AconjA*rho -> T2
                // std::cout << "HERE6\n";
                // optimized_multiply(rho, T1, T2, COMPLEX(1, 0), COMPLEX(1, 0), CUBLAS_OP_N, CUBLAS_OP_N, d_work, devSize, h_work, hostSize); // rho * AconjA + AconjA * rho
                // std::cout << "HERE7\n";
                // optimized_multiply(A, rho, T1, COMPLEX(1, 0), COMPLEX(0, 0), CUBLAS_OP_N, CUBLAS_OP_N, d_work, devSize, h_work, hostSize); // A*rho -> T1
                // std::cout << "HERE8\n";
                // optimized_multiply(T1, A, T2, COMPLEX(gamma, 0), COMPLEX(-0.5 * gamma, 0), CUBLAS_OP_N, CUBLAS_OP_C, d_work, devSize, h_work, hostSize); // res -> T2
                // std::cout << "HERE9\n";
                //auto Aconj = A.hermit();
                //auto AconjA = Aconj * A;
                //T2 = (A * rho * Aconj - (AconjA * rho + rho * AconjA) * COMPLEX(0.5, 0)) * gamma;
            }
        }
        );
    }

    // H.get_buffersize_geadd(hostSizeAdd, devSizeAdd);

    // void* d_work_geadd = nullptr;
    // void* h_work_geadd = nullptr;
    // if (devSizeAdd) CUDA_CHECK(cudaMalloc(&d_work_geadd, devSizeAdd));
    // if (hostSizeAdd) h_work_geadd = malloc(hostSizeAdd);


    std::function<void(double t, const BLOCKED_CUDA_Matrix<COMPLEX>&, BLOCKED_CUDA_Matrix<COMPLEX>&)> equation 
    // {[&H_matrix, &T1, &T2, &lindblads, d_work, h_work, devSize, hostSize, d_work_geadd, h_work_geadd, devSizeAdd, hostSizeAdd](double t, const BLOCKED_CUDA_Matrix<COMPLEX>& rho, 
    {[&H_matrix, &T1, &T2, &lindblads](double t, const BLOCKED_CUDA_Matrix<COMPLEX>& rho, 
    BLOCKED_CUDA_Matrix<COMPLEX>& res) {
        // std::cout << "HERE1\n";
        optimized_multiply(rho, H_matrix, res, COMPLEX(1, 0), COMPLEX(0, 0), CUBLAS_OP_N, CUBLAS_OP_N); // rho * H_matrix -> res
        // std::cout << "HERE2\n";
        optimized_multiply(H_matrix, rho, res, COMPLEX(0, -1 / QConfig::instance().h()), COMPLEX(0, 1 / QConfig::instance().h()), CUBLAS_OP_N, CUBLAS_OP_N); // result -> res
        // std::cout << "HERE3\n";

        for (const auto& lindblad: lindblads) {
            lindblad(rho);
            //std::cout << "HERE10\n";
            optimized_add(T2, res, COMPLEX(1 / QConfig::instance().h(), 0), COMPLEX(1, 0), CUBLAS_OP_N);
            //std::cout << "HERE11\n";
        }

        //         std::cout << "HERE1\n";
        // optimized_multiply(rho, H_matrix, res, COMPLEX(1, 0), COMPLEX(0, 0), CUBLAS_OP_N, CUBLAS_OP_N, d_work, devSize, h_work, hostSize); // rho * H_matrix -> res
        // std::cout << "HERE2\n";
        // optimized_multiply(H_matrix, rho, res, COMPLEX(0, -1 / QConfig::instance().h()), COMPLEX(0, 1 / QConfig::instance().h()), CUBLAS_OP_N, CUBLAS_OP_N, d_work, devSize, h_work, hostSize); // result -> res
        // std::cout << "HERE3\n";

        // for (const auto& lindblad: lindblads) {
        //     lindblad(rho);
        //     //std::cout << "HERE10\n";
        //     optimized_add(T2, res, COMPLEX(1 / QConfig::instance().h(), 0), COMPLEX(1, 0), CUBLAS_OP_N, d_work_geadd, devSizeAdd, h_work_geadd, hostSizeAdd);
        //     //std::cout << "HERE11\n";
        // }

        //res = (H_matrix * rho - rho * H_matrix) * COMPLEX(0, -1/QConfig::instance().h());
        //for (const auto& lindblad: lindblads) {
        //    lindblad(rho);
        //    res += (T2 / QConfig::instance().h());
        //}
    }};

    BLOCKED_CUDA_Matrix<COMPLEX> rho_0(create_BLOCKED_CUDA_init_rho(MPI_COMM_WORLD, H.nccl_comm(), H.handle(), H.grid(), init_state));
    //std::cout << "RHO_0: " << rho_0.matrix_type() << std::endl;
    //rho_0.show();
    // auto begin_c = std::chrono::steady_clock::now();
    //std::cout << "HERE\n";
    std::vector<BLOCKED_CUDA_Matrix<COMPLEX>> rho_vec;
    // std::cout << "HERE START\n";
    if (QConfig::instance().qme_algorithm() == RUNGE_KUTT_4) {
        //rho_vec = Runge_Kutt_4<double, BLOCKED_Rho>(time_vec, rho_0, equation);
        // rho_vec = CUDA_MPI_Runge_Kutt_4(time_vec, rho_0, equation);
        assert(false);
    } else if (QConfig::instance().qme_algorithm() == RUNGE_KUTT_2) {
        //rho_vec = Runge_Kutt_2<double, BLOCKED_Rho>(time_vec, rho_0, equation);
        rho_vec = CUDA_MPI_Runge_Kutt_2<COMPLEX>(time_vec, std::move(rho_0), equation);
    } else {
        assert(false); // Неизвестный алгоритм решения ОКУ
    }

    //std::cout << "HERE 2\n";

    // ILP_TYPE world_size, rank;
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // ILP_TYPE probs_ctxt;
    // mpi::init_grid(probs_ctxt, world_size, 1);
    // ILP_TYPE proc_rows, proc_cols, myrow, mycol;
    // mpi::blacs_gridinfo(probs_ctxt, proc_rows, proc_cols, myrow, mycol);
    // BLOCKED_Probs probs(probs_ctxt, GE, dim, time_vec.size(), (H.size() >= world_size ? H.size() / world_size : 1), time_vec.size());

    // //std::cout << myrow << " " << mycol << " : " << probs.local_n() << " " << probs.local_m() << std::endl;
    // for (size_t t = 0; t < time_vec.size(); t++) {
    //     auto probs_vec = mpi::get_diagonal_elements<COMPLEX>(rho_vec[t].get_local_matrix(), rho_vec[t].desc());
    //     //if (rank == 0) std::cout << probs_vec << std::endl;
    //     //std::cout << myrow << " " << mycol << " - " << start << std::endl;
    //     for (size_t i = 0; i < probs.local_n(); i++) {   
    //         probs(i, t) = std::abs(probs_vec[probs.get_global_row(i)]);
    //     }
    // }

    /*
    for (size_t t = 0; t < time_vec.size(); t++) {
        double res = 0.0;
        for (size_t i = 0; i < dim; i++) {
            res += probs[i][t];
        }

        //std::cout << t << " " << res << std::endl;

        if (std::abs(res - 1) >= QConfig::instance().eps()) {
            //std::cout << t << " " << res << std::endl;
        }
    }
    */
    // if (d_work) cudaFree(d_work);
    // if (h_work) free(h_work);
    // if (d_work_geadd) cudaFree(d_work_geadd);
    // if (h_work_geadd) free(h_work_geadd);

    // return probs;
}

void quantum_master_equation(const State<Basis_State>& init_state,
                            BLOCKED_CUDA_Hamiltonian& H,
                            const std::vector<double>& time_vec) {
    quantum_master_equation(init_state.fit_to_basis_state(H.get_basis()).get_vector(), H, time_vec);
}

#endif
#endif

}

#endif
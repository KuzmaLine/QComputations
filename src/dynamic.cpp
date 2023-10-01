#include "dynamic.hpp"
#include "additional_operators.hpp"
#include "functions.hpp"
#include "quantum_operators.hpp"
#include "mpi_functions.hpp"

#include "plot.hpp"

namespace QComputations {

Matrix<COMPLEX> Evolution::create_A_destroy(const std::set<State>& basis, size_t cavity_id) {
    size_t dim = basis.size();
    Matrix<COMPLEX> A(DEFAULT_MATRIX_STYLE, dim, dim, 0);

    size_t index = 0;
    for (const auto& state: basis) {
        auto n = state.n(cavity_id);
        if (n != 0) {
            auto tmp_state = state;
            tmp_state.set_n(n - 1, cavity_id);

            auto state_index = tmp_state.get_index(basis);
            if (state_index != -1) A[state_index][index] = COMPLEX(std::sqrt(n));
        }

        index++;
    }

    return A;
}

Matrix<COMPLEX> Evolution::create_A_create(const std::set<State>& basis, size_t cavity_id) {
    size_t dim = basis.size();
    Matrix<COMPLEX> A(DEFAULT_MATRIX_STYLE, dim, dim, 0);

    size_t index = 0;
    for (const auto& state: basis) {
        auto n = state.n(cavity_id);
        auto tmp_state = state;
        tmp_state.set_n(n + 1, cavity_id);

        auto state_index = tmp_state.get_index(basis);
        if (state_index != -1) A[state_index][index] = COMPLEX(std::sqrt(n + 1));

        index++;
    }

    return A;
}

Evolution::Rho Evolution::create_init_rho(const std::vector<COMPLEX>& init_state) {
    size_t dim = init_state.size();
    Evolution::Rho rho(C_STYLE, dim, dim);
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            rho[i][j] = init_state[i] * std::conj(init_state[j]);
        }
    }

    return rho;
}

// ON MPI NEEDED
Evolution::Probs Evolution::schrodinger(const std::vector<COMPLEX>& init_state, Hamiltonian& H, const std::vector<double>& time_vec) {
    using namespace quantum;

    std::vector<double> eigen_values;
    Matrix<COMPLEX> eigen_vectors;
#ifdef ENABLE_MPI
    int rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == mpi::ROOT_ID) {
        mpi::make_command(COMMAND::SCHRODINGER);
        mpi::bcast_vector_complex(init_state);
        mpi::bcast_vector_double(time_vec);
#endif
    auto p = H.eigen();
    eigen_values = p.first;
    eigen_vectors = p.second;

#ifdef ENABLE_MPI
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == mpi::ROOT_ID) {
        mpi::bcast_vector_double(eigen_values);
        mpi::bcast_vector_complex(eigen_vectors.get_mass());
    } else {
        eigen_values = mpi::bcast_vector_double();
        eigen_vectors = Matrix<COMPLEX>(mpi::bcast_vector_complex(), eigen_values.size(), eigen_values.size(), C_STYLE); // c_style
    }
#endif
    std::vector<COMPLEX> lambda;
    for (size_t i = 0; i < eigen_values.size(); i++) {
        //std::cout << norm(eigen_vectors.col(i)) << std::endl;
        lambda.emplace_back(eigen_vectors.col(i) | init_state); // <PHI_i|KSI(0)> 
    }

    //std::cout << "L - " << norm(lambda) << std::endl;
    Probs probs(C_STYLE, eigen_values.size(), time_vec.size());
    size_t time_index = 0;
    eigen_vectors = eigen_vectors.transpose();
#ifdef ENABLE_MPI
    size_t start_col;
    auto rank_map = make_rank_map(time_vec.size(), rank, world_size, start_col);

    for (size_t time_index = start_col; time_index < start_col + rank_map[rank]; time_index++) {
        auto t = time_vec[time_index];
        std::vector<COMPLEX> psi_t(eigen_values.size(), 0);

        for (size_t i = 0; i < eigen_values.size(); i++) {
            for (size_t j = 0; j < psi_t.size(); j++) {
                psi_t[j] += lambda[i] * std::exp(COMPLEX(0, 1 / QConfig::instance().h() * eigen_values[i] * t)) * eigen_vectors[i][j];
            }
        }

        //std::cout << norm(psi_t) << std::endl;
        for (size_t i = 0; i < eigen_values.size(); i++) {
            double tmp = std::abs(psi_t[i]);
            probs[i][time_index] = tmp * tmp;
        }
    }

    size_t size = eigen_values.size();
    if (rank == mpi::ROOT_ID) {
        size_t col_index = rank_map[mpi::ROOT_ID];
        for (size_t i = mpi::ROOT_ID + 1; i < world_size; i++) {
            for (size_t j = rank_map[i]; j != 0; j--) {
                std::vector<double> col(size);
                //std::cout << i << " " << j << " " << col_index << std::endl;

                MPI_Recv(col.data(), size, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                //show_vector(col);
                probs.modify_col(col_index, col);
                col_index++;
            }
        }
    } else {
        for (size_t i = start_col; i < start_col + rank_map[rank]; i++) {
            std::vector<double> col = probs.col(i);

            MPI_Send(col.data(), size, MPI_DOUBLE, mpi::ROOT_ID, 0, MPI_COMM_WORLD);
        }
    }
#else
    for (const auto& t: time_vec) {
        std::vector<COMPLEX> psi_t(eigen_values.size(), 0);
        auto h = QConfig::instance().h();

        for (size_t i = 0; i < eigen_values.size(); i++) {
            for (size_t j = 0; j < psi_t.size(); j++) {
                psi_t[j] += lambda[i] * std::exp(COMPLEX(0, 1 / h * eigen_values[i] * t)) * eigen_vectors[i][j];
            }
        }

        //std::cout << norm(psi_t) << std::endl;

        for (size_t i = 0; i < eigen_values.size(); i++) {
            double tmp = std::abs(psi_t[i]);
            probs[i][time_index] = tmp * tmp;
        }
        time_index++;
    }
#endif
    return probs;
}

// ON MPI NEEDED
Evolution::Probs Evolution::quantum_master_equation(const std::vector<COMPLEX>& init_state,
                                                    Hamiltonian& H,
                                                    const std::vector<double>& time_vec,
                                                    bool is_full_rho) {
    size_t dim = H.size();
    //A.show(config::WIDTH);
    auto grid = H.get_grid();
    std::vector<std::function<Evolution::Rho(const Evolution::Rho& rho)>> lindblads;

    auto cavities_with_leak = grid.get_cavities_with_leak();
    auto cavities_with_gain = grid.get_cavities_with_gain();

    //for (const auto& cavity_id: cavities_with_leak) {
    for (size_t cavity_id = 0; cavity_id < grid.cavities_count(); cavity_id++) {
        auto A = create_A_destroy(H.get_basis(), cavity_id);
        //A.show();
        auto gamma = grid.get_leak_gamma(cavity_id);
        if (!is_zero(gamma)) {
            //std::cout << gamma << std::endl;
            lindblads.push_back(std::function<Evolution::Rho(const Evolution::Rho& rho)> {
                [A, gamma](const Evolution::Rho& rho) {
                    auto Aconj = A.hermit();
                    auto AconjA = Aconj * A;
                    return (A * rho * Aconj - (AconjA * rho + rho * AconjA) * COMPLEX(0.5)) * gamma;
                }
            }
            );
        }
    }

    //for (const auto& cavity_id: cavities_with_gain) {
    for (size_t cavity_id = 0; cavity_id < grid.cavities_count(); cavity_id++) {
        auto A = create_A_create(H.get_basis(), cavity_id);
        //A.show();
        auto gamma = grid.get_gain_gamma(cavity_id);
        if (!is_zero(gamma)) {
            //std::cout << gamma << std::endl;
            lindblads.push_back(std::function<Evolution::Rho(const Evolution::Rho& rho)> {
                [A, gamma](const Evolution::Rho& rho) {
                    auto Aconj = A.hermit();
                    auto AconjA = Aconj * A;
                    return (A * rho * Aconj - (AconjA * rho + rho * AconjA) * COMPLEX(0.5)) * gamma;
                }
            }
            );
        }
    }

    auto H_matrix = H.get_matrix();
    std::function<Evolution::Rho(double t, const Evolution::Rho&)> equation {[&H_matrix, &lindblads](double t, const Evolution::Rho& rho) {
        auto tmp = (H_matrix * rho - rho * H_matrix) * COMPLEX(0, -1);
        for (const auto& lindblad: lindblads) {
            tmp += lindblad(rho);
        }

        return tmp;
    }};

    auto rho_0 = Evolution::create_init_rho(init_state);
    //rho_0.show();
    //auto begin_c = std::chrono::steady_clock::now();
    auto rho_vec = Runge_Kutt_4<double, Evolution::Rho>(time_vec, rho_0, equation);
    //auto end_c = std::chrono::steady_clock::now();
    //std::cout << " c " << std::chrono::duration_cast<std::chrono::milliseconds>(end_c - begin_c).count() << std::endl;
    if (!is_full_rho) {
        Evolution::Probs probs(C_STYLE, dim, time_vec.size());

        for (size_t i = 0; i < dim; i++) {
            for (size_t t = 0; t < time_vec.size(); t++) {
                probs[i][t] = std::abs(rho_vec[t][i][i]);
            }
        }

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
        return probs;
    }

    Evolution::Probs probs(C_STYLE, dim * dim, time_vec.size());

    bool is_null = true;

    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            for (size_t t = 0; t < time_vec.size(); t++) {
                probs[i * dim + j][t] = std::abs(rho_vec[t][i][j]);
                if (probs[i * dim + j][t] >= QConfig::instance().eps()) {
                    is_null = false;
                }
            }

            if (is_null) probs[i * dim + j][0] = -1;
            is_null = true;
        }
    }

    return probs;
}


// ON MPI NEEDED
std::vector<double> Evolution::scan_gamma(const std::vector<COMPLEX>& init_state,
                                          Hamiltonian& H,
                                          size_t cavity_id,
                                          const std::vector<double>& time_vec,
                                          const std::vector<double>& gamma_vec,
                                          double target) {
    auto basis = H.get_basis();

    bool zero_state_in_basis = false;
    size_t index = 0;
    for (const auto& state: basis) {
        if (state.get_index() == 0) {
            zero_state_in_basis = true;
            break;
        }
        index++;
    }

    assert(zero_state_in_basis);
    auto begin = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();
    std::vector<double> tau_vec;
    for (size_t i = 0; i < gamma_vec.size(); i++) {
        double gamma = gamma_vec[i];
        H.set_leak(cavity_id, gamma);
        //begin = std::chrono::steady_clock::now();
        auto probs = quantum_master_equation(init_state, H, time_vec);
        //end = std::chrono::steady_clock::now();
        //std::cout << i << " " << gamma_vec.size() << " " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
        auto func = Cubic_Spline_Interpolate(time_vec, probs.row(index));
        //begin = std::chrono::steady_clock::now();
        //std::cout << " interp " << std::chrono::duration_cast<std::chrono::milliseconds>(begin - end).count() << std::endl;
        double tau = fsolve(func, time_vec[0], time_vec[time_vec.size() - 1], target);
        //end = std::chrono::steady_clock::now();
        //std::cout << " fsolve " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
        //std::cout << tau << " " << target << std::endl;
        tau_vec.emplace_back(tau);
    }

    return tau_vec;
}

#ifdef ENABLE_MPI
#ifdef ENABLE_CLUSTER

BLOCKED_Probs schrodinger(const std::vector<COMPLEX>& init_state, BLOCKED_Hamiltonian& H, const std::vector<double>& time_vec) {
    std::vector<double> eigen_values;
    Matrix<COMPLEX> eigen_vectors;
    auto p = H.eigen();
    eigen_values = p.first;
    eigen_vectors = p.second;

    std::vector<COMPLEX> lambda;
    for (size_t i = 0; i < eigen_values.size(); i++) {
        //std::cout << norm(eigen_vectors.col(i)) << std::endl;
        lambda.emplace_back(eigen_vectors.col(i) | init_state); // <PHI_i|KSI(0)> 
    }

    //std::cout << "L - " << norm(lambda) << std::endl;
    Probs probs(C_STYLE, eigen_values.size(), time_vec.size());
    size_t time_index = 0;
    eigen_vectors = eigen_vectors.transpose();
    for (const auto& t: time_vec) {
        std::vector<COMPLEX> psi_t(eigen_values.size(), 0);
        auto h = QConfig::instance().h();

        for (size_t i = 0; i < eigen_values.size(); i++) {
            for (size_t j = 0; j < psi_t.size(); j++) {
                psi_t[j] += lambda[i] * std::exp(COMPLEX(0, 1 / h * eigen_values[i] * t)) * eigen_vectors[i][j];
            }
        }

        //std::cout << norm(psi_t) << std::endl;

        for (size_t i = 0; i < eigen_values.size(); i++) {
            double tmp = std::abs(psi_t[i]);
            probs[i][time_index] = tmp * tmp;
        }
        time_index++;
    }
    return probs;
}

Evolution::Probs Evolution::Parallel_QME(const std::vector<COMPLEX>& init_state,
                                         Hamiltonian& H,
                                         const std::vector<double>& time_vec,
                                         bool is_full_rho) {
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (rank == mpi::ROOT_ID) {
        mpi::make_command(COMMAND::QME);

        mpi::bcast_vector_complex(init_state);
        mpi::bcast_vector_double(time_vec);
        MPI_Bcast(&is_full_rho, 1, MPI_C_BOOL, mpi::ROOT_ID, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    //std::cout << rank << " QME\n";
    ILP_TYPE ctxt;
    mpi::init_grid(ctxt);

    ILP_TYPE dim, NB, MB;
    ILP_TYPE nrows, ncols;
    ILP_TYPE proc_rows, proc_cols, myrow, mycol;
    //std::cout << rank << " RUNGE_KUTT\n";
    auto localH = mpi::scatter_blacs_matrix<COMPLEX>(H.get_matrix(), dim, dim, NB, MB, nrows, ncols, ctxt, mpi::ROOT_ID);

    //std::cout << rank << " RUNGE_KUTT_1\n";
    State grid;
    if (rank == mpi::ROOT_ID) {
        grid = mpi::bcast_state(H.get_grid());
    } else {
        grid = mpi::bcast_state();
    }

    //std::cout << rank << " RUNGE_KUTT\n";
    //size_t dim = H.size();
    //A.show(config::WIDTH);
    //auto grid = H.get_grid();
    std::vector<std::function<Evolution::Rho(const Evolution::Rho& rho)>> lindblads;

    auto cavities_with_leak = grid.get_cavities_with_leak();
    auto cavities_with_gain = grid.get_cavities_with_gain();

    localH.to_fortran_style();
    ILP_TYPE LLD = localH.LD();
    //auto descg = new ILP_TYPE[9];
    ILP_TYPE rsrc = 0, csrc = 0, info;
    auto descg = mpi::descinit(dim, dim, NB, MB, rsrc, csrc, ctxt, LLD, info);
    //for (const auto& cavity_id: cavities_with_leak) {
    for (size_t cavity_id = 0; cavity_id < grid.cavities_count(); cavity_id++) {
        auto A = create_A_destroy(H.get_basis(), cavity_id);
        //A.show();
        auto gamma = grid.get_leak_gamma(cavity_id);

        auto localA = mpi::scatter_blacs_matrix<COMPLEX>(A, dim, dim, NB, MB, nrows, ncols, ctxt, mpi::ROOT_ID);
        localA.to_fortran_style();
        if (!is_zero(gamma)) {
            //std::cout << gamma << std::endl;
            lindblads.push_back(std::function<Evolution::Rho(const Evolution::Rho& rho)> {
                [localA, gamma, descg, nrows, ncols](const Evolution::Rho& rho) {
                    /*
                    auto Aconj = A.hermit();
                    auto AconjA = Aconj * A;
                    return (A * rho * Aconj - (AconjA * rho + rho * AconjA) * COMPLEX(0.5)) * gamma;
                    */
                    Matrix<COMPLEX> tmp(FORTRAN_STYLE, nrows, ncols);
                    Matrix<COMPLEX> res(FORTRAN_STYLE, nrows, ncols);
                    mpi::parallel_zgemm(localA, rho, tmp, descg, descg, descg);
                    mpi::parallel_zgemm(tmp, localA, res, descg, descg, descg, 'N', 'C');

                    Matrix<COMPLEX> AconjA(FORTRAN_STYLE, nrows, ncols);
                    mpi::parallel_zgemm(localA, localA, AconjA, descg, descg, descg, 'C', 'N');
                    
                    Matrix<COMPLEX> tmp_second(FORTRAN_STYLE, nrows, ncols);
                    mpi::parallel_zgemm(AconjA, rho, tmp, descg, descg, descg);
                    mpi::parallel_zgemm(rho, AconjA, tmp_second, descg, descg, descg);

                    res -= (tmp + tmp_second) * COMPLEX(0.5);
                    res *= gamma;

                    return res;
                }
            }
            );
        }
    }

    //for (const auto& cavity_id: cavities_with_gain) {
    for (size_t cavity_id = 0; cavity_id < grid.cavities_count(); cavity_id++) {
        auto A = create_A_create(H.get_basis(), cavity_id);
        //A.show();
        auto gamma = grid.get_gain_gamma(cavity_id);
        auto localA = mpi::scatter_blacs_matrix<COMPLEX>(A, dim, dim, NB, MB, nrows, ncols, ctxt, mpi::ROOT_ID);
        localA.to_fortran_style();
        if (!is_zero(gamma)) {
            //std::cout << gamma << std::endl;
            lindblads.push_back(std::function<Evolution::Rho(const Evolution::Rho& rho)> {
                [localA, gamma, descg, nrows, ncols](const Evolution::Rho& rho) {
                    /*
                    auto Aconj = A.hermit();
                    auto AconjA = Aconj * A;
                    return (A * rho * Aconj - (AconjA * rho + rho * AconjA) * COMPLEX(0.5)) * gamma;
                    */
                    Matrix<COMPLEX> tmp(FORTRAN_STYLE, nrows, ncols);
                    Matrix<COMPLEX> res(FORTRAN_STYLE, nrows, ncols);
                    mpi::parallel_zgemm(localA, rho, tmp, descg, descg, descg);
                    mpi::parallel_zgemm(tmp, localA, res, descg, descg, descg, 'N', 'C');

                    Matrix<COMPLEX> AconjA(FORTRAN_STYLE, nrows, ncols);
                    mpi::parallel_zgemm(localA, localA, AconjA, descg, descg, descg, 'C', 'N');
                    
                    Matrix<COMPLEX> tmp_second(FORTRAN_STYLE, nrows, ncols);
                    mpi::parallel_zgemm(AconjA, rho, tmp, descg, descg, descg);
                    mpi::parallel_zgemm(rho, AconjA, tmp_second, descg, descg, descg);

                    res -= (tmp + tmp_second) * COMPLEX(0.5);
                    res *= gamma;

                    return res;
                }
            }
            );
        }
    }

    std::function<Evolution::Rho(double t, const Evolution::Rho&)> equation {
        [&localH, &lindblads, &descg, nrows, ncols](double t, const Evolution::Rho& rho) {
            /*
            auto tmp = (H_matrix * rho - rho * H_matrix) * COMPLEX(0, -1);
            for (const auto& lindblad: lindblads) {
                tmp += lindblad(rho);
            }
            */

            Matrix<COMPLEX> tmp(FORTRAN_STYLE, nrows, ncols);
            Matrix<COMPLEX> res(FORTRAN_STYLE, nrows, ncols);
            mpi::parallel_zgemm(localH, rho, res, descg, descg, descg);
            mpi::parallel_zgemm(rho, localH, tmp, descg, descg, descg);
            res -= tmp;
            res *= COMPLEX(0, -1);
            
            for (const auto& lindblad: lindblads) {
                res += lindblad(rho);
            }

            return res;
        }
    };

    Matrix<COMPLEX> rho_0;
    if (rank == mpi::ROOT_ID) rho_0 = Evolution::create_init_rho(init_state);

    auto local_rho_0 = mpi::scatter_blacs_matrix<COMPLEX>(rho_0, dim, dim, NB, MB, nrows, ncols, ctxt, mpi::ROOT_ID);
    local_rho_0.to_fortran_style();
    
    //mpi::print_distributed_matrix<COMPLEX>(local_rho_0, "rho_0", MPI_COMM_WORLD);
    //mpi::print_distributed_matrix<COMPLEX>(localH, "H", MPI_COMM_WORLD);
    auto rho_vec_blocked = Runge_Kutt_4<double, Evolution::Rho>(time_vec, local_rho_0, equation);
    std::vector<Evolution::Rho> rho_vec;
    Evolution::Rho tmp(FORTRAN_STYLE, dim, dim);
    for (const auto& rho : rho_vec_blocked) {
        mpi::gather_blacs_matrix<COMPLEX>(rho, tmp, dim, dim, NB, MB, nrows, ncols, ctxt, mpi::ROOT_ID);
        rho_vec.emplace_back(tmp);
        //if (rank == mpi::ROOT_ID) tmp.show();
    }
// -------------------- stoped here --------------------

    /*
    auto rho_0 = Evolution::create_init_rho(init_state);
    //rho_0.show();
    //auto begin_c = std::chrono::steady_clock::now();
    auto rho_vec = Runge_Kutt_4<double, Evolution::Rho>(time_vec, rho_0, equation);
    //auto end_c = std::chrono::steady_clock::now();
    //std::cout << " c " << std::chrono::duration_cast<std::chrono::milliseconds>(end_c - begin_c).count() << std::endl;
    */

    if (rank == mpi::ROOT_ID) {
        if (!is_full_rho) {
            Evolution::Probs probs(C_STYLE, dim, time_vec.size());

            for (size_t i = 0; i < dim; i++) {
                for (size_t t = 0; t < time_vec.size(); t++) {
                    probs[i][t] = std::abs(rho_vec[t][i][i]);
                }
            }

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

            return probs;
        }

        Evolution::Probs probs(C_STYLE, dim * dim, time_vec.size());

        bool is_null = true;

        for (size_t i = 0; i < dim; i++) {
            for (size_t j = 0; j < dim; j++) {
                for (size_t t = 0; t < time_vec.size(); t++) {
                    probs[i * dim + j][t] = std::abs(rho_vec[t][i][j]);
                    if (probs[i * dim + j][t] >= QConfig::instance().eps()) {
                        is_null = false;
                    }
                }

                if (is_null) probs[i * dim + j][0] = -1;
                is_null = true;
            }
        }

        return probs;
    } else {
        return {};
    }
}

#endif // ENABLE_CLUSTER
#endif // ENABLE_MPI

} // namespace QComputations
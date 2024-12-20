#include "dynamic.hpp"

#include "additional_operators.hpp"
#include "functions.hpp"
#include "mpi_functions.hpp"
#include "plot.hpp"
#include "quantum_operators.hpp"

#ifdef EMABLE_MPI
#ifdef ENABLE_CLUSTER

#include <mkl_pblas.h>
#include <mkl_scalapack.h>

#endif
#endif

namespace QComputations {

    Rho create_init_rho(const std::vector<COMPLEX>& init_state) {
        size_t dim = init_state.size();
        Rho rho(C_STYLE, dim, dim);
        for (size_t i = 0; i < dim; i++) {
            for (size_t j = 0; j < dim; j++) {
                rho[i][j] = init_state[i] * std::conj(init_state[j]);
            }
        }

        return rho;
    }

    Probs exp_evolution(const State<Basis_State>& init_state,
                        Hamiltonian& H,
                        double dt,
                        size_t STEPS_COUNT,
                        const std::function<void(std::vector<COMPLEX>&)>& func) {
        auto state = init_state.fit_to_basis(H.get_basis()).get_vector();
        H.find_exp(dt);
        Probs res(C_STYLE, state.size(), STEPS_COUNT + 1);

        for (size_t i = 0; i < init_state.size(); i++) {
            func(state);
            vector_normalize(state);
            double tmp = std::abs(state[i]);
            res[i][0] = tmp * tmp;
        }

        for (size_t i = 1; i <= STEPS_COUNT; i++) {
            state = H.run_exp(state);
            func(state);
            //vector_normalize(state);

            for (size_t j = 0; j < state.size(); j++) {
                double tmp = std::abs(state[j]);
                res[j][i] = tmp * tmp;
            }
        }

        return res;
    }

    Probs schrodinger(const State<Basis_State>& init_state, Hamiltonian& H, const std::vector<double>& time_vec) {
        std::vector<double> eigen_values;
        Matrix<COMPLEX> eigen_vectors;
        eigen_values = H.eigenvalues();
        eigen_vectors = H.eigenvectors();

        auto init_state_vec = init_state.fit_to_basis(H.get_basis());
        std::vector<COMPLEX> lambda;
        for (size_t i = 0; i < eigen_values.size(); i++) {
            lambda.emplace_back(eigen_vectors.col(i) | init_state_vec.get_vector());  // <PHI_i|KSI(0)>
        }

        Probs probs(C_STYLE, eigen_values.size(), time_vec.size());
        size_t time_index = 0;

        for (const auto& t : time_vec) {
            std::vector<COMPLEX> psi_t(eigen_values.size(), 0);
            std::vector<COMPLEX> tmp(eigen_values.size(), 0);
            auto h = QConfig::instance().h();

            for (size_t i = 0; i < eigen_values.size(); i++) {
                for (size_t j = 0; j < psi_t.size(); j++) {
                    psi_t[j] += lambda[i] * std::exp(COMPLEX(0, -1 / h * eigen_values[i] * t)) * eigen_vectors[j][i];
                }
            }

            for (size_t i = 0; i < eigen_values.size(); i++) {
                double tmp = std::abs(psi_t[i]);
                probs[i][time_index] = tmp * tmp;
            }
            time_index++;
        }

        return probs;
    }

    std::vector<double> scan_gamma(const TCH_State& init_state,
                                   size_t cavity_id,
                                   const std::vector<double>& time_vec,
                                   const std::vector<double>& gamma_vec,
                                   double target) {
        auto cur_state = init_state;
        cur_state.set_leak_for_cavity(cavity_id, gamma_vec[0]);
        H_TCH H(cur_state);

        auto basis = H.get_basis();
        bool zero_state_in_basis = false;
        size_t index = 0;
        for (const auto& state : basis) {
            if (state->is_all_qudits_zero()) {
                zero_state_in_basis = true;
                break;
            }
            index++;
        }

        assert(zero_state_in_basis);
        std::vector<double> tau_vec;
        for (size_t i = 0; i < gamma_vec.size(); i++) {
            cur_state.set_leak_for_cavity(cavity_id, gamma_vec[i]);
            H_TCH H(cur_state);
            auto probs = quantum_master_equation(cur_state, H, time_vec);
            auto func = Cubic_Spline_Interpolate(time_vec, probs.row(index));
            double tau = fsolve(func, time_vec[0], time_vec[time_vec.size() - 1], target);
            tau_vec.emplace_back(tau);
        }

        return tau_vec;
    }

#ifdef ENABLE_MPI
#ifdef ENABLE_CLUSTER

    BLOCKED_Probs schrodinger(const State<Basis_State>& init_state,
                              BLOCKED_Hamiltonian& H,
                              const std::vector<double>& time_vec) {
        std::vector<double> eigen_values = H.eigenvalues();
        BLOCKED_Matrix<COMPLEX> eigen_vectors = H.eigenvectors();

        ILP_TYPE vector_ctxt;
        mpi::init_vector_grid(vector_ctxt);
        std::vector<BLOCKED_Vector<COMPLEX>> vector_of_eigen_vectors =
            blocked_matrix_to_blocked_vectors(vector_ctxt, eigen_vectors);

        auto init_state_vec = init_state.fit_to_basis(H.get_basis());
        BLOCKED_Vector<COMPLEX> blocked_init_state(vector_ctxt, init_state_vec.get_vector());

        std::vector<COMPLEX> lambda;
        for (size_t i = 0; i < eigen_values.size(); i++) {
            lambda.emplace_back(scalar_product(vector_of_eigen_vectors[i], blocked_init_state));  // <PHI_i|KSI(0)>
        }

        auto ctxt = H.ctxt();
        BLOCKED_Probs probs(
            vector_ctxt, GE, eigen_values.size(), time_vec.size(), blocked_init_state.NB(), time_vec.size());
        size_t time_index = 0;

        for (const auto& t : time_vec) {
            BLOCKED_Vector<COMPLEX> psi_t(vector_ctxt, eigen_values.size(), 0, blocked_init_state.NB());
            auto h = QConfig::instance().h();

            for (size_t i = 0; i < eigen_values.size(); i++) {
                psi_t += vector_of_eigen_vectors[i] * lambda[i] * std::exp(COMPLEX(0, -1 / h * eigen_values[i] * t));
            }

            for (size_t i = 0; i < probs.local_n(); i++) {
                double tmp = std::abs(psi_t[i]);
                probs(i, time_index) = tmp * tmp;
            }
            time_index++;
        }

        return probs;
    }

#endif
#endif

    Probs quantum_master_equation(const State<Basis_State>& init_state,
                                  Hamiltonian& H,
                                  const std::vector<double>& time_vec,
                                  bool is_full_rho) {
        size_t dim = H.size();
        std::vector<std::function<void(const Rho& rho)>> lindblads;

        Matrix<COMPLEX> T1(C_STYLE, dim, dim);
        Matrix<COMPLEX> T2(C_STYLE, dim, dim);

        auto H_matrix = H.get_matrix();

        for (const auto& p : H.get_decoherence()) {
            auto gamma = p.first;
            const Matrix<COMPLEX>& A = p.second;
            lindblads.push_back(std::function<void(const Rho& rho)>{[A, &T1, &T2, gamma](const Rho& rho) {
                optimized_multiply(A, A, T1, COMPLEX(1, 0), COMPLEX(0, 0), 'C', 'N');  // AconjA -> T1
                optimized_multiply(T1, rho, T2, COMPLEX(1, 0), COMPLEX(0, 0));         // AconjA*rho -> T2
                optimized_multiply(rho, T1, T2, COMPLEX(1, 0), COMPLEX(1, 0));         // rho * AconjA + AconjA * rho
                optimized_multiply(A, rho, T1, COMPLEX(1, 0), COMPLEX(0, 0));          // A*rho -> T1
                optimized_multiply(T1, A, T2, COMPLEX(gamma, 0), COMPLEX(-0.5 * gamma, 0), 'N', 'C');  // res -> T2
            }});
        }

        std::function<void(double t, const Rho&, Matrix<COMPLEX>&)> equation{
            [&H_matrix, &T1, &T2, &lindblads](double t, const Rho& rho, Matrix<COMPLEX>& res) {
                optimized_multiply(rho, H_matrix, res, COMPLEX(1, 0), COMPLEX(0, 0));  // rho * H_matrix -> res
                optimized_multiply(H_matrix,
                                   rho,
                                   res,
                                   COMPLEX(0, -1 / QConfig::instance().h()),
                                   COMPLEX(0, 1 / QConfig::instance().h()));  // result -> res

                for (const auto& lindblad : lindblads) {
                    lindblad(rho);
                    T2 /= QConfig::instance().h();
                    res += T2;
                }
            }};

        auto rho_0 = create_init_rho(init_state.fit_to_basis(H.get_basis()).get_vector());
        std::vector<Rho> rho_vec;
        if (QConfig::instance().qme_algorithm() == RUNGE_KUTT_4) {
            rho_vec = OPT_Runge_Kutt_4(time_vec, rho_0, equation);
        } else if (QConfig::instance().qme_algorithm() == RUNGE_KUTT_2) {
            rho_vec = OPT_Runge_Kutt_2(time_vec, rho_0, equation);
        } else {
            assert(false);  // Неизвестный алгоритм решения ОКУ
        }

        Probs probs(C_STYLE, dim, time_vec.size());

        for (size_t i = 0; i < dim; i++) {
            for (size_t t = 0; t < time_vec.size(); t++) {
                probs[i][t] = std::abs(rho_vec[t][i][i]);
            }
        }

        return probs;
    }

    std::pair<Probs, BasisType<Basis_State>> probs_to_cavity_probs(const Probs& probs,
                                                                   const BasisType<Basis_State>& basis,
                                                                   size_t cavity_id) {
        BasisType<Basis_State> basis_res;
        for (auto cur_state : basis) {
            basis_res.insert(std::shared_ptr<Basis_State>(new Basis_State(cur_state->get_group(cavity_id))));
        }

        size_t m = probs.m();
        Probs res(C_STYLE, basis_res.size(), m, double(0));

        for (size_t t = 0; t < probs.m(); t++) {
            for (size_t i = 0; i < probs.n(); i++) {
                size_t res_index =
                    Basis_State(*get_state_from_basis(basis, i)).get_group(cavity_id).get_index(basis_res);
                res[res_index][t] += probs.elem(i, t);
            }
        }

        return std::make_pair(res, basis_res);
    }

#ifdef ENABLE_MPI
#ifdef ENABLE_CLUSTER

    std::pair<BLOCKED_Probs, BasisType<Basis_State>> probs_to_cavity_probs(const BLOCKED_Probs& probs,
                                                                           const BasisType<Basis_State>& basis,
                                                                           size_t cavity_id) {
        ILP_TYPE rank, world_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Status status;

        ILP_TYPE proc_rows, proc_cols, myrow, mycol;
        mpi::blacs_gridinfo(probs.ctxt(), proc_rows, proc_cols, myrow, mycol);

        BasisType<Basis_State> basis_res;
        for (auto cur_state : basis) {
            basis_res.insert(std::shared_ptr<Basis_State>(new Basis_State(cur_state->get_group(cavity_id))));
        }

        std::vector<size_t> inplace_states(basis_res.size(), 0);
        for (auto cur_state : basis) {
            size_t res_index = cur_state->get_group(cavity_id).get_index(basis_res);
            inplace_states[res_index] += 1;
        }

        size_t NB = basis_res.size() / world_size;

        if (basis_res.size() < world_size) {
            if (rank < basis_res.size()) {
                NB = 1;
            }
        }

        BLOCKED_Probs res(probs.ctxt(), GE, basis_res.size(), probs.m(), 0, NB, probs.MB());
        std::vector<MPI_Request> requests_send;
        std::vector<std::vector<double>> buf;

        for (size_t i_local = 0; i_local < probs.local_n(); i_local++) {
            auto i = probs.get_global_row(i_local);
            size_t res_index = Basis_State(*get_state_from_basis(basis, i)).get_group(cavity_id).get_index(basis_res);
            auto proc_row = res.get_row_proc(res_index);
            auto res_local_row = res.get_local_row(res_index);

            if (proc_row == myrow) {
                inplace_states[res_index] -= 1;
                for (size_t t = 0; t < probs.m(); t++) {
                    res(res_local_row, t) += probs(i_local, t);
                }
            } else {
                requests_send.emplace_back(MPI_REQUEST_NULL);
                buf.emplace_back(probs.m());

                for (size_t t = 0; t < probs.m(); t++) {
                    buf[buf.size() - 1][t] = probs(i_local, t);
                }

                MPI_Isend(buf[buf.size() - 1].data(),
                          probs.m(),
                          MPI_DOUBLE,
                          proc_row,
                          res_index,
                          MPI_COMM_WORLD,
                          &requests_send[requests_send.size() - 1]);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        std::vector<double> res_index_probs(probs.m());
        for (size_t i = 0; i < res.local_n(); i++) {
            size_t i_global = res.get_global_row(i);

            for (size_t k = 0; k < inplace_states[i_global]; k++) {
                MPI_Recv(res_index_probs.data(),
                         probs.m(),
                         MPI_DOUBLE,
                         MPI_ANY_SOURCE,
                         i_global,
                         MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);

                for (size_t t = 0; t < probs.m(); t++) {
                    res(i, t) += res_index_probs[t];
                }
            }
        }

        for (auto& request : requests_send) {
            MPI_Wait(&request, MPI_STATUS_IGNORE);
        }

        return std::make_pair(res, basis_res);
    }

    BLOCKED_Rho create_BLOCKED_init_rho(ILP_TYPE ctxt, const std::vector<COMPLEX>& init_state) {
        std::function<COMPLEX(size_t i, size_t j)> func = {
            [&init_state](size_t i, size_t j) { return init_state[i] * std::conj(init_state[j]); }};

        size_t dim = init_state.size();
        BLOCKED_Rho rho(ctxt, GE, dim, dim, func);

        return rho;
    }

    BLOCKED_Matrix<COMPLEX> create_BLOCKED_A_destroy(ILP_TYPE ctxt,
                                                     const BasisType<Basis_State>& basis,
                                                     size_t cavity_id) {
        size_t dim = basis.size();

        std::function<COMPLEX(size_t i, size_t j)> func = {[&basis, &cavity_id](size_t i, size_t j) {
            auto state_from = get_state_from_basis(basis, j);
            auto state_to = get_state_from_basis(basis, i);

            return COMPLEX(0);
        }};

        BLOCKED_Matrix<COMPLEX> A(ctxt, GE, dim, dim, func);

        return A;
    }

    BLOCKED_Matrix<COMPLEX> create_BLOCKED_A_create(ILP_TYPE ctxt,
                                                    const BasisType<Basis_State>& basis,
                                                    size_t cavity_id) {
        size_t dim = basis.size();

        std::function<COMPLEX(size_t i, size_t j)> func = {[&basis, &cavity_id](size_t i, size_t j) {
            auto state_from = get_state_from_basis(basis, j);
            auto state_to = get_state_from_basis(basis, i);

            return COMPLEX(0);
        }};

        BLOCKED_Matrix<COMPLEX> A(ctxt, GE, dim, dim, func);

        return A;
    }

    BLOCKED_Probs quantum_master_equation(const State<Basis_State>& init_state,
                                          BLOCKED_Hamiltonian& H,
                                          const std::vector<double>& time_vec,
                                          bool is_full_rho) {
        size_t dim = H.size();
        std::vector<std::function<void(const BLOCKED_Rho& rho)>> lindblads;

        BLOCKED_Matrix<COMPLEX> T1(H.ctxt(), GE, dim, dim);
        BLOCKED_Matrix<COMPLEX> T2(H.ctxt(), GE, dim, dim);
        auto H_matrix = H.get_blocked_matrix();

        for (const auto& p : H.get_decoherence()) {
            auto gamma = p.first;

            const BLOCKED_Matrix<COMPLEX>& A = p.second;
            lindblads.push_back(
                std::function<void(const BLOCKED_Rho& rho)>{[&H_matrix, A, &T1, &T2, gamma](const BLOCKED_Rho& rho) {
                    optimized_multiply(A, A, T1, COMPLEX(1, 0), COMPLEX(0, 0), 'C', 'N');  // AconjA -> T1
                    optimized_multiply(T1, rho, T2, COMPLEX(1, 0), COMPLEX(0, 0));         // AconjA*rho -> T2
                    optimized_multiply(rho, T1, T2, COMPLEX(1, 0), COMPLEX(1, 0));  // rho * AconjA + AconjA * rho
                    optimized_multiply(A, rho, T1, COMPLEX(1, 0), COMPLEX(0, 0));   // A*rho -> T1
                    optimized_multiply(T1, A, T2, COMPLEX(gamma, 0), COMPLEX(-0.5 * gamma, 0), 'N', 'C');  // res -> T2
                }});
        }

        std::function<void(double t, const BLOCKED_Rho&, BLOCKED_Matrix<COMPLEX>&)> equation{
            [&H_matrix, &T1, &T2, &lindblads](double t, const BLOCKED_Rho& rho, BLOCKED_Matrix<COMPLEX>& res) {
                optimized_multiply(rho, H_matrix, res, COMPLEX(1, 0), COMPLEX(0, 0));  // rho * H_matrix -> res
                optimized_multiply(H_matrix,
                                   rho,
                                   res,
                                   COMPLEX(0, -1 / QConfig::instance().h()),
                                   COMPLEX(0, 1 / QConfig::instance().h()));  // result -> res

                for (const auto& lindblad : lindblads) {
                    lindblad(rho);
                    optimized_add(T2, res, COMPLEX(1 / QConfig::instance().h(), 0), COMPLEX(1, 0));
                }
            }};

        BLOCKED_Matrix<COMPLEX> rho_0(
            create_BLOCKED_init_rho(H.ctxt(), init_state.fit_to_basis(H.get_basis()).get_vector()));
        std::vector<BLOCKED_Rho> rho_vec;
        if (QConfig::instance().qme_algorithm() == RUNGE_KUTT_4) {
            rho_vec = MPI_Runge_Kutt_4(time_vec, rho_0, equation);
        } else if (QConfig::instance().qme_algorithm() == RUNGE_KUTT_2) {
            rho_vec = MPI_Runge_Kutt_2(time_vec, rho_0, equation);
        } else {
            assert(false);  // Неизвестный алгоритм решения ОКУ
        }

        ILP_TYPE world_size, rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        ILP_TYPE probs_ctxt;
        mpi::init_grid(probs_ctxt, world_size, 1);
        ILP_TYPE proc_rows, proc_cols, myrow, mycol;
        mpi::blacs_gridinfo(probs_ctxt, proc_rows, proc_cols, myrow, mycol);
        BLOCKED_Probs probs(probs_ctxt,
                            GE,
                            dim,
                            time_vec.size(),
                            (H.size() >= world_size ? H.size() / world_size : 1),
                            time_vec.size());

        for (size_t t = 0; t < time_vec.size(); t++) {
            auto probs_vec = mpi::get_diagonal_elements<COMPLEX>(rho_vec[t].get_local_matrix(), rho_vec[t].desc());
            for (size_t i = 0; i < probs.local_n(); i++) {
                probs(i, t) = std::abs(probs_vec[probs.get_global_row(i)]);
            }
        }

        return probs;
    }

    BLOCKED_Probs quantum_master_equation(const std::vector<COMPLEX>& init_state,
                                          BLOCKED_Hamiltonian& H,
                                          const std::vector<double>& time_vec,
                                          bool is_full_rho) {
        size_t dim = H.size();
        auto grid = H.get_grid();
        std::vector<std::function<BLOCKED_Rho(const BLOCKED_Rho& rho)>> lindblads;

        auto cavities_with_leak = grid.get_cavities_with_leak();
        auto cavities_with_gain = grid.get_cavities_with_gain();

        for (size_t cavity_id = 0; cavity_id < grid.cavities_count(); cavity_id++) {
            auto gamma = grid.get_leak_gamma(cavity_id);
            if (!is_zero(gamma)) {
                auto A = create_BLOCKED_A_destroy(H.ctxt(), H.get_basis(), cavity_id);
                lindblads.push_back(
                    std::function<BLOCKED_Rho(const BLOCKED_Rho& rho)>{[A, gamma](const BLOCKED_Rho& rho) {
                        auto Aconj = A.hermit();
                        auto AconjA = Aconj * A;
                        return (A * rho * Aconj - (AconjA * rho + rho * AconjA) * COMPLEX(0.5)) * gamma;
                    }});
            }
        }

        for (size_t cavity_id = 0; cavity_id < grid.cavities_count(); cavity_id++) {
            auto gamma = grid.get_gain_gamma(cavity_id);
            if (!is_zero(gamma)) {
                auto A = create_BLOCKED_A_create(H.ctxt(), H.get_basis(), cavity_id);
                lindblads.push_back(
                    std::function<BLOCKED_Rho(const BLOCKED_Rho& rho)>{[A, gamma](const BLOCKED_Rho& rho) {
                        auto Aconj = A.hermit();
                        auto AconjA = Aconj * A;
                        return (A * rho * Aconj - (AconjA * rho + rho * AconjA) * COMPLEX(0.5)) * gamma;
                    }});
            }
        }

        auto H_matrix = H.get_blocked_matrix();
        std::function<BLOCKED_Rho(double t, const BLOCKED_Rho&)> equation{
            [&H_matrix, &lindblads](double t, const BLOCKED_Rho& rho) {
                auto tmp = (H_matrix * rho - rho * H_matrix) * COMPLEX(0, -1);
                for (const auto& lindblad : lindblads) {
                    tmp += lindblad(rho);
                }

                return tmp;
            }};

        auto rho_0 = create_BLOCKED_init_rho(H.ctxt(), init_state);
        auto rho_vec = Runge_Kutt_4<double, BLOCKED_Rho>(time_vec, rho_0, equation);

        if (!is_full_rho) {
            ILP_TYPE world_size, rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &world_size);
            ILP_TYPE probs_ctxt;
            mpi::init_grid(probs_ctxt, world_size, 1);
            ILP_TYPE proc_rows, proc_cols, myrow, mycol;
            mpi::blacs_gridinfo(probs_ctxt, proc_rows, proc_cols, myrow, mycol);
            BLOCKED_Probs probs(probs_ctxt,
                                GE,
                                dim,
                                time_vec.size(),
                                (H.size() >= world_size ? H.size() / world_size : 1),
                                time_vec.size());

            for (size_t t = 0; t < time_vec.size(); t++) {
                auto probs_vec = mpi::get_diagonal_elements<COMPLEX>(rho_vec[t].get_local_matrix(), rho_vec[t].desc());
                for (size_t i = 0; i < probs.local_n(); i++) {
                    probs(i, t) = std::abs(probs_vec[probs.get_global_row(i)]);
                }
            }

            return probs;
        }

        ILP_TYPE world_size, rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        return BLOCKED_Probs(
            H.ctxt(), GE, dim, time_vec.size(), (H.size() >= world_size ? H.size() / world_size : 1), time_vec.size());
    }

#endif  // ENABLE_CLUSTER
#endif  // ENABLE_MPI

}  // namespace QComputations
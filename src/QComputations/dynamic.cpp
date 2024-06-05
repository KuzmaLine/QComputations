#include "dynamic.hpp"
#include "additional_operators.hpp"
#include "functions.hpp"
#include "quantum_operators.hpp"
#include "mpi_functions.hpp"

#include "plot.hpp"

#ifdef EMABLE_MPI
#ifdef ENABLE_CLUSTER

#include <mkl_pblas.h>
#include <mkl_scalapack.h>

#endif
#endif

namespace QComputations {


Matrix<COMPLEX> Evolution::create_A_destroy(const std::set<Basis_State>& basis, size_t cavity_id) {
    size_t dim = basis.size();
    Matrix<COMPLEX> A(DEFAULT_MATRIX_STYLE, dim, dim, COMPLEX(0));

    size_t index = 0;
    for (const auto& state: basis) {
        auto n = 0;
        if (n != 0) {
            auto tmp_state = state;
            auto state_index = 0;

            if (state_index != -1) A[state_index][index] = COMPLEX(std::sqrt(n));
        }

        index++;
    }

    return A;
}

Matrix<COMPLEX> Evolution::create_A_create(const std::set<Basis_State>& basis, size_t cavity_id) {
    size_t dim = basis.size();
    Matrix<COMPLEX> A(DEFAULT_MATRIX_STYLE, dim, dim, COMPLEX(0));

    size_t index = 0;
    for (const auto& state: basis) {
        auto n = 0;
        auto tmp_state = state;

        auto state_index = 0;
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



Evolution::Probs Evolution::schrodinger(const State<Basis_State>& init_state, Hamiltonian& H, const std::vector<double>& time_vec) {
    std::vector<double> eigen_values;
    Matrix<COMPLEX> eigen_vectors;
    eigen_values = H.eigenvalues();
    eigen_vectors = H.eigenvectors();

    std::vector<COMPLEX> lambda;
    for (size_t i = 0; i < eigen_values.size(); i++) {
        lambda.emplace_back(eigen_vectors.col(i) | init_state.get_vector()); // <PHI_i|KSI(0)> 
    }

    Probs probs(C_STYLE, eigen_values.size(), time_vec.size());
    size_t time_index = 0;

    for (const auto& t: time_vec) {
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

#ifdef ENABLE_MPI
#ifdef ENABLE_CLUSTER

Evolution::BLOCKED_Probs Evolution::schrodinger(const State<Basis_State>& init_state, BLOCKED_Hamiltonian& H,
                                                const std::vector<double>& time_vec) {
    auto eigen_values = H.eigenvalues();
    auto eigen_vectors = H.eigenvectors();

    ILP_TYPE vector_ctxt;
    mpi::init_vector_grid(vector_ctxt);
    auto vector_of_eigen_vectors = blocked_matrix_to_blocked_vectors(vector_ctxt, eigen_vectors);
    BLOCKED_Vector<COMPLEX> blocked_init_state(vector_ctxt, init_state.get_vector());

    std::vector<COMPLEX> lambda;
    for (size_t i = 0; i < eigen_values.size(); i++) {
        lambda.emplace_back(scalar_product(blocked_init_state, blocked_matrix_get_col(blocked_init_state.ctxt(), eigen_vectors, i))); // <PHI_i|KSI(0)> 
    }

    auto ctxt = H.ctxt();
    Evolution::BLOCKED_Probs probs(vector_ctxt, GE, eigen_values.size(), time_vec.size(), blocked_init_state.NB(), time_vec.size());
    size_t time_index = 0;

    for (const auto& t: time_vec) {
        BLOCKED_Vector<COMPLEX> psi_t(vector_ctxt, eigen_values.size(), 0, blocked_init_state.NB());
        auto h = QConfig::instance().h();

        for (size_t i = 0; i < eigen_values.size(); i++) {
            psi_t += vector_of_eigen_vectors[i] *  lambda[i] * std::exp(COMPLEX(0, -1 / h * eigen_values[i] * t));
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

Evolution::Probs Evolution::quantum_master_equation(const State<Basis_State>& init_state,
                            Hamiltonian& H,
                            const std::vector<double>& time_vec,
                            bool is_full_rho) {
    size_t dim = H.size();
    std::vector<std::function<Evolution::Rho(const Evolution::Rho& rho)>> lindblads;

    for (const auto& p: H.get_decoherence()) {
        auto gamma = p.first;
        auto A = p.second;

        lindblads.push_back(std::function<Evolution::Rho(const Evolution::Rho& rho)> {
            [A, gamma](const Evolution::Rho& rho) {
                auto Aconj = A.hermit();
                auto AconjA = Aconj * A;
                return (A * rho * Aconj - (AconjA * rho + rho * AconjA) * COMPLEX(0.5, 0)) * gamma;
            }
        }
        );
    }

    auto H_matrix = H.get_matrix();
    std::function<Evolution::Rho(double t, const Evolution::Rho&)> equation {[&H_matrix, &lindblads](double t, const Evolution::Rho& rho) {
        auto tmp = (H_matrix * rho - rho * H_matrix) * COMPLEX(0, -1 / QConfig::instance().h());
        for (const auto& lindblad: lindblads) {
            tmp += lindblad(rho);
        }

        return tmp;
    }};

    auto rho_0 = Evolution::create_init_rho(init_state.get_vector());
    auto rho_vec = Runge_Kutt_4<double, Evolution::Rho>(time_vec, rho_0, equation);

    Evolution::Probs probs(C_STYLE, dim, time_vec.size());

    for (size_t i = 0; i < dim; i++) {
        for (size_t t = 0; t < time_vec.size(); t++) {
            probs[i][t] = std::abs(rho_vec[t][i][i]);
        }
    }

    return probs;
}

std::pair<Evolution::Probs, std::set<Basis_State>> Evolution::probs_to_cavity_probs(const Evolution::Probs& probs,
                                                          const std::set<Basis_State>& basis, size_t cavity_id) {
    std::set<Basis_State> basis_res;
    for (auto& cur_state: basis) {
        basis_res.insert(cur_state.get_group(cavity_id));
    }

    size_t m = probs.m();
    Probs res(C_STYLE, basis_res.size(), m, double(0));

    for (size_t t = 0; t < probs.m(); t++) {
        for (size_t i = 0; i < probs.n(); i++) {
            size_t res_index = Basis_State(get_elem_from_set(basis, i)).get_group(cavity_id).get_index(basis_res);
            res[res_index][t] += probs.elem(i, t);
        }
    }

    return std::make_pair(res, basis_res);
}

#ifdef ENABLE_MPI
#ifdef ENABLE_CLUSTER


 std::pair<Evolution::BLOCKED_Probs, std::set<Basis_State>> Evolution::probs_to_cavity_probs(const Evolution::BLOCKED_Probs& probs,
                                                          const std::set<Basis_State>& basis, size_t cavity_id) {
    ILP_TYPE rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Status status;

    ILP_TYPE proc_rows, proc_cols, myrow, mycol;
    mpi::blacs_gridinfo(probs.ctxt(), proc_rows, proc_cols, myrow, mycol);

    std::set<Basis_State> basis_res;
    for (auto& cur_state: basis) {
        basis_res.insert(cur_state.get_group(cavity_id));
    }

    std::vector<size_t> inplace_states(basis_res.size(), 0);
    for (auto& cur_state: basis) {
        size_t res_index = cur_state.get_group(cavity_id).get_index(basis_res);
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
        size_t res_index = Basis_State(get_elem_from_set(basis, i)).get_group(cavity_id).get_index(basis_res);
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
    
            MPI_Isend(buf[buf.size() - 1].data(), probs.m(), MPI_DOUBLE, proc_row, res_index, MPI_COMM_WORLD, &requests_send[requests_send.size() - 1]);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<double> res_index_probs(probs.m());
    for (size_t i = 0; i < res.local_n(); i++) {
        size_t i_global = res.get_global_row(i);

        for (size_t k = 0; k < inplace_states[i_global]; k++) {
            MPI_Recv(res_index_probs.data(), probs.m(), MPI_DOUBLE, MPI_ANY_SOURCE, i_global, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for (size_t t = 0; t < probs.m(); t++) {
                res(i, t) += res_index_probs[t];
            }
        }
    }

    for (auto& request: requests_send) {
        MPI_Wait(&request, MPI_STATUS_IGNORE);
    }

    return std::make_pair(res, basis_res);
}


Evolution::BLOCKED_Rho Evolution::create_BLOCKED_init_rho(ILP_TYPE ctxt, const std::vector<COMPLEX>& init_state) {
    std::function<COMPLEX(size_t i, size_t j)> func = {
        [&init_state](size_t i, size_t j) { return init_state[i] * std::conj(init_state[j]); }};
    
    size_t dim = init_state.size();
    Evolution::BLOCKED_Rho rho(ctxt, GE, dim, dim, func);

    return rho;
}

BLOCKED_Matrix<COMPLEX> create_BLOCKED_A_destroy(ILP_TYPE ctxt, const std::set<Basis_State>& basis, size_t cavity_id) {
    size_t dim = basis.size();

    std::function<COMPLEX(size_t i, size_t j)> func = {
    [&basis, &cavity_id](size_t i, size_t j) {
        auto state_from = get_elem_from_set(basis, j);
        auto state_to = get_elem_from_set(basis, i);

        return COMPLEX(0);
    }};

    BLOCKED_Matrix<COMPLEX> A(ctxt, GE, dim, dim, func);

    return A;
}

BLOCKED_Matrix<COMPLEX> create_BLOCKED_A_create(ILP_TYPE ctxt, const std::set<Basis_State>& basis, size_t cavity_id) {
    size_t dim = basis.size();

    std::function<COMPLEX(size_t i, size_t j)> func = {
    [&basis, &cavity_id](size_t i, size_t j) {
        auto state_from = get_elem_from_set(basis, j);
        auto state_to = get_elem_from_set(basis, i);

        return COMPLEX(0);
    }};

    BLOCKED_Matrix<COMPLEX> A(ctxt, GE, dim, dim, func);

    return A;
}

Evolution::BLOCKED_Probs Evolution::quantum_master_equation(const State<Basis_State>& init_state,
                            BLOCKED_Hamiltonian& H,
                            const std::vector<double>& time_vec,
                            bool is_full_rho) {
    size_t dim = H.size();
    std::vector<std::function<Evolution::BLOCKED_Rho(const Evolution::BLOCKED_Rho& rho)>> lindblads;

    for (const auto& p: H.get_decoherence()) {
        auto gamma = p.first;
        auto A = p.second;

        lindblads.push_back(std::function<Evolution::BLOCKED_Rho(const Evolution::BLOCKED_Rho& rho)> {
            [A, gamma](const Evolution::BLOCKED_Rho& rho) {
                auto Aconj = A.hermit();
                auto AconjA = Aconj * A;
                return (A * rho * Aconj - (AconjA * rho + rho * AconjA) * COMPLEX(0.5)) * gamma;
            }
        }
        );
    }

    auto H_matrix = H.get_blocked_matrix();
    std::function<Evolution::BLOCKED_Rho(double t, const Evolution::BLOCKED_Rho&)> equation {[&H_matrix, &lindblads](double t, const Evolution::BLOCKED_Rho& rho) {
        auto tmp = (H_matrix * rho - rho * H_matrix) * COMPLEX(0, -1);
        for (const auto& lindblad: lindblads) {
            tmp += lindblad(rho);
        }

        return tmp;
    }};

    auto rho_0 = Evolution::create_BLOCKED_init_rho(H.ctxt(), init_state.get_vector());
    auto rho_vec = Runge_Kutt_4<double, Evolution::BLOCKED_Rho>(time_vec, rho_0, equation);

    ILP_TYPE world_size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    ILP_TYPE probs_ctxt;
    mpi::init_grid(probs_ctxt, world_size, 1);
    ILP_TYPE proc_rows, proc_cols, myrow, mycol;
    mpi::blacs_gridinfo(probs_ctxt, proc_rows, proc_cols, myrow, mycol);
    Evolution::BLOCKED_Probs probs(probs_ctxt, GE, dim, time_vec.size(), (H.size() >= world_size ? H.size() / world_size : 1), time_vec.size());

    for (size_t t = 0; t < time_vec.size(); t++) {
        auto probs_vec = mpi::get_diagonal_elements<COMPLEX>(rho_vec[t].get_local_matrix(), rho_vec[t].desc());

        for (size_t i = 0; i < probs.local_n(); i++) {   
            probs(i, t) = std::abs(probs_vec[probs.get_global_row(i)]);
        }
    }

    return probs;
}

Evolution::BLOCKED_Probs Evolution::quantum_master_equation(const std::vector<COMPLEX>& init_state,
                                BLOCKED_Hamiltonian& H,
                                const std::vector<double>& time_vec,
                                bool is_full_rho) {
    size_t dim = H.size();
    auto grid = H.get_grid();
    std::vector<std::function<Evolution::BLOCKED_Rho(const Evolution::BLOCKED_Rho& rho)>> lindblads;

    auto cavities_with_leak = grid.get_cavities_with_leak();
    auto cavities_with_gain = grid.get_cavities_with_gain();

    for (size_t cavity_id = 0; cavity_id < grid.cavities_count(); cavity_id++) {
        auto gamma = grid.get_leak_gamma(cavity_id);

        if (!is_zero(gamma)) {
            auto A = create_BLOCKED_A_destroy(H.ctxt(), H.get_basis(), cavity_id);
            lindblads.push_back(std::function<Evolution::BLOCKED_Rho(const Evolution::BLOCKED_Rho& rho)> {
                [A, gamma](const Evolution::BLOCKED_Rho& rho) {
                    auto Aconj = A.hermit();
                    auto AconjA = Aconj * A;
                    return (A * rho * Aconj - (AconjA * rho + rho * AconjA) * COMPLEX(0.5)) * gamma;
                }
            }
            );
        }
    }

    for (size_t cavity_id = 0; cavity_id < grid.cavities_count(); cavity_id++) {
        auto gamma = grid.get_gain_gamma(cavity_id);

        if (!is_zero(gamma)) {
            auto A = create_BLOCKED_A_create(H.ctxt(), H.get_basis(), cavity_id);
            lindblads.push_back(std::function<Evolution::BLOCKED_Rho(const Evolution::BLOCKED_Rho& rho)> {
                [A, gamma](const Evolution::BLOCKED_Rho& rho) {
                    auto Aconj = A.hermit();
                    auto AconjA = Aconj * A;
                    return (A * rho * Aconj - (AconjA * rho + rho * AconjA) * COMPLEX(0.5)) * gamma;
                }
            }
            );
        }
    }

    auto H_matrix = H.get_blocked_matrix();
    std::function<Evolution::BLOCKED_Rho(double t, const Evolution::BLOCKED_Rho&)> equation {[&H_matrix, &lindblads](double t, const Evolution::BLOCKED_Rho& rho) {
        auto tmp = (H_matrix * rho - rho * H_matrix) * COMPLEX(0, -1);
        for (const auto& lindblad: lindblads) {
            tmp += lindblad(rho);
        }

        return tmp;
    }};

    auto rho_0 = Evolution::create_BLOCKED_init_rho(H.ctxt(), init_state);
    auto rho_vec = Runge_Kutt_4<double, Evolution::BLOCKED_Rho>(time_vec, rho_0, equation);

    if (!is_full_rho) {
        ILP_TYPE world_size, rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        ILP_TYPE probs_ctxt;
        mpi::init_grid(probs_ctxt, world_size, 1);
        ILP_TYPE proc_rows, proc_cols, myrow, mycol;
        mpi::blacs_gridinfo(probs_ctxt, proc_rows, proc_cols, myrow, mycol);
        Evolution::BLOCKED_Probs probs(probs_ctxt, GE, dim, time_vec.size(), (H.size() >= world_size ? H.size() / world_size : 1), time_vec.size());

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
    return Evolution::BLOCKED_Probs(H.ctxt(), GE, dim, time_vec.size(), (H.size() >= world_size ? H.size() / world_size : 1), time_vec.size());
}

#endif // ENABLE_CLUSTER
#endif // ENABLE_MPI

} // namespace QComputations
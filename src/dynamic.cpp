#include "dynamic.hpp"
#include "additional_operators.hpp"
#include "functions.hpp"
#include "quantum_operators.hpp"

#include "plot.hpp"

Matrix<COMPLEX> Evolution::create_A_destroy(const std::set<State>& basis) {
    size_t dim = basis.size();
    Matrix<COMPLEX> A(dim, dim, 0);

    size_t index = 0;
    for (const auto& state: basis) {
        auto n = state.n();
        if (n != 0) {
            auto tmp_state = state;
            tmp_state.set_n(n - 1);

            auto state_index = tmp_state.get_index(basis);
            if (state_index != -1) A[state_index][index] = COMPLEX(1);
        }

        index++;
    }

    return A;
}

Evolution::Rho Evolution::create_init_rho(const std::vector<COMPLEX>& init_state) {
    size_t dim = init_state.size();
    Evolution::Rho rho(dim, dim);
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

    auto p = H.eigen();
    auto eigen_values = p.first;
    auto eigen_vectors = p.second;
    size_t n = eigen_values.size();

    std::vector<COMPLEX> lambda;
    for (size_t i = 0; i < eigen_values.size(); i++) {
        //std::cout << norm(eigen_vectors.col(i)) << std::endl;
        lambda.emplace_back(init_state | eigen_vectors.col(i)); // <KSI(0)|PHI_i> 
    }

    //std::cout << "L - " << norm(lambda) << std::endl;
    Probs probs(eigen_values.size(), time_vec.size());
    size_t time_index = 0;

    eigen_vectors = eigen_vectors.transpose();
    for (const auto& t: time_vec) {
        std::vector<COMPLEX> psi_t(eigen_values.size(), 0);

        for (size_t i = 0; i < eigen_values.size(); i++) {
            for (size_t j = 0; j < psi_t.size(); j++) {
                psi_t[j] += lambda[i] * std::exp(COMPLEX(0, 1 / config::h * eigen_values[i] * t)) * eigen_vectors[i][j];
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

// ON MPI NEEDED
Evolution::Probs Evolution::quantum_master_equation(const std::vector<COMPLEX>& init_state,
                                         Hamiltonian& H,
                                         const std::vector<double>& time_vec,
                                         const double gamma,
                                         bool is_full_rho) {
    size_t dim = H.size();
    Evolution::Rho rho(dim, dim, 0);

    auto A = create_A_destroy(H.get_basis());

    std::function<Evolution::Rho(const Evolution::Rho& rho)> lindblad {
        [&A, gamma](const Evolution::Rho& rho) {
            auto Aconj = A.hermit();
            auto AconjA = Aconj * A;
            return (A * rho * Aconj - (AconjA * rho + rho * AconjA) * COMPLEX(0.5)) * gamma;
        }
    };

    std::function<Evolution::Rho(double t, const Evolution::Rho&)> equation {[&H, &A, &lindblad, gamma](double t, const Evolution::Rho& rho) {
        return (H.get_matrix() * rho - rho * H.get_matrix()) * COMPLEX(0, -1) + lindblad(rho);
    }};

    auto rho_0 = Evolution::create_init_rho(init_state);
    //auto begin_c = std::chrono::steady_clock::now();
    auto rho_vec = Runge_Kutt_4<double, Evolution::Rho>(time_vec, rho_0, equation);
    //auto end_c = std::chrono::steady_clock::now();
    //std::cout << " c " << std::chrono::duration_cast<std::chrono::milliseconds>(end_c - begin_c).count() << std::endl;
    if (!is_full_rho) {
        Evolution::Probs probs(dim, time_vec.size());

        for (size_t i = 0; i < dim; i++) {
            for (size_t t = 0; t < time_vec.size(); t++) {
                probs[i][t] = std::abs(rho_vec[t][i][i]);
            }
        }

        return probs;
    }

    Evolution::Probs probs(dim * dim, time_vec.size());

    bool is_null = true;

    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            for (size_t t = 0; t < time_vec.size(); t++) {
                probs[i * dim + j][t] = std::abs(rho_vec[t][i][j]);
                if (probs[i * dim + j][t] >= config::eps) {
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
        //begin = std::chrono::steady_clock::now();
        auto probs = quantum_master_equation(init_state, H, time_vec, gamma);
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


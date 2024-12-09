#pragma once
#include <memory>
#include <vector>

#include "hamiltonian.hpp"
#include "state.hpp"

#ifdef ENABLE_MPI
#ifdef ENABLE_CLUSTER

#include "blocked_vector.hpp"
#include "hamiltonian_blocked.hpp"

#endif
#endif

namespace QComputations {

using COMPLEX = std::complex<double>;
using Probs = Matrix<double>;
using Rho = Matrix<COMPLEX>;

// Create rho of pure state (rho = |ksi><ksi|)
Rho create_init_rho(const std::vector<COMPLEX>& init_state);

// Solve Schrodinger equation
// Return Matrix<double> where row(i) - state(basis[i]), cols(j) - probability
// in time[j]
Probs schrodinger(const State<Basis_State>& init_state, Hamiltonian& H, const std::vector<double>& time_vec);

// Solve Schrodinger equation with rho. USE ONLY if you have leaks or gains of
// photons in cavities because it's too slow Return Matrix<double> where row(i)
// - state(basis[i]), cols(j) - probability in time[j]
// !!!!!! If you get incorrect probs - decrease step in time_vec !!!!!!!!
// Probs quantum_master_equation(const std::vector<COMPLEX>& init_state,
//                              Hamiltonian& H,
//                              const std::vector<double>& time_vec,
//                              bool is_full_rho = false);

Probs quantum_master_equation(const State<Basis_State>& init_state,
                              Hamiltonian& H,
                              const std::vector<double>& time_vec,
                              bool is_full_rho = false);

// Solve quantum master equation with different leaks of photons from cavity and
// return vector of time, when probability zero state equal target
std::vector<double> scan_gamma(const std::vector<COMPLEX>& init_state,
                               Hamiltonian& H,
                               size_t cavity_id,
                               const std::vector<double>& time_vec,
                               const std::vector<double>& gamma_vec,
                               double target);

std::pair<Probs, BasisType<Basis_State>> probs_to_cavity_probs(const Probs& probs,
                                                               const BasisType<Basis_State>& basis,
                                                               size_t cavity_id);

Probs exp_evolution(const State<Basis_State>& init_state, Hamiltonian& H, const std::vector<double>& time_vec);

#ifdef ENABLE_MPI
#ifdef ENABLE_CLUSTER
using BLOCKED_Probs = BLOCKED_Matrix<double>;
using BLOCKED_Rho = BLOCKED_Matrix<COMPLEX>;

std::pair<BLOCKED_Probs, BasisType<Basis_State>> probs_to_group_probs(const BLOCKED_Probs& probs,
                                                                      const BasisType<Basis_State>& basis,
                                                                      size_t group_id);

std::pair<BLOCKED_Probs, BasisType<Basis_State>> probs_to_cavity_probs(const BLOCKED_Probs& probs,
                                                                       const BasisType<Basis_State>& basis,
                                                                       size_t cavity_id);

BLOCKED_Rho create_BLOCKED_init_rho(ILP_TYPE ctxt, const std::vector<COMPLEX>& init_state);

BLOCKED_Probs schrodinger(const State<Basis_State>& init_state,
                          BLOCKED_Hamiltonian& H,
                          const std::vector<double>& time_vec);

BLOCKED_Probs quantum_master_equation(const std::vector<COMPLEX>& init_state,
                                      BLOCKED_Hamiltonian& H,
                                      const std::vector<double>& time_vec,
                                      bool is_full_rho = false);

BLOCKED_Probs quantum_master_equation(const State<Basis_State>& init_state,
                                      BLOCKED_Hamiltonian& H,
                                      const std::vector<double>& time_vec,
                                      bool is_full_rho = false);

std::vector<double> scan_gamma(const std::vector<COMPLEX>& init_state,
                               BLOCKED_Hamiltonian& H,
                               size_t cavity_id,
                               const std::vector<double>& time_vec,
                               const std::vector<double>& gamma_vec,
                               double target);

#endif
#endif

template <typename StateType>
State<StateType> schrodinger_step(const State<StateType>& init_state,
                                  Hamiltonian& H,
                                  double t,
                                  const BasisType<StateType>& basis) {
    std::vector<double> eigen_values;
    Matrix<COMPLEX> eigen_vectors;
    eigen_values = H.eigenvalues();
    eigen_vectors = H.eigenvectors();

    auto init_state_vec = init_state.fit_to_basis(basis);
    std::vector<COMPLEX> lambda;
    for (size_t i = 0; i < eigen_values.size(); i++) {
        lambda.emplace_back(eigen_vectors.col(i) | init_state_vec.get_vector());  // <PHI_i|KSI(0)>
    }

    std::vector<COMPLEX> psi_t(eigen_values.size(), 0);
    auto h = QConfig::instance().h();

    for (size_t i = 0; i < eigen_values.size(); i++) {
        for (size_t j = 0; j < psi_t.size(); j++) {
            psi_t[j] += lambda[i] * std::exp(COMPLEX(0, -1 / h * eigen_values[i] * t)) * eigen_vectors[j][i];
        }
    }

    // std::cout << norm(psi_t) << std::endl;

    for (size_t i = 0; i < eigen_values.size(); i++) {
        init_state_vec[i] = psi_t[i];
    }

    return init_state_vec;
}

template <typename StateType>
State<StateType> exp_evolution_step(const State<StateType>& init_state,
                                    Hamiltonian& H,
                                    double dt,
                                    const BasisType<StateType>& basis) {
    H.find_exp(dt);

    auto res = init_state.fit_to_basis(basis);
    auto res_v = H.run_exp(res.get_vector());
    res.set_vector(res_v);
    return res;
}

}  // namespace QComputations

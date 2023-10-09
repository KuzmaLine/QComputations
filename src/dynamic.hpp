#pragma once
#include <vector>
#include <memory>
#include "state.hpp"
#include "hamiltonian.hpp"

#ifdef ENABLE_MPI
#ifdef ENABLE_CLUSTER

#include "hamiltonian_blocked.hpp"

#endif
#endif

namespace QComputations {

namespace Evolution {
    using COMPLEX = std::complex<double>;
    using Probs = Matrix<double>;
    using Rho = Matrix<COMPLEX>;

    // (!!!) REPLACE TO QUANTUM OPERATORS
    Matrix<COMPLEX> create_A_destroy(const std::set<State>& basis, size_t cavity_id);
    Matrix<COMPLEX> create_A_create(const std::set<State>& basis, size_t cavity_id);

    // Create rho of pure state (rho = |ksi><ksi|)
    Rho create_init_rho(const std::vector<COMPLEX>& init_state);

    // Solve Schrodinger equation (dont't work with leaks or gains of photons in cavities)
    // Return Matrix<double> where row(i) - state(basis[i]), cols(j) - probability in time[j]
    Probs schrodinger(const std::vector<COMPLEX>& init_state, Hamiltonian& H, const std::vector<double>& time_vec);

    // Solve Schrodinger equation with rho. USE ONLY if you have leaks or gains of photons in cavities
    // because it's too slow
    // Return Matrix<double> where row(i) - state(basis[i]), cols(j) - probability in time[j]
    // !!!!!! If you get incorrect probs - decrease step in time_vec !!!!!!!!
    Probs quantum_master_equation(const std::vector<COMPLEX>& init_state,
                                  Hamiltonian& H,
                                  const std::vector<double>& time_vec,
                                  bool is_full_rho = false);

    // Solve quantum master equation with different leaks of photons from cavity and return vector of time, when probability
    // zero state equal target
    std::vector<double> scan_gamma(const std::vector<COMPLEX>& init_state,
                                   Hamiltonian& H,
                                   size_t cavity_id,
                                   const std::vector<double>& time_vec,
                                   const std::vector<double>& gamma_vec,
                                   double target);

#ifdef ENABLE_MPI
#ifdef ENABLE_CLUSTER
    using BLOCKED_Probs = BLOCKED_Matrix<double>;
    using BLOCKED_Rho = BLOCKED_Matrix<COMPLEX>;

    BLOCKED_Rho create_BLOCKED_init_rho(const std::vector<COMPLEX>& init_state);

    BLOCKED_Probs schrodinger(const std::vector<COMPLEX>& init_state, BLOCKED_Hamiltonian& H, const std::vector<double>& time_vec);

    BLOCKED_Probs quantum_master_equation(const std::vector<COMPLEX>& init_state,
                                  BLOCKED_Hamiltonian& H,
                                  const std::vector<double>& time_vec,
                                  bool is_full_rho = false);

    Probs Parallel_QME(const std::vector<COMPLEX>& init_state,
                       Hamiltonian& H,
                       const std::vector<double>& time_vec,
                       bool is_full_rho = false);
#endif
#endif
    // TO BE CONTINUED...
}

} // namespace QComputations
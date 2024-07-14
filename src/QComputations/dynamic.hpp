#pragma once
#include <vector>
#include <memory>
#include "state.hpp"
#include "hamiltonian.hpp"

#ifdef ENABLE_MPI
#ifdef ENABLE_CLUSTER

#include "hamiltonian_blocked.hpp"
#include "blocked_vector.hpp"

#endif
#endif

namespace QComputations {

    using COMPLEX = std::complex<double>;
    using Probs = Matrix<double>;
    using Rho = Matrix<COMPLEX>;

    // Create rho of pure state (rho = |ksi><ksi|)
    Rho create_init_rho(const std::vector<COMPLEX>& init_state);

    // Solve Schrodinger equation
    // Return Matrix<double> where row(i) - state(basis[i]), cols(j) - probability in time[j]
    Probs schrodinger(const State<Basis_State>& init_state, Hamiltonian& H, const std::vector<double>& time_vec);

    // Solve Schrodinger equation with rho. USE ONLY if you have leaks or gains of photons in cavities
    // because it's too slow
    // Return Matrix<double> where row(i) - state(basis[i]), cols(j) - probability in time[j]
    // !!!!!! If you get incorrect probs - decrease step in time_vec !!!!!!!!
    //Probs quantum_master_equation(const std::vector<COMPLEX>& init_state,
    //                              Hamiltonian& H,
    //                              const std::vector<double>& time_vec,
    //                              bool is_full_rho = false);

    Probs quantum_master_equation(const State<Basis_State>& init_state,
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
    
    std::pair<Probs, BasisType<Basis_State>> probs_to_cavity_probs(const Probs& probs,
                                            const BasisType<Basis_State>& basis, size_t cavity_id);

#ifdef ENABLE_MPI
#ifdef ENABLE_CLUSTER
    using BLOCKED_Probs = BLOCKED_Matrix<double>;
    using BLOCKED_Rho = BLOCKED_Matrix<COMPLEX>;

    std::pair<BLOCKED_Probs, BasisType<Basis_State>> probs_to_group_probs(const BLOCKED_Probs& probs,
                                                const BasisType<Basis_State>& basis, size_t group_id);

    std::pair<BLOCKED_Probs, BasisType<Basis_State>> probs_to_cavity_probs(const BLOCKED_Probs& probs,
                                                const BasisType<Basis_State>& basis, size_t cavity_id);

    BLOCKED_Rho create_BLOCKED_init_rho(ILP_TYPE ctxt, const std::vector<COMPLEX>& init_state);

    BLOCKED_Probs schrodinger(const State<Basis_State>& init_state, BLOCKED_Hamiltonian& H, const std::vector<double>& time_vec);

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

} // namespace QComputations
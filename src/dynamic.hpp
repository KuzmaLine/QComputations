#pragma once
#include <vector>
#include <memory>
#include "state.hpp"
#include "hamiltonian.hpp"

namespace Evolution {
    using COMPLEX = std::complex<double>;
    using Probs = Matrix<double>;
    using Rho = Matrix<COMPLEX>;

    Matrix<COMPLEX> create_A_destroy(const std::set<State>& basis, size_t cavity_id);
    Matrix<COMPLEX> create_A_create(const std::set<State>& basis, size_t cavity_id);
    Rho create_init_rho(const std::vector<COMPLEX>& init_state);
    Probs schrodinger(const std::vector<COMPLEX>& init_state, Hamiltonian& H, const std::vector<double>& time_vec);
    Probs quantum_master_equation(const std::vector<COMPLEX>& init_state,
                                  Hamiltonian& H,
                                const std::vector<double>& time_vec,
                                const double gamma,
                                bool is_full_rho = false);
    std::vector<double> scan_gamma(const std::vector<COMPLEX>& init_state,
                                   Hamiltonian& H,
                                   size_t cavity_id,
                                   const std::vector<double>& time_vec,
                                   const std::vector<double>& gamma_vec,
                                   double target);
}
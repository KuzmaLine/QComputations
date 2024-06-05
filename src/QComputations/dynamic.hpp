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

namespace Evolution {
    using COMPLEX = std::complex<double>;
    using Probs = Matrix<double>;
    using Rho = Matrix<COMPLEX>;

    // (!!!) REPLACE TO QUANTUM OPERATORS
    Matrix<COMPLEX> create_A_destroy(const std::set<Basis_State>& basis, size_t cavity_id);
    Matrix<COMPLEX> create_A_create(const std::set<Basis_State>& basis, size_t cavity_id);

    // Создать матрицу плотности чистого состояния
    Rho create_init_rho(const std::vector<COMPLEX>& init_state);

    // Уравнение Шредингера
    // Return Matrix<double> где row(i) - state(basis[i]), cols(j) - вероятность в time[j]
    Probs schrodinger(const State<Basis_State>& init_state, Hamiltonian& H, const std::vector<double>& time_vec);

    // Решает основное квантовое уравнение, использовать только при наличии фактора декогерентности из-за плохого ускорения
    // Return Matrix<double> where row(i) - state(basis[i]), cols(j) - probability in time[j]

    // !!!!!! Если получаете некорректные вероятности (улетающие в бесконечность) - уменьшите шаг во времени !!!!!!!!
    Probs quantum_master_equation(const State<Basis_State>& init_state,
                            Hamiltonian& H,
                            const std::vector<double>& time_vec,
                            bool is_full_rho = false);

    // Решает основное квантовое уравнение с разными интенсивности улёта фотона, и строит по ним вектор времени, 
    // когда вероятность состояния всех нулей становится равно target
    std::vector<double> scan_gamma(const std::vector<COMPLEX>& init_state,
                                   Hamiltonian& H,
                                   size_t cavity_id,
                                   const std::vector<double>& time_vec,
                                   const std::vector<double>& gamma_vec,
                                   double target);
    
    std::pair<Probs, std::set<Basis_State>> probs_to_cavity_probs(const Probs& probs,
                                            const std::set<Basis_State>& basis, size_t cavity_id);

#ifdef ENABLE_MPI
#ifdef ENABLE_CLUSTER
    using BLOCKED_Probs = BLOCKED_Matrix<double>;
    using BLOCKED_Rho = BLOCKED_Matrix<COMPLEX>;

    // Объединяет все вероятности в конкретную группу, то есть, если мы рассматриваем состояния только конкретной группы.
    std::pair<BLOCKED_Probs, std::set<Basis_State>> probs_to_group_probs(const BLOCKED_Probs& probs,
                                                const std::set<Basis_State>& basis, size_t group_id);

    // Другое название probs_to_group_probs
    std::pair<BLOCKED_Probs, std::set<Basis_State>> probs_to_cavity_probs(const BLOCKED_Probs& probs,
                                                const std::set<Basis_State>& basis, size_t cavity_id);

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
    // TO BE CONTINUED...
}

} // namespace QComputations
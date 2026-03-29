#pragma once
#include <vector>
#include <memory>
#include "state.hpp"
#include "hamiltonian.hpp"

/* modified: 
    schrodinger_step */

#ifdef ENABLE_MPI
#ifdef ENABLE_CLUSTER

#include "hamiltonian_blocked.hpp"
#include "blocked_vector.hpp"

#endif
#endif

#ifdef __CUDACC__
#include "cuda_hamiltonian.hpp"
#endif

namespace QComputations {
    using COMPLEX = std::complex<double>;
    using Probs = Matrix<double>;
    using Rho = Matrix<COMPLEX>;

    // (!!!) REPLACE TO QUANTUM OPERATORS
    Matrix<COMPLEX> create_A_destroy(const BasisType<Basis_State>& basis, size_t cavity_id);
    Matrix<COMPLEX> create_A_create(const BasisType<Basis_State>& basis, size_t cavity_id);

    // Create rho of pure state (rho = |ksi><ksi|)
    Rho create_init_rho(const std::vector<COMPLEX>& init_state);

#ifdef ENABLE_ONEAPI

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

    // Solve Schrodinger equation
    // Return Matrix<double> where row(i) - state(basis[i]), cols(j) - probability in time[j]
    Probs schrodinger(const State<Basis_State>& init_state, Hamiltonian& H, const std::vector<double>& time_vec);

    // Solve quantum master equation with different leaks of photons from cavity and return vector of time, when probability
    // zero state equal target
    std::vector<double> scan_gamma(const TCH_State& init_state,
                                   size_t cavity_id,
                                   const std::vector<double>& time_vec,
                                   const std::vector<double>& gamma_vec,
                                   double target);

    //Probs exp_evolution(const State<Basis_State>& init_state, Hamiltonian& H,
    //                    double dt, size_t STEPS_COUNT, 
    //                    const std::function<void(std::vector<COMPLEX>&)>& func = nothing_function);
#endif

    std::pair<Probs, std::vector<std::shared_ptr<Basis_State>>> probs_to_group_probs(const Probs& probs,
                                            const BasisType<Basis_State>& basis, size_t cavity_id);

    std::pair<Probs, std::vector<std::shared_ptr<Basis_State>>> probs_to_qudits(const Probs& probs, const BasisType<Basis_State>& basis, const std::vector<size_t>& qudits);


#ifdef __CUDACC__
    using CUDA_Rho = CUDA_Matrix<COMPLEX>;
    using CUDA_Probs = CUDA_Matrix<double>;

    __global__ void rho_to_probs(cuDoubleComplex** rho_vec, double* probs, size_t time_length, size_t basis_size);

    /*
    Probs quantum_master_equation(const State<Basis_State>& init_state,
                            CUDA_Hamiltonian& H,
                            const std::vector<double>& time_vec,
                            bool is_full_rho = false);

    // Solve Schrodinger equation
    // Return Matrix<double> where row(i) - state(basis[i]), cols(j) - probability in time[j]
    Probs schrodinger(const State<Basis_State>& init_state, Hamiltonian& H, const std::vector<double>& time_vec);

    // Solve quantum master equation with different leaks of photons from cavity and return vector of time, when probability
    // zero state equal target
    std::vector<double> scan_gamma(const TCH_State& init_state,
                                   size_t cavity_id,
                                   const std::vector<double>& time_vec,
                                   const std::vector<double>& gamma_vec,
                                   double target);

    Probs exp_evolution(const State<Basis_State>& init_state, Hamiltonian& H,
                        double dt, size_t STEPS_COUNT, 
                        const std::function<void(std::vector<COMPLEX>&)>& func = nothing_function);
    */

    Probs quantum_master_equation(const State<Basis_State>& init_state,
                        CUDA_Hamiltonian& H,
                        const std::vector<double>& time_vec,
                        bool is_full_rho = false);
    
    Probs quantum_master_equation(const std::vector<COMPLEX>& init_state,
        CUDA_Hamiltonian& H,
        const std::vector<double>& time_vec,
        bool is_full_rho = false);

    Probs schrodinger(const State<Basis_State>& init_state, CUDA_Hamiltonian& H, const std::vector<double>& time_vec);
#endif

#ifdef ENABLE_MPI
#ifdef ENABLE_CLUSTER
    using BLOCKED_Probs = BLOCKED_Matrix<double>;
    using BLOCKED_Rho = BLOCKED_Matrix<COMPLEX>;

    //std::pair<BLOCKED_Probs, BasisType<Basis_State>> probs_to_group_probs(const BLOCKED_Probs& probs,
    //                                            const BasisType<Basis_State>& basis, size_t group_id);
    //std::pair<BLOCKED_Probs, BasisType<Basis_State>> probs_to_qudits(const Probs& probs, const BasisType<Basis_State>& basis, const std::vector<size_t>& qudits);

    std::pair<BLOCKED_Probs, BasisType<Basis_State>> probs_to_cavity_probs(const BLOCKED_Probs& probs,
                                                const BasisType<Basis_State>& basis, size_t cavity_id);

    BLOCKED_Rho create_BLOCKED_init_rho(ILP_TYPE ctxt, const std::vector<COMPLEX>& init_state);

    //BLOCKED_Probs schrodinger(const std::vector<COMPLEX>& init_state, BLOCKED_Hamiltonian& H, const std::vector<double>& time_vec);
    BLOCKED_Probs schrodinger(const State<Basis_State>& init_state, BLOCKED_Hamiltonian& H, const std::vector<double>& time_vec);

    BLOCKED_Probs quantum_master_equation(const std::vector<COMPLEX>& init_state,
                                  BLOCKED_Hamiltonian& H,
                                  const std::vector<double>& time_vec,
                                  bool is_full_rho = false);

    BLOCKED_Probs quantum_master_equation(const State<Basis_State>& init_state,
                                BLOCKED_Hamiltonian& H,
                                const std::vector<double>& time_vec,
                                bool is_full_rho = false);

    // РУДИМЕНТ
    /*
    Probs Parallel_QME(const std::vector<COMPLEX>& init_state,
                       Hamiltonian& H,
                       const std::vector<double>& time_vec,
                       bool is_full_rho = false);
    */

    std::vector<double> blocked_scan_gamma(const TCH_State& init_state,
                                    size_t cavity_id,
                                    const std::vector<double>& time_vec,
                                    const std::vector<double>& gamma_vec,
                                    double target, const BasisType<TCH_State>& basis);

#endif
#endif
    // TO BE CONTINUED...

    /* ----------------- TEMPLATE FUNCTIONS --------------------- */

#ifdef ENABLE_ONEAPI
    template<typename StateType>
    State<StateType> schrodinger_step(const State<StateType>& init_state, Hamiltonian& H, double t, const BasisType<StateType>& basis) {
        std::vector<double> eigen_values;
        Matrix<COMPLEX> eigen_vectors;
        eigen_values = H.eigenvalues();
        eigen_vectors = H.eigenvectors();

        auto init_state_vec = init_state.fit_to_basis(basis);
        std::vector<COMPLEX> lambda;
        for (size_t i = 0; i < eigen_values.size(); i++) {
            lambda.emplace_back(eigen_vectors.col(i) | init_state_vec.get_vector()); // <PHI_i|KSI(0)> 
        }

        std::vector<COMPLEX> psi_t(eigen_values.size(), 0);
        auto h = QConfig::instance().h();

        for (size_t i = 0; i < eigen_values.size(); i++) {

            #pragma omp parallel for
            for (size_t j = 0; j < psi_t.size(); j++) {
                psi_t[j] += lambda[i] * std::exp(COMPLEX(0, -1 / h * eigen_values[i] * t)) * eigen_vectors[j][i];
            }
        }

        //std::cout << norm(psi_t) << std::endl;

        for (size_t i = 0; i < eigen_values.size(); i++) {
            init_state_vec[i] = psi_t[i];
        }

        return init_state_vec;
    }

    template<typename StateType>
    State<StateType> exp_evolution_step(const State<StateType>& init_state, Hamiltonian& H, double dt, const BasisType<StateType>& basis) {
        H.find_exp(dt);

        auto res = init_state.fit_to_basis(basis);
        auto res_v = H.run_exp(res.get_vector());
        res.set_vector(res_v);
        return res;
    }

#endif

} // namespace QComputations
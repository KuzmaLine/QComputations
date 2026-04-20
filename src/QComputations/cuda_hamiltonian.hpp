#ifdef __CUDACC__

#pragma once
#include <vector>
#include <functional>
#include <complex>
#include <iostream>
#include "state.hpp"
#include "config.hpp"
#include "matrix.hpp"
#include "graph.hpp"
#include "functions.hpp"
#include "cuda_matrix.hpp"
#include "quantum_operators.hpp"

namespace QComputations {

namespace {
    typedef std::complex<double> COMPLEX;
}

class CUDA_Hamiltonian {
    public:
        explicit CUDA_Hamiltonian() = default;
        size_t n() const { return H_.n(); }
        size_t size() const { return H_.n(); }
        std::vector<std::shared_ptr<Basis_State>> get_basis() const { return basis_; }
        std::vector<std::pair<double, CUDA_Matrix<COMPLEX>>> get_decoherence() const { return decoherence_;}
        cublasHandle_t handle() const { return H_.handle(); }
        //void write_to_csv_file(const std::string& filename) const { H_.write_to_csv_file(filename); }
        Matrix<COMPLEX> to_cpu() const { return H_.to_cpu(); }

        virtual ~CUDA_Hamiltonian() {
            if (is_calculated_eigen_) {
                cudaFree(eigenvalues_);
                eigenvalues_ = nullptr;
            }
        }
        
        void virtual eigen() {
            if (!is_calculated_eigen_) {
                auto p = HermitEigen(H_);
                eigenvalues_ = p.first;
                eigenvectors_ = p.second;
                is_calculated_eigen_ = true;
            }
        }

        virtual double* eigenvalues() {
            this->eigen();
            return eigenvalues_;
        }

        CUDA_Matrix<COMPLEX> virtual eigenvectors() {
            this->eigen();
            return eigenvectors_;
        }

        /*
        void find_exp(double dt) {
            if (std::abs(dt - dt_exp_) >= ZERO_EPS) {
                H_EXP_ = exp(H_, dt, COMPLEX(0, -1/QConfig::instance().h()));
            }
            //H_EXP_.show();
        }

        State<Basis_State> run_exp(const State<Basis_State>& state) {
            auto res = state;
            res.set_vector(H_EXP_ * state.get_vector());
            return res;
        }

        std::vector<COMPLEX> run_exp(const std::vector<COMPLEX>& state) {
            return H_EXP_ * state;
        }
        */

        void show(size_t width = QConfig::instance().width()) const { H_.show(width); }
        CUDA_Matrix<COMPLEX> get_matrix() const { return H_; }

        void write_to_csv_file(const std::string& filename) const { H_.write_to_csv_file(filename); }
    protected:
        bool is_calculated_eigen_ = false;
        std::vector<std::shared_ptr<Basis_State>> basis_;
        CUDA_Matrix<COMPLEX> H_;
        CUDA_Matrix<COMPLEX> eigenvectors_;
        double* eigenvalues_;
        std::vector<std::pair<double, CUDA_Matrix<COMPLEX>>> decoherence_;
        //CUDA_Matrix<COMPLEX> H_EXP_;
        //double dt_exp_ = 0;
};

template<typename StateType>
class CUDA_H_by_Operator: public CUDA_Hamiltonian {
    public:
        explicit CUDA_H_by_Operator(cublasHandle_t handle, const State<StateType>& init_state, const Operator<StateType>& H_op,
                                     const std::vector<std::pair<double, Operator<StateType>>>& decoherence = {});
};

template<typename StateType>
CUDA_H_by_Operator<StateType>::CUDA_H_by_Operator(cublasHandle_t handle, const State<StateType>& init_state, const Operator<StateType>& H_op,
                                     const std::vector<std::pair<double, Operator<StateType>>>& decoherence) {
    std::vector<Operator<StateType>> dec_tmp;
    for (const auto& p: decoherence) {
        dec_tmp.push_back(p.second);
    }

    auto basis_original = sort_basis(State_Graph<StateType>(init_state, H_op, dec_tmp).get_basis());
    State<StateType> basis_map(basis_original);
    basis_ = convert_to<StateType>(basis_original);

    size_t dim = basis_original.size();
    Matrix<COMPLEX> H_tmp(FORTRAN_STYLE, dim, dim, COMPLEX(0, 0));

    size_t col_state = 0;
    for (auto state: basis_original) {
        //std::cout << state->to_string() << std::endl;
        State<StateType> res_state(H_op.run(State<StateType>(*state)));

        res_state.set_sorted(true);
        for (auto p: res_state.state_map()) {
            auto idx = basis_map.get_index(p.first);
            H_tmp(idx, col_state) = res_state[p.second];
        }

        col_state++;
    }

    H_ = CUDA_Matrix<COMPLEX>(handle, H_tmp);
    //H_ = Matrix<COMPLEX>(C_STYLE, size, size, func);
    for (const auto& p: decoherence) {
        auto A = CUDA_Matrix<COMPLEX>(handle, operator_to_matrix<StateType>(p.second, basis_original, FORTRAN_STYLE));
        decoherence_.push_back(std::make_pair(p.first, A));
    }
}

/*
template<typename StateType>
class CUDA_H_by_Scalar_Product: public CUDA_Hamiltonian {
    public:
        explicit CUDA_H_by_Scalar_Product(const State<StateType>& init_state, 
                                     const std::function<COMPLEX(const StateType& i, const StateType& j)>& func,
                                     BasisType<StateType> basis = {});
};

template<typename StateType>
CUDA_H_by_Scalar_Product<StateType>::H_by_Scalar_Product(const State<StateType>& init_state,
                                                    const std::function<COMPLEX(const StateType& i, const StateType& j)>& func,
                                                    BasisType<StateType> basis) {
    if (basis.empty()) {
        auto zero_state = init_state(0);
        zero_state->set_zero();
        basis.insert(std::shared_ptr<StateType>(new StateType(*zero_state)));

        StateType cur_state = *zero_state;

        for (size_t i = 0; i < cur_state.qudits_count(); i++) {
            if (cur_state.get_max_val(i) > cur_state.get_qudit(i)) {
                cur_state.set_qudit(cur_state.get_qudit(i) + 1, i);
                basis.insert(std::shared_ptr<StateType>(new StateType(cur_state)));

                for (size_t j = i - 1; j >= 0; j--) {
                    cur_state.set_qudit(0, j);
                }

                i = 0;
            }
        }
    }

    basis_ = convert_to<StateType>(basis);

    auto matrix_func = std::function<COMPLEX(size_t i, size_t j)>{
        [&func, &basis](size_t i, size_t j) {
            return func(*get_state_from_basis(basis, i), *get_state_from_basis(basis, j));
        }
    };

    H_ = Matrix<COMPLEX>(C_STYLE, basis.size(), basis.size(), matrix_func);
}

*/

class CUDA_H_TCH : public CUDA_H_by_Operator<TCH_State> {
    public:
        explicit CUDA_H_TCH(cublasHandle_t handle, const State<TCH_State>& state);
};


} // namespace QComputations

#endif
#pragma once
#include <vector>
#include <functional>
#include <complex>
#include <iostream>
#include "state.hpp"
#include "config.hpp"
#include "matrix.hpp"
#include "graph.hpp"
#include "quantum_operators.hpp"

namespace QComputations {

namespace {
    typedef std::complex<double> COMPLEX;
}

class Hamiltonian {
    public:
        explicit Hamiltonian() = default;
        size_t n() const { return H_.n(); }
        size_t size() const { return H_.n(); }
        TCH_State grid() const { return grid_; }
        TCH_State get_grid() const { return grid_; }
        void set_grid(const TCH_State& grid) { grid_ = grid; }
        std::set<Basis_State> get_basis() const { return basis_; }
        std::vector<std::pair<double, Matrix<COMPLEX>>> get_decoherence() const { return decoherence_;}

        void virtual eigen() {
            if (!is_calculated_eigen_) {
                auto p = Hermit_Lanczos(H_);
                eigenvalues_ = p.first;
                eigenvectors_ = p.second;
                is_calculated_eigen_ = true;
            }
        }

        std::vector<double> virtual eigenvalues() {
            this->eigen();
            return eigenvalues_;
        }

        Matrix<COMPLEX> virtual eigenvectors() {
            this->eigen();
            return eigenvectors_;
        }

        void show(size_t width = QConfig::instance().width()) const { H_.show(width); }
        Matrix<COMPLEX> get_matrix() const { return H_; }

        void write_to_csv_file(const std::string& filename) const { H_.write_to_csv_file(filename); }
    protected:
        bool is_calculated_eigen_ = false;
        std::set<Basis_State> basis_;
        Matrix<COMPLEX> H_;
        Matrix<COMPLEX> eigenvectors_;
        std::vector<double> eigenvalues_;
        std::vector<std::pair<double, Matrix<COMPLEX>>> decoherence_;
        TCH_State grid_;
};

template<typename StateType>
class H_by_Operator: public Hamiltonian {
    public:
        explicit H_by_Operator(const State<StateType>& init_state, const Operator<StateType>& H_op,
                                     const std::vector<std::pair<double, Operator<StateType>>>& decoherence = {});
};

template<typename StateType>
H_by_Operator<StateType>::H_by_Operator(const State<StateType>& init_state, const Operator<StateType>& H_op,
                                     const std::vector<std::pair<double, Operator<StateType>>>& decoherence) {
    std::vector<Operator<StateType>> dec_tmp;
    for (const auto& p: decoherence) {
        dec_tmp.push_back(p.second);
    }

    auto basis = State_Graph<StateType>(init_state, H_op, dec_tmp).get_basis();
    basis_ = convert_to<StateType>(basis);

    size_t size = basis_.size();

    std::function<COMPLEX(size_t i, size_t j)> func = {
        [&basis, &H_op](size_t i, size_t j) {
            auto state_from = get_elem_from_set<StateType>(basis, j);
            auto state_to = get_elem_from_set<StateType>(basis, i);
            auto res_state = H_op.run(State<StateType>(state_from));
            
            COMPLEX res = COMPLEX(0, 0);

            size_t index = 0;
            for (const auto& state: res_state.get_state_components()) {
                if (state == state_to) {
                    res = res_state[index];
                    break;
                }

                index++;
            }

            return res;
        }
    };

    H_ = Matrix<COMPLEX>(C_STYLE, size, size, func);
    for (const auto& p: decoherence) {
        auto A = Matrix<COMPLEX>(operator_to_matrix<StateType>(p.second, basis));
        decoherence_.push_back(std::make_pair(p.first, A));
    }
}

class H_TCH : public H_by_Operator<TCH_State> {
    public:
        explicit H_TCH(const State<TCH_State>& state);
};

} // namespace QComputations
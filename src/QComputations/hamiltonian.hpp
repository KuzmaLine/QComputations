#pragma once
#include <complex>
#include <functional>
#include <iostream>
#include <vector>

#include "config.hpp"
#include "functions.hpp"
#include "graph.hpp"
#include "matrix.hpp"
#include "quantum_operators.hpp"
#include "state.hpp"

namespace QComputations {

    namespace {
        typedef std::complex<double> COMPLEX;
        constexpr double ZERO_EPS = 1e-32;
    }  // namespace

    std::set<TCH_State> define_basis_of_hamiltonian(const TCH_State& grid);

    class Hamiltonian {
       public:
        explicit Hamiltonian() = default;
        size_t n() const { return H_.n(); }
        size_t size() const { return H_.n(); }
        TCH_State grid() const { return grid_; }
        TCH_State get_grid() const { return grid_; }
        void set_grid(const TCH_State& grid) { grid_ = grid; }
        BasisType<Basis_State> get_basis() const { return basis_; }
        std::vector<std::pair<double, Matrix<COMPLEX>>> get_decoherence() const { return decoherence_; }

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

        void find_exp(double dt) {
            if (std::abs(dt - dt_exp_) >= ZERO_EPS) {
                H_EXP_ = exp(H_, dt, COMPLEX(0, -1 / QConfig::instance().h()));
            }
        }

        State<Basis_State> run_exp(const State<Basis_State>& state) {
            auto res = state;
            res.set_vector(H_EXP_ * state.get_vector());
            return res;
        }

        std::vector<COMPLEX> run_exp(const std::vector<COMPLEX>& state) { return H_EXP_ * state; }

        void show(size_t width = QConfig::instance().width()) const { H_.show(width); }
        Matrix<COMPLEX> get_matrix() const { return H_; }

        void write_to_csv_file(const std::string& filename) const { H_.write_to_csv_file(filename); }

       protected:
        bool is_calculated_eigen_ = false;
        BasisType<Basis_State> basis_;
        Matrix<COMPLEX> H_;
        Matrix<COMPLEX> eigenvectors_;
        std::vector<double> eigenvalues_;
        std::vector<std::pair<double, Matrix<COMPLEX>>> decoherence_;
        TCH_State grid_;
        Matrix<COMPLEX> H_EXP_;
        double dt_exp_ = 0;
    };

    template <typename StateType>
    class H_by_Operator : public Hamiltonian {
       public:
        explicit H_by_Operator(const State<StateType>& init_state,
                               const Operator<StateType>& H_op,
                               const std::vector<std::pair<double, Operator<StateType>>>& decoherence = {});
    };

    template <typename StateType>
    H_by_Operator<StateType>::H_by_Operator(const State<StateType>& init_state,
                                            const Operator<StateType>& H_op,
                                            const std::vector<std::pair<double, Operator<StateType>>>& decoherence) {
        std::vector<Operator<StateType>> dec_tmp;
        for (const auto& p : decoherence) {
            dec_tmp.push_back(p.second);
        }

        auto basis = State_Graph<StateType>(init_state, H_op, dec_tmp).get_basis();
        basis_ = convert_to<StateType>(basis);

        size_t dim = basis.size();
        H_ = Matrix<COMPLEX>(C_STYLE, dim, dim, COMPLEX(0, 0));

        size_t col_state = 0;
        for (auto state : basis) {
            auto res_state = H_op.run(State<StateType>(*state));

            size_t index = 0;
            for (auto state_res : res_state.state_components()) {
                H_[get_index_state_in_basis(*state_res, basis)][col_state] = res_state[index++];
            }

            col_state++;
        }

        for (const auto& p : decoherence) {
            auto A = Matrix<COMPLEX>(operator_to_matrix<StateType>(p.second, basis));
            decoherence_.push_back(std::make_pair(p.first, A));
        }
    }

    template <typename StateType>
    class H_by_Scalar_Product : public Hamiltonian {
       public:
        explicit H_by_Scalar_Product(const State<StateType>& init_state,
                                     const std::function<COMPLEX(const StateType& i, const StateType& j)>& func,
                                     BasisType<StateType> basis = {});
    };

    template <typename StateType>
    H_by_Scalar_Product<StateType>::H_by_Scalar_Product(
        const State<StateType>& init_state,
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

        auto matrix_func = std::function<COMPLEX(size_t i, size_t j)>{[&func, &basis](size_t i, size_t j) {
            return func(*get_state_from_basis(basis, i), *get_state_from_basis(basis, j));
        }};

        H_ = Matrix<COMPLEX>(C_STYLE, basis.size(), basis.size(), matrix_func);
    }

    class H_TCH : public H_by_Operator<TCH_State> {
       public:
        explicit H_TCH(const State<TCH_State>& state, const std::vector<std::pair<double, Operator<TCH_State>>>& dec);
    };

}  // namespace QComputations

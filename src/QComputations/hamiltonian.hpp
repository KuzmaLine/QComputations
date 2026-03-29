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
#include "quantum_operators.hpp"

namespace QComputations {

namespace {
    typedef std::complex<double> COMPLEX;
    
    constexpr double ZERO_EPS = 1e-32;
}

std::set<TCH_State> define_basis_of_hamiltonian(const TCH_State& grid);

class Hamiltonian {
    public:
        explicit Hamiltonian() = default;
        Hamiltonian(const Matrix<COMPLEX>& H): H_(H) {}
        size_t n() const { return H_.n(); }
        size_t size() const { return H_.n(); }
        // TCH_State grid() const { return grid_; }
        // TCH_State get_grid() const { return grid_; }
        // void set_grid(const TCH_State& grid) { grid_ = grid; }
        std::vector<std::shared_ptr<Basis_State>> get_basis() const { return basis_; }
        std::vector<std::pair<double, Matrix<COMPLEX>>> get_decoherence() const { return decoherence_;}
        //void write_to_csv_file(const std::string& filename) const { H_.write_to_csv_file(filename); }

#ifdef ENABLE_ONEAPI
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
#endif

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

        void show(size_t width = QConfig::instance().width()) const { H_.show(width); }
        Matrix<COMPLEX> get_matrix() const { return H_; }

        void write_to_csv_file(const std::string& filename) const { H_.write_to_csv_file(filename); }
    protected:
        bool is_calculated_eigen_ = false;
        std::vector<std::shared_ptr<Basis_State>> basis_;
        Matrix<COMPLEX> H_;
        Matrix<COMPLEX> eigenvectors_;
        std::vector<double> eigenvalues_;
        std::vector<std::pair<double, Matrix<COMPLEX>>> decoherence_;
        // TCH_State grid_;
        Matrix<COMPLEX> H_EXP_;
        double dt_exp_ = 0;
};

/*
class H_by_func : public Hamiltonian {
    public:
        H_by_func(size_t n, std::function<COMPLEX(size_t, size_t)> func);
        void set_basis(const std::set<Basis_State>& basis) { basis_ = basis; }
        void set_grid(const TCH_State& grid) { grid_ = grid; }
    private:
        std::function<COMPLEX(size_t, size_t)> func_;
};

class H_by_Matrix : public Hamiltonian {
    public:
        H_by_Matrix(const Matrix<COMPLEX>& H) { H_ = H; }
        void set_basis(const std::set<Basis_State>& basis) { basis_ = basis; }
        void set_grid(const TCH_State& grid) { grid_ = grid; }
};

// NEED UPDATE
class H_JC : public Hamiltonian {
    public:
        explicit H_JC(const TCH_State& state);  // Генерируется по умолчанию в RWA приближении
        void make_exact();                  // Делает гамильтониан точным
};

// NEED UPDATE
class H_TC : public Hamiltonian {
    public:
        explicit H_TC(const TCH_State& state);
    private:
        size_t n_;
        size_t m_;
};

class H_TCH : public Hamiltonian {
    public:
        H_TCH(const TCH_State& init_state);
};

*/

template<typename StateType>
class H_by_Operator: public Hamiltonian {
    public:
        explicit H_by_Operator(const State<StateType>& init_state, const Operator<StateType>& H_op,
                                     const std::vector<std::pair<double, Operator<StateType>>>& decoherence = {});
        //BasisType<StateType> get_original_basis() const {
        //    return basis_original_;
        //}
    private:
            //BasisType<StateType> basis_original_;
};

template<typename StateType>
H_by_Operator<StateType>::H_by_Operator(const State<StateType>& init_state, const Operator<StateType>& H_op,
                                     const std::vector<std::pair<double, Operator<StateType>>>& decoherence) {
    std::vector<Operator<StateType>> dec_tmp;
    for (const auto& p: decoherence) {
        dec_tmp.push_back(p.second);
    }

    //basis_original_ = State_Graph<StateType>(init_state, H_op, dec_tmp).get_basis();
    auto basis_original = sort_basis(State_Graph<StateType>(init_state, H_op, dec_tmp).get_basis());
    State<StateType> basis_map(basis_original);
    basis_ = convert_to<StateType>(basis_original);

    /*
    size_t size = basis_.size();

    std::function<COMPLEX(size_t i, size_t j)> func = {
        [&basis, &H_op](size_t i, size_t j) {
            auto state_from = get_state_from_basis<StateType>(basis, j);
            auto state_to = get_state_from_basis<StateType>(basis, i);
            auto res_state = H_op.run(State<StateType>(*state_from));
            
            if (res_state.is_in_state(*state_to)) {
                return res_state[*state_to];
            } else {
                return COMPLEX(0, 0);
            }
        }
    };
    */

    size_t dim = basis_original.size();
    H_ = Matrix<COMPLEX>(C_STYLE, dim, dim, COMPLEX(0, 0));

    size_t col_state = 0;
    for (auto state: basis_original) {
        //std::cout << state->to_string() << std::endl;
        State<StateType> res_state(H_op.run(State<StateType>(*state)));
        // std::cout << state->to_string() << std::endl;
        // std::cout << res_state.to_string() << std::endl;
        // // res_state.sort();
        // // auto res_state_full = res_state.fit_to_basis(basis_original);
        // res_state_full.sort();

        // size_t index = 0;
        // for (auto state_res: res_state.state_components()) {
        //     H_[get_index_state_in_basis(*state_res, basis_original)][col_state] = res_state[index++];
        // }

        // auto idxs = res_state.fit_indexes_to_basis(basis_original);

        res_state.set_sorted(true);
        for (auto p: res_state.state_map()) {
            auto idx = basis_map.get_index(p.first);
            H_[idx][col_state] = res_state[p.second];
        }

        col_state++;
    }

    //H_ = Matrix<COMPLEX>(C_STYLE, size, size, func);
    for (const auto& p: decoherence) {
        auto A = Matrix<COMPLEX>(operator_to_matrix<StateType>(p.second, basis_original));
        decoherence_.push_back(std::make_pair(p.first, A));
    }
}

template<typename StateType>
class H_by_Scalar_Product: public Hamiltonian {
    public:
        explicit H_by_Scalar_Product(const State<StateType>& init_state, 
                                     const std::function<COMPLEX(const StateType& i, const StateType& j)>& func);
        explicit H_by_Scalar_Product(const State<StateType>& init_state, 
                                     const std::function<COMPLEX(const StateType& i, const StateType& j)>& func,
                                     const BasisType<StateType>& basis) {
            generate_scalar_product_H(init_state, func, basis);
        }
    private:
        void generate_scalar_product_H(const State<StateType>& init_state,
                                              const std::function<COMPLEX(const StateType& i, const StateType& j)>& func,
                                              const BasisType<StateType>& basis) {
            this->basis_ = convert_to<StateType>(basis);
            auto basis_sorted = sort_basis(basis);

            auto matrix_func = std::function<COMPLEX(size_t i, size_t j)>{
                [&func, &basis_sorted](size_t i, size_t j) {
                    return func(*(basis_sorted[i]), *(basis_sorted[j]));
                }
            };

            this->H_ = Matrix<COMPLEX>(C_STYLE, basis.size(), basis.size(), matrix_func);
        }
};

template<typename StateType>
H_by_Scalar_Product<StateType>::H_by_Scalar_Product(const State<StateType>& init_state,
                                                    const std::function<COMPLEX(const StateType& i, const StateType& j)>& func) {
    // auto basis = generate_full_basis(init_state);
    auto zero_state = init_state(0);
    zero_state->set_zero();
    BasisType<StateType> basis;
    basis.insert(std::make_shared(StateType(*zero_state)));

    StateType cur_state(*zero_state);
    bool is_not_max = true;

    while(is_not_max) {
        is_not_max = false;
        for (size_t i = 0; i < cur_state.qudits_count() && !is_not_max; i++) {
            auto cur_qudit = cur_state.get_qudit(i);
            if(cur_qudit != cur_state.get_max_val(i)) {
                cur_state.set_qudit(cur_qudit + 1, i);
                is_not_max = true;

                for (size_t j = i; j != 0; j--) {
                    cur_state.set_qudit(0, j - 1);
                }
            }
        }

        basis.insert(std::make_shared(StateType(cur_state)));
    }

    generate_scalar_product_H(init_state, func, basis);
}

// template<typename StateType>
// H_by_Scalar_Product<StateType>::H_by_Scalar_Product(const State<StateType>& init_state,
//                                                     const std::function<COMPLEX(const StateType& i, const StateType& j)>& func,
//                                                     const BasisType<StateType>& basis) {
//     // if (basis.empty()) {
//     //     // auto basis = generate_full_basis(init_state);
//     //     auto zero_state = init_state(0);
//     //     zero_state->set_zero();
//     //     basis.insert(std::make_shared(StateType(*zero_state))));

//     //     StateType cur_state(*zero_state);
//     //     bool is_not_max = true;

//     //     while(is_not_max) {
//     //         is_not_max = false;
//     //         for (size_t i = 0; i < cur_state.qudits_count() && !is_not_max; i++) {
//     //             auto cur_qudit = cur_state.get_qudit(i);
//     //             if(cur_qudit != cur_state.get_max_val(i)) {
//     //                 cur_state.set_qudit(cur_qudit + 1, i);
//     //                 is_not_max = true;

//     //                 for (size_t j = i; j != 0; j--) {
//     //                     cur_state.set_qudit(0, j - 1);
//     //                 }
//     //             }
//     //         }

//     //         basis.insert(std::make_shared(StateType(cur_state)));
//     //     }
//     // }

//     generate_scalar_product_H(init_state, func, basis);
// }

class H_TCH : public H_by_Operator<TCH_State> {
    public:
        explicit H_TCH(const State<TCH_State>& state);
};

} // namespace QComputations
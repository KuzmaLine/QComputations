#pragma once
#include "hamiltonian.hpp"
#include "csr_matrix.hpp"

namespace QComputations {

class CSR_Hamiltonian {
    public:
        explicit CSR_Hamiltonian() = default;
        CSR_Hamiltonian(const CSR_Matrix<COMPLEX>& H): H_(H) {}
        size_t n() const { return H_.n(); }
        size_t size() const { return H_.n(); }
        std::vector<std::shared_ptr<Basis_State>> get_basis() const { return basis_; }
        std::vector<std::pair<double, CSR_Matrix<COMPLEX>>> get_decoherence() const { return decoherence_;}

// #ifdef ENABLE_ONEAPI
//         void virtual eigen() {
//             if (!is_calculated_eigen_) {
//                 auto p = Hermit_Lanczos(H_);
//                 eigenvalues_ = p.first;
//                 eigenvectors_ = p.second;
//                 is_calculated_eigen_ = true;
//             }
//         }

//         std::vector<double> virtual eigenvalues() {
//             this->eigen();
//             return eigenvalues_;
//         }

//         Matrix<COMPLEX> virtual eigenvectors() {
//             this->eigen();
//             return eigenvectors_;
//         }
// #endif

        // void find_exp(double dt) {
        //     if (std::abs(dt - dt_exp_) >= ZERO_EPS) {
        //         H_EXP_ = exp(H_, dt, COMPLEX(0, -1/QConfig::instance().h()));
        //     }
        //     //H_EXP_.show();
        // }

        // State<Basis_State> run_exp(const State<Basis_State>& state) {
        //     auto res = state;
        //     res.set_vector(H_EXP_ * state.get_vector());
        //     return res;
        // }

        // std::vector<COMPLEX> run_exp(const std::vector<COMPLEX>& state) {
        //     return H_EXP_ * state;
        // }

        void show(size_t width = QConfig::instance().width()) const { H_.show(); }
        CSR_Matrix<COMPLEX> get_matrix() const { return H_; }

        // void write_to_csv_file(const std::string& filename) const { H_.write_to_csv_file(filename); }
    protected:
        // bool is_calculated_eigen_ = false;
        std::vector<std::shared_ptr<Basis_State>> basis_;
        CSR_Matrix<COMPLEX> H_;
        // CSR_Matrix<COMPLEX> eigenvectors_;
        // std::vector<double> eigenvalues_;
        std::vector<std::pair<double, CSR_Matrix<COMPLEX>>> decoherence_;
        // CSR_Matrix<COMPLEX> H_EXP_;
        // double dt_exp_ = 0;
};

template<typename StateType>
class CSR_H_by_Operator: public CSR_Hamiltonian {
    public:
        explicit CSR_H_by_Operator(const State<StateType>& init_state, const Operator<StateType>& H_op,
                                     const std::vector<std::pair<double, Operator<StateType>>>& decoherence = {});
        //BasisType<StateType> get_original_basis() const {
        //    return basis_original_;
        //}
    private:
            //BasisType<StateType> basis_original_;
};

template<typename StateType>
CSR_H_by_Operator<StateType>::CSR_H_by_Operator(const State<StateType>& init_state, const Operator<StateType>& H_op,
                                     const std::vector<std::pair<double, Operator<StateType>>>& decoherence) {
    std::vector<Operator<StateType>> dec_tmp;
    for (const auto& p: decoherence) {
        dec_tmp.push_back(p.second);
    }

    auto basis_original = sort_basis(State_Graph<StateType>(init_state, H_op, dec_tmp).get_basis());
    State<StateType> basis_map(basis_original);
    basis_ = convert_to<StateType>(basis_original);

    size_t dim = basis_original.size();

    std::vector<COMPLEX> vals;
    std::vector<ILP_TYPE> ia({0});
    std::vector<ILP_TYPE> ja;
    ILP_TYPE vals_count = 0;

    for (auto state: basis_original) {
        State<StateType> res_state(H_op.run(State<StateType>(*state)));


        // std::cout << state->to_string() << " : " << res_state.to_string() << std::endl;
        res_state.set_sorted(true);
        for (auto p: res_state.state_map()) {
            auto idx = basis_map.get_index(p.first);
            vals.emplace_back(std::conj(res_state[p.second]));
            vals_count++;
            ja.emplace_back(idx);
        }

        ia.emplace_back(vals_count);

        // show_vector(vals);
        // show_vector(ia);
        // show_vector(ja);
    }

    H_ = CSR_Matrix<COMPLEX>(ia.size() - 1, basis_original.size(), vals, ia, ja);

    H_.sort_ja();

    for (const auto& p: decoherence) {
        CSR_Matrix<COMPLEX> A(std::move(operator_to_matrix_csr<StateType>(p.second, basis_original)));
        A.sort_ja();
        decoherence_.push_back(std::make_pair(p.first, std::move(A)));
    }
}

// template<typename StateType>
// class CSR_H_by_Scalar_Product: public Hamiltonian {
//     public:
//         explicit H_by_Scalar_Product(const State<StateType>& init_state, 
//                                      const std::function<COMPLEX(const StateType& i, const StateType& j)>& func);
//         explicit H_by_Scalar_Product(const State<StateType>& init_state, 
//                                      const std::function<COMPLEX(const StateType& i, const StateType& j)>& func,
//                                      const BasisType<StateType>& basis) {
//             generate_scalar_product_H(init_state, func, basis);
//         }
//     private:
//         void generate_scalar_product_H(const State<StateType>& init_state,
//                                               const std::function<COMPLEX(const StateType& i, const StateType& j)>& func,
//                                               const BasisType<StateType>& basis) {
//             this->basis_ = convert_to<StateType>(basis);
//             auto basis_sorted = sort_basis(basis);

//             auto matrix_func = std::function<COMPLEX(size_t i, size_t j)>{
//                 [&func, &basis_sorted](size_t i, size_t j) {
//                     return func(*(basis_sorted[i]), *(basis_sorted[j]));
//                 }
//             };

//             this->H_ = Matrix<COMPLEX>(C_STYLE, basis.size(), basis.size(), matrix_func);
//         }
// };

// template<typename StateType>
// CSR_H_by_Scalar_Product<StateType>::H_by_Scalar_Product(const State<StateType>& init_state,
//                                                     const std::function<COMPLEX(const StateType& i, const StateType& j)>& func) {
//     // auto basis = generate_full_basis(init_state);
//     auto zero_state = init_state(0);
//     zero_state->set_zero();
//     BasisType<StateType> basis;
//     basis.insert(std::make_shared(StateType(*zero_state)));

//     StateType cur_state(*zero_state);
//     bool is_not_max = true;

//     while(is_not_max) {
//         is_not_max = false;
//         for (size_t i = 0; i < cur_state.qudits_count() && !is_not_max; i++) {
//             auto cur_qudit = cur_state.get_qudit(i);
//             if(cur_qudit != cur_state.get_max_val(i)) {
//                 cur_state.set_qudit(cur_qudit + 1, i);
//                 is_not_max = true;

//                 for (size_t j = i; j != 0; j--) {
//                     cur_state.set_qudit(0, j - 1);
//                 }
//             }
//         }

//         basis.insert(std::make_shared(StateType(cur_state)));
//     }

//     generate_scalar_product_H(init_state, func, basis);
// }

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

class CSR_H_TCH : public CSR_H_by_Operator<TCH_State> {
    public:
        explicit CSR_H_TCH(const State<TCH_State>& state);
};

} // namespace QComputations
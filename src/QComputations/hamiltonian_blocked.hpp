#ifdef ENABLE_MPI
#pragma once
#include "graph.hpp"
#include "mpi_functions.hpp"
#include "blocked_matrix.hpp"
#include <complex>
#include <functional>
#include <string>
#include <set>
#include "state.hpp"
#include "config.hpp"
#include "quantum_operators.hpp"

namespace QComputations {

namespace {
    using COMPLEX = std::complex<double>;
}

class BLOCKED_Hamiltonian {
    public:
        explicit BLOCKED_Hamiltonian() = default;
        size_t n() const { return H_.n(); }
        size_t size() const { return H_.n(); }
        size_t n_loc() const { return H_.local_n(); }
        size_t m_loc() const { return H_.local_m(); }
        // TCH_State grid() const { return grid_; }
        // TCH_State get_grid() const { return grid_; }
        // void set_grid(const TCH_State& grid) { grid_ = grid; }
        ILP_TYPE ctxt() const { return H_.ctxt(); }
        std::vector<std::shared_ptr<Basis_State>> get_basis() const { return basis_; }
        std::vector<std::pair<double, BLOCKED_Matrix<COMPLEX>>> get_decoherence() const { return decoherence_;}
        void write_to_csv_file(const std::string& filename) const { H_.write_to_csv_file(filename); }

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

        BLOCKED_Matrix<COMPLEX> virtual eigenvectors() {
            this->eigen();
            return eigenvectors_;
        }

        //COMPLEX operator() (size_t i, size_t j) const { return H_(i, j); }
        void show(size_t width = QConfig::instance().width(), ILP_TYPE root_id = mpi::ROOT_ID) const { H_.show(width, root_id); }
        void print_distributed(const std::string& name) const { H_.print_distributed(name); }
        Matrix<COMPLEX> get_local_matrix() const { return H_.get_local_matrix(); }
        BLOCKED_Matrix<COMPLEX> get_blocked_matrix() const { return H_; }
    protected:
        bool is_calculated_eigen_ = false;
        std::vector<std::shared_ptr<Basis_State>> basis_;
        BLOCKED_Matrix<COMPLEX> H_;
        BLOCKED_Matrix<COMPLEX> eigenvectors_;
        std::vector<double> eigenvalues_;
        std::vector<std::pair<double, BLOCKED_Matrix<COMPLEX>>> decoherence_;
        // TCH_State grid_;
};

/*

class BLOCKED_H_TC : public BLOCKED_Hamiltonian {
    public:
        explicit BLOCKED_H_TC(ILP_TYPE ctxt, const State& state);
};

class BLOCKED_H_JC : public BLOCKED_Hamiltonian {
    public:
        explicit BLOCKED_H_JC(ILP_TYPE ctxt, const State& state);
};

class BLOCKED_H_by_func: public BLOCKED_Hamiltonian {
    public:
        explicit BLOCKED_H_by_func(ILP_TYPE ctxt, size_t n, std::function<COMPLEX(size_t, size_t)> func);
    private:
        std::function<COMPLEX(size_t, size_t)> func_;
};
*/

template<typename StateType>
class BLOCKED_H_by_Operator: public BLOCKED_Hamiltonian {
    public:
        explicit BLOCKED_H_by_Operator(ILP_TYPE ctxt, const State<StateType>& init_state, const Operator<StateType>& H_op,
                                     const std::vector<std::pair<double, Operator<StateType>>>& decoherence = {});
};

template<typename StateType>
BLOCKED_H_by_Operator<StateType>::BLOCKED_H_by_Operator(ILP_TYPE ctxt, const State<StateType>& init_state, const Operator<StateType>& H_op,
                                     const std::vector<std::pair<double, Operator<StateType>>>& decoherence) {
    std::vector<Operator<StateType>> dec_tmp;
    for (const auto& p: decoherence) {
        dec_tmp.push_back(p.second);
    }

    auto basis = State<StateType>(State_Graph<StateType>(init_state, H_op, dec_tmp).get_basis());
    auto sorted_basis = basis.get_basis();
    basis_ = convert_to<StateType>(sorted_basis);

    size_t size = basis_.size();

    ILP_TYPE proc_rows, proc_cols, myrow, mycol, NB, MB;
    mpi::blacs_gridinfo(ctxt, proc_rows, proc_cols, myrow, mycol);

    NB = size / proc_rows;

    MB = size / proc_cols;

    if (NB == 0) {
        NB = 1;
    }

    if (MB == 0) {
        MB = 1;
    }

    NB = std::min(NB, MB);
    MB = NB;

    H_ = BLOCKED_Matrix<COMPLEX>(ctxt, HE, size, size, COMPLEX(0, 0), NB, MB);

    for (size_t j = 0; j < H_.local_m(); j++) {
        size_t global_state_index = H_.get_global_col(j);
        // auto state_from = get_state_from_basis<StateType>(basis, global_state_index);
        auto state_from = sorted_basis[global_state_index];
        auto res_state = H_op.run(State<StateType>(state_from));

        size_t index = 0;
        auto sorted_basis = res_state.get_basis();
        for (auto state: sorted_basis) {
            // auto cur_global_row = get_index_state_in_basis(*state, basis);
            auto cur_global_row = basis.get_index(state);
            if (H_.is_my_elem_row(cur_global_row)) {
                H_(H_.get_local_row(cur_global_row), j) = res_state[index];
            }

            index++;
        }
    }

    /*
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
    //H_ = BLOCKED_Matrix<COMPLEX>(ctxt, HE, size, size, func, NB, MB);
    for (const auto& p: decoherence) {
        auto A = operator_to_matrix<StateType>(H_.ctxt(), p.second, sorted_basis);
        //std::cout << "A HAMILTONIAN: " << A.matrix_type() << std::endl;
        decoherence_.push_back(std::make_pair(p.first, A));
        //for (const auto& w: decoherence_) {
        //    std::cout << "A HAM AFTER: " << w.second.matrix_type() << std::endl;
        //}
    }
}

template<typename StateType>
class BLOCKED_H_by_Scalar_Product: public BLOCKED_Hamiltonian {
    public:
        explicit BLOCKED_H_by_Scalar_Product(ILP_TYPE ctxt, const State<StateType>& init_state, 
                                     const std::function<COMPLEX(const StateType& i, const StateType& j)>& func,
                                     BasisType<StateType> basis = {});
};

template<typename StateType>
BLOCKED_H_by_Scalar_Product<StateType>::BLOCKED_H_by_Scalar_Product(ILP_TYPE ctxt, const State<StateType>& init_state,
                                                                    const std::function<COMPLEX(const StateType& i, const StateType& j)>& func,
                                                                    BasisType<StateType> basis) {
    if (basis.empty()) {
        auto zero_state = init_state(0);
        zero_state->set_zero();
        basis.insert(std::shared_ptr<StateType>(new StateType(*zero_state)));

        StateType cur_state = *zero_state;

        for (ILP_TYPE i = 0; i < cur_state.qudits_count(); i++) {
            if (cur_state.get_max_val(i) > cur_state.get_qudit(i)) {
                cur_state.set_qudit(cur_state.get_qudit(i) + 1, i);
                basis.insert(std::shared_ptr<StateType>(new StateType(cur_state)));

                for (ILP_TYPE j = i - 1; j >= 0; j--) {
                    cur_state.set_qudit(0, j);
                }

                i = 0;
            }
        }
    }

    basis_ = convert_to<StateType>(basis);

    size_t size = basis_.size();

    ILP_TYPE proc_rows, proc_cols, myrow, mycol, NB, MB;
    mpi::blacs_gridinfo(ctxt, proc_rows, proc_cols, myrow, mycol);

    NB = size / proc_rows;

    MB = size / proc_cols;

    if (NB == 0) {
        NB = 1;
    }

    if (MB == 0) {
        MB = 1;
    }

    NB = std::min(NB, MB);
    MB = NB;

    auto matrix_func = std::function<COMPLEX(size_t i, size_t j)>{
        [&func, &basis](size_t i, size_t j) {
            return func(*get_state_from_basis(basis, i), *get_state_from_basis(basis, j));
        }
    };

    H_ = BLOCKED_Matrix<COMPLEX>(ctxt, HE, size, size, matrix_func, NB, MB);
}

class BLOCKED_H_TCH : public BLOCKED_H_by_Operator<TCH_State> {
    public:
        explicit BLOCKED_H_TCH(ILP_TYPE ctxt, const State<TCH_State>& state);
};

/*
class BLOCKED_H_TCH_EXC: BLOCKED_Hamiltonian {
    public:
        explicit BLOCKED_H_TCH_EXC(ILP_TYPE ctxt, const EXC_State& state);
};
*/

} // namespace QComputations

#endif

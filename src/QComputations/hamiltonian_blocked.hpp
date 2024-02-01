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
        CHE_State grid() const { return grid_; }
        CHE_State get_grid() const { return grid_; }
        void set_grid(const CHE_State& grid) { grid_ = grid; }
        ILP_TYPE ctxt() const { return H_.ctxt(); }
        std::set<Basis_State> get_basis() const { return basis_; }
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
        std::set<Basis_State> basis_;
        BLOCKED_Matrix<COMPLEX> H_;
        BLOCKED_Matrix<COMPLEX> eigenvectors_;
        std::vector<double> eigenvalues_;
        std::vector<std::pair<double, BLOCKED_Matrix<COMPLEX>>> decoherence_;
        CHE_State grid_;
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
    std::cout << "HERE-1\n";
    for (const auto& p: decoherence) {
        dec_tmp.push_back(p.second);
    }

    std::cout << "HERE0\n";
    auto basis = State_Graph<StateType>(init_state, H_op, dec_tmp).get_basis();
    std::cout << "HERE00\n";
    basis_ = convert_to<StateType>(basis);

    std::cout << "HERE000\n";
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

    std::cout << "HERE1\n";
    std::function<COMPLEX(size_t i, size_t j)> func = {
        [&basis, &H_op](size_t i, size_t j) {
            std::cout << "HERE2 - " << i << " " << j << std::endl;
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

    std::cout << "HERE3\n";
    H_ = BLOCKED_Matrix<COMPLEX>(ctxt, HE, size, size, func);
    std::cout << "HERE4\n";
    for (const auto& p: decoherence) {
        auto A = BLOCKED_Matrix<COMPLEX>(operator_to_matrix<StateType>(H_.ctxt(), p.second, basis));
        decoherence_.push_back(std::make_pair(p.first, A));
    }
    std::cout << "HERE5\n";
}

class BLOCKED_H_TCH : public BLOCKED_H_by_Operator<CHE_State> {
    public:
        explicit BLOCKED_H_TCH(ILP_TYPE ctxt, const State<CHE_State>& state);
};

/*
class BLOCKED_H_TCH_EXC: BLOCKED_Hamiltonian {
    public:
        explicit BLOCKED_H_TCH_EXC(ILP_TYPE ctxt, const EXC_State& state);
};
*/

} // namespace QComputations

#endif

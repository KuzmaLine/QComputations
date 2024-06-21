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
}

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
        std::vector<std::pair<double, Matrix<COMPLEX>>> get_decoherence() const { return decoherence_;}
        //void write_to_csv_file(const std::string& filename) const { H_.write_to_csv_file(filename); }

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
        BasisType<Basis_State> basis_;
        Matrix<COMPLEX> H_;
        Matrix<COMPLEX> eigenvectors_;
        std::vector<double> eigenvalues_;
        std::vector<std::pair<double, Matrix<COMPLEX>>> decoherence_;
        TCH_State grid_;
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

    size_t dim = basis.size();
    H_ = Matrix<COMPLEX>(C_STYLE, dim, dim, COMPLEX(0, 0));

    for (auto state: basis) {
        size_t col_state = 0;
        auto res_state = H_op.run(State<StateType>(*state));

        size_t index = 0;
        for (auto state_res: res_state.state_components()) {
            H_[get_index_state_in_basis(*state_res, basis)][col_state] = res_state[index++];
        }

        col_state++;
    }

    //H_ = Matrix<COMPLEX>(C_STYLE, size, size, func);
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
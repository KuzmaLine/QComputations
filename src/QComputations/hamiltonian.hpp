#pragma once
#include <vector>
#include <functional>
#include <complex>
#include <iostream>
#include "state.hpp"
#include "config.hpp"
#include "matrix.hpp"
#include "quantum_operators.hpp"

namespace QComputations {

namespace {
    typedef std::complex<double> COMPLEX;
}

std::set<CHE_State> define_basis_of_hamiltonian(const CHE_State& grid);

class Hamiltonian {
    public:
        Hamiltonian() {};
        Hamiltonian(const Hamiltonian& H) = default;
        size_t size() const { return H_.size(); }
        std::set<Basis_State> get_basis() const { return basis_; }     // Return basis of Hamiltonian
        CHE_State get_grid() const { return grid_; }                 // Return grid (if it included)
        void show(const size_t width = QConfig::instance().width()) const;
        Matrix<COMPLEX> get_matrix() const { return H_; }
        std::pair<std::vector<double>, Matrix<COMPLEX>> eigen(); // Find eigenvalues and eigenvectors (functions.hpp)
        COMPLEX get_leak(size_t cavity_id) { return grid_.get_leak_gamma(cavity_id); } // (state.hpp)
        void set_leak(size_t cavity_id, COMPLEX gamma) { grid_.set_leak_for_cavity(cavity_id, gamma); } // (state.hpp)
    protected:
        bool is_eigen_ = false;
        CHE_State grid_;
        std::set<Basis_State> basis_;
        Matrix<COMPLEX> H_ = Matrix<COMPLEX>(DEFAULT_MATRIX_STYLE, 0, 0);
        Matrix<COMPLEX> eigenvectors_;
        std::vector<double> eigenvalues_;
        std::vector<std::pair<double, Operator<Basis_State>>> decoherence_;
        Operator<Basis_State> operator_;
};

class H_by_func : public Hamiltonian {
    public:
        H_by_func(size_t n, std::function<COMPLEX(size_t, size_t)> func);
        void set_basis(const std::set<Basis_State>& basis) { basis_ = basis; }
        void set_grid(const CHE_State& grid) { grid_ = grid; }
    private:
        std::function<COMPLEX(size_t, size_t)> func_;
};

class H_by_Matrix : public Hamiltonian {
    public:
        H_by_Matrix(const Matrix<COMPLEX>& H) { H_ = H; }
        void set_basis(const std::set<Basis_State>& basis) { basis_ = basis; }
        void set_grid(const CHE_State& grid) { grid_ = grid; }
};

// NEED UPDATE
class H_JC : public Hamiltonian {
    public:
        explicit H_JC(const CHE_State& state);  // Генерируется по умолчанию в RWA приближении
        void make_exact();                  // Делает гамильтониан точным
};

// NEED UPDATE
class H_TC : public Hamiltonian {
    public:
        explicit H_TC(const CHE_State& state);
    private:
        size_t n_;
        size_t m_;
};

class H_TCH : public Hamiltonian {
    public:
        H_TCH(const CHE_State& init_state);
};

} // namespace QComputations
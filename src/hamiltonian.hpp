#pragma once
#include <vector>
#include <functional>
#include <complex>
#include <iostream>
#include "state.hpp"
#include "config.hpp"
#include "matrix.hpp"

namespace {
    typedef std::complex<double> COMPLEX;
}

class Hamiltonian {
    public:
        Hamiltonian() {};
        Hamiltonian(const Hamiltonian& H) = default;
        size_t size() const { return H_.size(); }
        std::set<State> get_basis() const { return basis_; }
        State get_init_state() const { return init_state_; }
        void show(const size_t width = 10) const;
        Matrix<COMPLEX> get_matrix() const { return H_; }
        std::pair<std::vector<double>, Matrix<COMPLEX>> eigen();
        COMPLEX get_leak(size_t cavity_id) { return init_state_.get_leak_gamma(cavity_id); }
        void set_leak(size_t cavity_id, COMPLEX gamma) { init_state_.set_leak_for_cavity(cavity_id, gamma); }
    protected:
        bool is_eigen_ = false;
        State init_state_;
        std::set<State> basis_;
        Matrix<COMPLEX> H_;
        Matrix<COMPLEX> eigenvectors_;
        std::vector<double> eigenvalues_;
};

class H_by_func : public Hamiltonian {
    public:
        H_by_func(size_t n, std::function<COMPLEX(size_t, size_t)> func);
    private:
        size_t n_;
        std::function<COMPLEX(size_t, size_t)> func_;
};

class H_JC : public Hamiltonian {
    public:
        explicit H_JC(const State& state);
        void make_exact();
    private:
        size_t n_;
};

class H_TC : public Hamiltonian {
    public:
        explicit H_TC(const State& state);
    private:
        size_t n_;
        size_t m_;
        COMPLEX g(int atom_number) {
            return config::g;
        }
};

class H_TCH : public Hamiltonian {
    public:
        H_TCH(const State& init_state);
};

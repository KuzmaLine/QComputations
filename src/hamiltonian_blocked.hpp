#ifdef ENABLE_MPI
#pragma once
#include "mpi_functions.hpp"
#include "blocked_matrix.hpp"
#include <complex>
#include <functional>
#include <string>
#include <set>
#include "state.hpp"
#include "config.hpp"

namespace QComputations {

namespace {
    using COMPLEX = std::complex<double>;
}

class BLOCKED_Hamiltonian {
    public:
        explicit BLOCKED_Hamiltonian() = default;
        size_t n() const { return H_.n(); }
        size_t n_loc() const { return H_.local_n(); }
        size_t m_loc() const { return H_.local_m(); }
        std::set<State> get_basis() const { return basis_; }
        //COMPLEX operator() (size_t i, size_t j) const { return H_(i, j); }
        void show(ILP_TYPE root_id = mpi::ROOT_ID, size_t width = QConfig::instance().width()) const { H_.show(root_id, width); }
        void print_distributed(const std::string& name) const { H_.print_distributed(name); }
        BLOCKED_Matrix<COMPLEX> get_local_matrix() const { return H_.get_local_matrix(); }
    protected:
        std::set<State> basis_;
        BLOCKED_Matrix<COMPLEX> H_;
};

class BLOCKED_H_TCH : public BLOCKED_Hamiltonian {
    public:
        explicit BLOCKED_H_TCH(ILP_TYPE ctxt, const State& state);
        State grid() const { return grid_; }
    private:
        State grid_;
};

class BLOCKED_H_TC : public BLOCKED_Hamiltonian {
    public:
        explicit BLOCKED_H_TC(ILP_TYPE ctxt, const State& state);
    private:
        State grid_;
};

class BLOCKED_H_JC : public BLOCKED_Hamiltonian {
    public:
        explicit BLOCKED_H_JC(ILP_TYPE ctxt, const State& state);
    private:
        State grid_;
};

class BLOCKED_H_by_func: public BLOCKED_Hamiltonian {
    public:
        explicit BLOCKED_H_by_func(ILP_TYPE ctxt, size_t n, std::function<COMPLEX(size_t, size_t)> func);
    private:
        std::function<COMPLEX(size_t, size_t)> func_;
};

} // namespace QComputations

#endif

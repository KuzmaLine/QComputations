#ifdef ENABLE_MPI
#ifdef ENABLE_CLUSTER

#include "hamiltonian_blocked.hpp"
#include "quantum_operators.hpp"
#include "hamiltonian.hpp"
#include <cassert>

namespace QComputations {

namespace {
    using COMPLEX = std::complex<double>;
}

BLOCKED_H_TCH::BLOCKED_H_TCH(ILP_TYPE ctxt, const State& grid) {
    grid_ = grid;
    auto basis = define_basis_of_hamiltonian(grid);
    basis_ = basis;

    size_t size = basis_.size();

    std::function<COMPLEX(size_t i, size_t j)> func = {
        [&basis, &grid](size_t i, size_t j) {
            auto state_from = get_elem_from_set(basis, i);
            auto state_to = get_elem_from_set(basis, j);

            return TCH_ADD(state_from, state_to, grid);
        }
    };

    H_ = BLOCKED_Matrix<COMPLEX>(ctxt, HE, size, size, func);
}

BLOCKED_H_TC::BLOCKED_H_TC(ILP_TYPE ctxt, const State& grid) {
    assert(grid.cavities_count() == 1);

    grid_ = grid;
    auto basis = define_basis_of_hamiltonian(grid);
    basis_ = basis;

    size_t size = basis_.size();

    std::function<COMPLEX(size_t i, size_t j)> func = {
        [&basis, &grid](size_t i, size_t j) {
            auto state_from = get_elem_from_set(basis, i);
            auto state_to = get_elem_from_set(basis, j);

            return TC_ADD(state_from, state_to, grid);
        }
    };

    H_ = BLOCKED_Matrix<COMPLEX>(ctxt, HE, size, size, func);
}

BLOCKED_H_JC::BLOCKED_H_JC(ILP_TYPE ctxt, const State& grid) {
    assert(grid.cavities_count() == 1);

    grid_ = grid;
    auto basis = define_basis_of_hamiltonian(grid);
    basis_ = basis;

    size_t size = basis_.size();

    std::function<COMPLEX(size_t i, size_t j)> func = {
        [&basis, &grid](size_t i, size_t j) {
            auto state_from = get_elem_from_set(basis, i);
            auto state_to = get_elem_from_set(basis, j);

            return JC_ADD(state_from, state_to, grid);
        }
    };

    H_ = BLOCKED_Matrix<COMPLEX>(ctxt, HE, size, size, func);
}

} // namespace QComputations

#endif
#endif

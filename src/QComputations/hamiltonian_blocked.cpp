#ifdef ENABLE_MPI
#ifdef ENABLE_CLUSTER

#include "hamiltonian_blocked.hpp"
#include "quantum_operators.hpp"
#include "hamiltonian.hpp"
#include <cassert>
#include "graph.hpp"

namespace QComputations {

namespace {
    using COMPLEX = std::complex<double>;
}

namespace {
    Operator<CHE_State> H_TCH_OP() {
        using OpType = Operator<CHE_State>;

        OpType my_H;
        my_H = my_H + OpType(photons_count) + OpType(atoms_exc_count) + OpType(exc_relax_atoms) + OpType(photons_transfer);

        return my_H;
    }
    
    std::vector<std::pair<double, Operator<CHE_State>>> decs(const State<CHE_State>& state) {
        using OpType = Operator<CHE_State>;

        auto st = *(state.get_state_components().begin());
        std::vector<std::pair<double, OpType>> dec;

        for (size_t i = 0; i < st.cavities_count(); i++) {
            if (!is_zero(st.get_leak_gamma(i))) {
                OperatorType<CHE_State> a_destroy_i = {[i](const CHE_State& che_state) {
                    return set_qudit(che_state, che_state.n(i) - 1, 0, i) * std::sqrt(che_state.n(i));
                }};

                OpType my_A_out(a_destroy_i);

                dec.emplace_back(std::make_pair(st.get_leak_gamma(i), my_A_out));
            }

            if (!is_zero(st.get_gain_gamma(i))) {
                OperatorType<CHE_State> a_create_i = {[i](const CHE_State& che_state) {
                    return set_qudit(che_state, che_state.n(i) + 1, 0, i) * std::sqrt(che_state.n(i) + 1);
                }};

                OpType my_A_in(a_create_i);

                dec.emplace_back(std::make_pair(st.get_gain_gamma(i), my_A_in));
            }
        }

        return dec;
    }
}

BLOCKED_H_TCH::BLOCKED_H_TCH(ILP_TYPE ctxt, const State<CHE_State>& state):
                        BLOCKED_H_by_Operator<CHE_State>(ctxt, state, H_TCH_OP(), decs(state)) {}

/*
BLOCKED_H_TC::BLOCKED_H_TC(ILP_TYPE ctxt, const State& grid) {
    assert(grid.cavities_count() == 1);

    grid_ = grid;
    //auto basis = define_basis_of_hamiltonian(grid);
    auto basis = State_Graph(grid).get_basis();
    basis_ = basis;

    size_t size = basis_.size();

    std::function<COMPLEX(size_t i, size_t j)> func = {
        [&basis, &grid](size_t i, size_t j) {
            auto state_from = get_elem_from_set(basis, i);
            auto state_to = get_elem_from_set(basis, j);

            return TC_ADD(state_from, state_to, grid);
        }
    };

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

    H_ = BLOCKED_Matrix<COMPLEX>(ctxt, HE, size, size, func);
}

BLOCKED_H_JC::BLOCKED_H_JC(ILP_TYPE ctxt, const State& grid) {
    assert(grid.cavities_count() == 1);

    grid_ = grid;
    //auto basis = define_basis_of_hamiltonian(grid);
    auto basis = State_Graph(grid).get_basis();
    basis_ = basis;

    size_t size = basis_.size();

    std::function<COMPLEX(size_t i, size_t j)> func = {
        [&basis, &grid](size_t i, size_t j) {
            auto state_from = get_elem_from_set(basis, i);
            auto state_to = get_elem_from_set(basis, j);

            return JC_ADD(state_from, state_to, grid);
        }
    };

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

    H_ = BLOCKED_Matrix<COMPLEX>(ctxt, HE, size, size, func);
}
*/

/*
BLOCKED_H_by_Operator::BLOCKED_H_by_Operator(ILP_TYPE ctxt, const State<Basis_State>& init_state, const Operator<Basis_State>& H_op,
                                     const std::vector<std::pair<double, Operator<Basis_State>>>& decoherence) {
    operator_ = H_op;
    decoherence_ = decoherence;

    std::vector<Operator<Basis_State>> dec_tmp;
    for (const auto& p: decoherence) {
        dec_tmp.push_back(p.second);
    }

    auto basis = State_Graph(init_state, H_op, dec_tmp).get_basis();
    basis_ = basis;

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

    std::function<COMPLEX(size_t i, size_t j)> func = {
        [&basis, &H_op](size_t i, size_t j) {
            auto state_from = get_elem_from_set(basis, j);
            auto state_to = get_elem_from_set(basis, i);
            auto res_state = H_op.run(State<Basis_State>(state_from));
            
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

    H_ = BLOCKED_Matrix<COMPLEX>(ctxt, HE, size, size, func);
}
*/

} // namespace QComputations

#endif
#endif

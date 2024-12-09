#ifdef ENABLE_MPI
#ifdef ENABLE_CLUSTER

#include "hamiltonian_blocked.hpp"

#include <cassert>

#include "graph.hpp"
#include "hamiltonian.hpp"
#include "quantum_operators.hpp"

namespace QComputations {

namespace {
using COMPLEX = std::complex<double>;
}

namespace {
Operator<TCH_State> H_TCH_OP() {
    using OpType = Operator<TCH_State>;

    OpType my_H;
    my_H = my_H + OpType(photons_count) + OpType(atoms_exc_count) + OpType(exc_relax_atoms) + OpType(photons_transfer);

    return my_H;
}

std::vector<std::pair<double, Operator<TCH_State>>> decs(const State<TCH_State>& state) {
    using OpType = Operator<TCH_State>;

    auto st = *(state.get_state_components().begin());
    std::vector<std::pair<double, OpType>> dec;

    for (size_t i = 0; i < st->cavities_count(); i++) {
        if (!is_zero(st->get_leak_gamma(i))) {
            std::function<State<TCH_State>(const TCH_State&)> a_destroy_i = {[i](const TCH_State& che_state) {
                return set_qudit(che_state, che_state.n(i) - 1, 0, i) * std::sqrt(che_state.n(i));
            }};

            OpType my_A_out(a_destroy_i);

            dec.emplace_back(std::make_pair(st->get_leak_gamma(i), my_A_out));
        }

        if (!is_zero(st->get_gain_gamma(i))) {
            std::function<State<TCH_State>(const TCH_State&)> a_create_i = {[i](const TCH_State& che_state) {
                return set_qudit(che_state, che_state.n(i) + 1, 0, i) * std::sqrt(che_state.n(i) + 1);
            }};

            OpType my_A_in(a_create_i);

            dec.emplace_back(std::make_pair(st->get_gain_gamma(i), my_A_in));
        }
    }

    return dec;
}
}  // namespace

BLOCKED_H_TCH::BLOCKED_H_TCH(ILP_TYPE ctxt, const State<TCH_State>& state)
    : BLOCKED_H_by_Operator<TCH_State>(ctxt, state, H_TCH_OP(), decs(state)) {}

BLOCKED_H_by_BLOCKED_Matrix::BLOCKED_H_by_BLOCKED_Matrix(const BLOCKED_Matrix<COMPLEX>& H) { H_ = H; }

}  // namespace QComputations

#endif
#endif

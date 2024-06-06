#include <vector>
#include <functional>
#include <complex>
#include <iostream>
#include <iomanip>
#include "hamiltonian.hpp"
#include "config.hpp"
#include "graph.hpp"
#include "functions.hpp"

#ifdef ENABLE_MPI
#include "mpi_functions.hpp"
#endif

namespace QComputations {

namespace {
    typedef std::complex<double> COMPLEX;
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

        for (size_t i = 0; i < st.cavities_count(); i++) {
            if (!is_zero(st.get_leak_gamma(i))) {
                OperatorType<TCH_State> a_destroy_i = {[i](const TCH_State& che_state) {
                    return set_qudit(che_state, che_state.n(i) - 1, 0, i) * std::sqrt(che_state.n(i));
                }};

                OpType my_A_out(a_destroy_i);

                dec.emplace_back(std::make_pair(st.get_leak_gamma(i), my_A_out));
            }

            if (!is_zero(st.get_gain_gamma(i))) {
                OperatorType<TCH_State> a_create_i = {[i](const TCH_State& che_state) {
                    return set_qudit(che_state, che_state.n(i) + 1, 0, i) * std::sqrt(che_state.n(i) + 1);
                }};

                OpType my_A_in(a_create_i);

                dec.emplace_back(std::make_pair(st.get_gain_gamma(i), my_A_in));
            }
        }

        return dec;
    }
}

H_TCH::H_TCH(const State<TCH_State>& state):
                       H_by_Operator<TCH_State>(state, H_TCH_OP(), decs(state)) {}

} // namespace QComputations
#include "quantum_operators.hpp"

#include "functions.hpp"

namespace QComputations {

State<TCH_State> photons_transfer(const TCH_State& st) {
    State<TCH_State> res;

    TCH_State state(st);
    for (size_t i = 0; i < state.cavities_count(); i++) {
        auto neighbours = state.get_neighbours(i);

        for (auto cavity_id : neighbours) {
            if (i < cavity_id) {
                if (state.n(i) != 0) {
                    state.set_n(state.n(i) - 1, i);

                    res += set_qudit(state, state.n(cavity_id) + 1, 0, cavity_id) * state.get_gamma(i, cavity_id) *
                           std::sqrt(state.n(i) + 1) * std::sqrt(state.n(cavity_id) + 1);

                    state.set_n(state.n(i) + 1, i);
                }

                if (state.n(cavity_id) != 0) {
                    state.set_n(state.n(i) + 1, i);

                    res += set_qudit(state, state.n(cavity_id) - 1, 0, cavity_id) * state.get_gamma(cavity_id, i) *
                           std::sqrt(state.n(i)) * std::sqrt(state.n(cavity_id));

                    state.set_n(state.n(i) - 1, i);
                }
            }
        }
    }

    return res;
}

State<TCH_State> exc_relax_atoms(const TCH_State& st) {
    State<TCH_State> res;

    TCH_State state(st);

    for (size_t i = 0; i < state.cavities_count(); i++) {
        if (state.n(i) != 0) {
            state.set_n(state.n(i) - 1, i);
            for (size_t j = 1; j <= state.m(i); j++) {
                if (state.get_qudit(j, i) == 0) {
                    res += set_qudit(state, 1, j, i) * QConfig::instance().g() * std::sqrt(state.n(i) + 1);
                }
            }
            state.set_n(state.n(i) + 1, i);
        }

        for (size_t j = 1; j <= state.m(i); j++) {
            if (state.get_qudit(j, i) == 1) {
                state.set_qudit(0, j, i);
                res += set_qudit(state, state.n(i) + 1, 0, i) * QConfig::instance().g() * std::sqrt(state.n(i) + 1);
                state.set_qudit(1, j, i);
            }
        }
    }

    return res;
}

State<TCH_State> photons_count(const TCH_State& state) {
    State<TCH_State> res(state);
    res[0] = 0;

    for (size_t i = 0; i < state.cavities_count(); i++) {
        res[0] += state.get_qudit(0, i) * QConfig::instance().h() * QConfig::instance().w();
    }

    return res;
}

State<TCH_State> atoms_exc_count(const TCH_State& state) {
    State<TCH_State> res(state);
    res[0] = 0;

    for (size_t i = 0; i < state.cavities_count(); i++) {
        for (size_t j = 1; j <= state.m(i); j++) {
            res[0] += state.get_qudit(j, i) * QConfig::instance().h() * QConfig::instance().w();
        }
    }

    return res;
}

}  // namespace QComputations
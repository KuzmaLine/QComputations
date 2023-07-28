#include "quantum_operators.hpp"

COMPLEX self_energy_photon(const State& state_from, const State& state_to, COMPLEX hw) {
    if (state_from == state_to) {
        COMPLEX res(0);

        for (size_t i = 0; i < state_from.cavities_count(); i++) {
            res += COMPLEX(state_from.n(i));
        }

        return res * hw;
    } else {
        return 0;
    }
}

COMPLEX self_energy_atom(const State& state_from, const State& state_to, COMPLEX hw) {
    if (state_from == state_to) {
        COMPLEX res(0);

        for (size_t i = 0; i < state_from.cavities_count(); i++) {
            res += COMPLEX(state_from[i].up_count());
        }

        return res * hw;
    } else {
        return 0;
    }
}

COMPLEX excitation_atom(const State& state_from, const State& state_to, COMPLEX g) {
    long photon_pos = -1;
    long atom_pos = -1;

    for (size_t i = 0; i < state_from.cavities_count(); i++) {
        if (state_from[i].n() != state_to[i].n()) {
            if (state_from[i].n() > 0 and state_from[i].n() - 1 == state_to[i].n() and photon_pos == -1) {
                photon_pos = i;
            } else {
                return 0;
            }
        }

        for (size_t j = 0; j < state_from[i].m(); j++) {
            if (state_from[i].get_qubit(j) != state_to[i].get_qubit(j)) {
                if (state_from[i].get_qubit(j) == 0 and state_to[i].get_qubit(j) == 1 and atom_pos == -1) {
                    atom_pos = i;
                } else {
                    return 0;
                }
            }
        }
    }

    if (photon_pos == -1 or atom_pos == -1 or atom_pos != photon_pos) {
        return 0;
    }

    //std::cout << "exc - " << photon_pos << " " << state_from.to_string() << " " << state_to.to_string() << std::endl;

    return g * COMPLEX(std::sqrt(state_from[photon_pos].n()));
}

COMPLEX de_excitation_atom(const State& state_from, const State& state_to, COMPLEX g) {
    long photon_pos = -1;
    long atom_pos = -1;

    for (size_t i = 0; i < state_from.cavities_count(); i++) {
        if (state_from[i].n() != state_to[i].n()) {
            if (state_from[i].n() + 1 == state_to[i].n() and photon_pos == -1) {
                photon_pos = i;
            } else {
                return 0;
            }
        }

        for (size_t j = 0; j < state_from[i].m(); j++) {
            if (state_from[i].get_qubit(j) != state_to[i].get_qubit(j)) {
                if (state_from[i].get_qubit(j) == 1 and state_to[i].get_qubit(j) == 0 and atom_pos == -1) {
                    atom_pos = i;
                } else {
                    return 0;
                }
            }
        }
    }

    if (photon_pos == -1 or atom_pos == -1 or atom_pos != photon_pos) {
        return 0;
    }

    //std::cout << "de-exc - " << photon_pos << " " << state_from.to_string() << " " << state_to.to_string() << std::endl;

    return g * COMPLEX(std::sqrt(state_from[photon_pos].n() + 1));
}
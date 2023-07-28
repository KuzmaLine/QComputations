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
        if (state_from.n(i) != state_to.n(i)) {
            if (state_from.n(i) + 1 == state_to.n(i) and photon_pos == -1) {
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

COMPLEX photon_exchange(const State& state_from, const State& state_to, const State& grid) {
    for (size_t i = 0; i < state_from.cavities_count(); i++) {
        for (size_t j = 0; j < state_from.m(i); j++) {
            if (state_from[i].get_qubit(j) != state_to[i].get_qubit(j)) {
                return 0;
            }
        }
    }

    long photon_index_from = -1;
    long photon_index_to = -1;

    //std::cout << state_from.to_string() << " " << state_to.to_string() << std::endl;
    for (size_t i = 0; i < state_from.cavities_count(); i++) {
        if (state_from.n(i) != state_to.n(i)) {
            if (state_from.n(i) > 0 and state_from.n(i) - 1 == state_to.n(i)) {
                photon_index_from = i;

                //std::cout << "FROM\n";
                for (size_t j = i + 1; j < state_to.cavities_count(); j++) {
                    if (state_from.n(j) != state_to.n(j)) {
                        //std::cout << j << " " << state_from.n(j) + 1 << " " << state_to.n(j) << std::endl;
                        if (state_from.n(j) + 1 == state_to.n(j) and photon_index_to == -1) {
                            photon_index_to = j;
                        } else {
                            return 0;
                        }
                    }
                }

                //std::cout << "BREAK_FROM\n";
                break;
            } else if (state_from.n(i) + 1 == state_to.n(i)) {
                photon_index_to = i;

                //std::cout << "TO\n";
                for (size_t j = i + 1; j < state_to.cavities_count(); j++) {
                    if (state_from.n(j) != state_to.n(j)) {
                        //std::cout << j << " " << state_from.n(j) - 1 << " " << state_to.n(j) << std::endl;
                        if (state_from.n(j) > 0 and state_from.n(j) - 1 == state_to.n(j) and photon_index_from == -1) {
                            photon_index_from = j;
                        } else {
                            return 0;
                        }
                    }
                }

                //std::cout << "TO_FROM\n";
                break;
            } else {
                return 0;
            }
        }
    }

    if (photon_index_from == -1 or photon_index_to == -1) {
        return 0;
    }

    //std::cout << "END\n";

    return grid.get_gamma(photon_index_from, photon_index_to) * std::sqrt(state_from.n(photon_index_from)) * std::sqrt(state_from.n(photon_index_to) + 1);
}
#include "quantum_operators.hpp"
#include "functions.hpp"

namespace QComputations {

// ---------------------------- OPEATORS ---------------------------
/*
Basis_State a_destroy(const Basis_State& state, size_t photon_index, size_t cavity_id) {
    auto cur_photons = state.get_qudit(photon_index, cavity_id);

    Basis_State res();
    if (cur_photons != 0) {
        res = state;
        res.set_qudit(cur_photons - 1, photon_index, cavity_id);
    }

    res.set_coef(std::sqrt(cur_photons));

    return res;
}

Basis_State a_create(const Basis_State& state, size_t photon_index, size_t cavity_id) {
    auto cur_photons = state.get_qudit(photon_index, cavity_id);

    Basis_State res();
    if (cur_photons != state.get_max_val) {
        res = state;
        res.set_qudit(cur_photons + 1, photon_index, cavity_id);
    }

    res.set_coef(std::sqrt(cur_photons + 1));
    return res;
}

Basis_State sigma_destroy(const Basis_State& state, size_t atom_index, size_t cavity_id) {
    auto cur_val = state.get_qudit(atom_index, cavity_id);

    Basis_State res();
    if (cur_val != 0) {
        res = state;
        res.set_qudit(cur_val - 1, atom_index, cavity_id);
    }

    return res;
}

Basis_State sigma_create(const Basis_State& state, size_t atom_index, size_t cavity_id) {
    auto cur_val = state.get_qudit(atom_index, cavity_id);

    Basis_State res();
    if (cur_val != state.get_max_val(atom_index, cavity_id)) {
        res = state;
        res.set_qudit(cur_val + 1, atom_index, cavity_id);
    }

    return res;
}

Basis_State photons_count(const Basis_State& state, size_t photon_index, size_t cavity_id) {
    ValType coef = 0;

    coef += state.get_qudit(photon_index, cavity_id);

    auto res = state;
    res.set_coef(coef);
    return res;
}

Basis_State atoms_exc_count(const Basis_State& state, size_t atom_index, size_t cavity_id) {
    ValType coef = 0;

    coef += state.get_qudit(atom_index, cavity_id);

    auto res = state;
    res.set_coef(coef);
    return res;
}

Basis_State check(const Basis_State& state, ValType check_val, size_t qudit_index, size_t group_id) {
    if (state.get_index(qudit_index, group_id) == check_val) {
        return state;
    } else {
        return Basis_State();
    }
}
*/

// OLD

/*
COMPLEX self_energy_photon(const TCH_State& state_from, const TCH_State& state_to, COMPLEX h) {
    if (state_from == state_to) {
        COMPLEX res(0);

        for (size_t i = 0; i < state_from.cavities_count(); i++) {
            res += COMPLEX(state_from.n(i)) * state_from[i].w_ph();
        }

        return res * h;
    } else {
        return 0;
    }
}

COMPLEX self_energy_atom(const TCH_State& state_from, const TCH_State& state_to, COMPLEX h) {
    if (state_from == state_to) {
        COMPLEX res(0);

        for (size_t i = 0; i < state_from.cavities_count(); i++) {
            res += COMPLEX(state_from[i].up_count()) * state_from[i].w_at();
        }

        return res * h;
    } else {
        return 0;
    }
}

COMPLEX excitation_atom(const TCH_State& state_from, const TCH_State& state_to, COMPLEX g) {
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

COMPLEX de_excitation_atom(const TCH_State& state_from, const TCH_State& state_to, COMPLEX g) {
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

COMPLEX photon_exchange(const TCH_State& state_from, const TCH_State& state_to, const TCH_State& grid) {
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

COMPLEX photon_destroy(const TCH_State& state_from, const TCH_State& state_to, COMPLEX gamma) {
    auto tmp_state = state_from;

    bool is_founded = false;

    for (size_t i = 0; i < tmp_state.cavities_count(); i++) {
        tmp_state.set_n(tmp_state.n(i) - 1, i);

        if (tmp_state == state_to) {
            if (!is_founded) {
                is_founded = true;
            } else {
                return 0;
            }
        }

        tmp_state.set_n(tmp_state.n(i) + 1, i);   
    }

    if (is_founded) {
        return gamma;
    }

    return 0;
}

COMPLEX photon_create(const TCH_State& state_from, const TCH_State& state_to, COMPLEX gamma) {
    auto tmp_state = state_from;

    bool is_founded = false;

    for (size_t i = 0; i < tmp_state.cavities_count(); i++) {
        tmp_state.set_n(tmp_state.n(i) + 1, i);

        if (tmp_state == state_to) {
            if (!is_founded) {
                is_founded = true;
            } else {
                return 0;
            }
        }

        tmp_state.set_n(tmp_state.n(i) - 1, i);   
    }

    if (is_founded) {
        return gamma;
    }

    return 0;
}

COMPLEX JC_addition(const TCH_State& state_from, const TCH_State& state_to, COMPLEX g) {
    if (state_from.n(0) - 1 == state_to.n(0) and state_from.get_qubit(0, 0) == 1 and state_to.get_qubit(0, 0) == 0) {
        return g * sqrt(state_from.n(0));
    } else if (state_from.n(0) + 1 == state_to.n(0) and state_from.get_qubit(0, 0) == 0 and state_to.get_qubit(0, 0) == 1) {
        return g * sqrt(state_from.n(0) + 1);
    } else {
        return 0;
    }
}

COMPLEX TCH_ADD(const TCH_State& state_from, const TCH_State& state_to, const TCH_State& grid) {
    COMPLEX res(0);
    res += self_energy_atom(state_from, state_to);
    //std::cout << "Energy_atom PASSED\n";
    res += self_energy_photon(state_from, state_to);
    //std::cout << "Energy_photon PASSED\n";
    res += excitation_atom(state_from, state_to);
    //std::cout << "excitation_atom PASSED\n";
    res += de_excitation_atom(state_from, state_to);
    //std::cout << "de_excitation_atom PASSED\n";
    res += photon_exchange(state_from, state_to, grid);

    return res;
}

COMPLEX TC_ADD(const TCH_State& state_from, const TCH_State& state_to, const TCH_State& grid) {
    COMPLEX res(0);
    res += self_energy_atom(state_from, state_to);
    //std::cout << "Energy_atom PASSED\n";
    res += self_energy_photon(state_from, state_to);
    //std::cout << "Energy_photon PASSED\n";
    res += excitation_atom(state_from, state_to);
    //std::cout << "excitation_atom PASSED\n";
    res += de_excitation_atom(state_from, state_to);
    //std::cout << "de_excitation_atom PASSED\n";

    return res;
}

COMPLEX JC_ADD(const TCH_State& state_from, const TCH_State& state_to, const TCH_State& grid) {
    COMPLEX res(0);
    res += self_energy_atom(state_from, state_to);
    //std::cout << "Energy_atom PASSED\n";
    res += self_energy_photon(state_from, state_to);
    //std::cout << "Energy_photon PASSED\n";
    res += excitation_atom(state_from, state_to);
    //std::cout << "excitation_atom PASSED\n";
    res += de_excitation_atom(state_from, state_to);
    //std::cout << "de_excitation_atom PASSED\n";

    return res;
}
*/

State<TCH_State> photons_transfer(const TCH_State& st) {
    State<TCH_State> res;

    TCH_State state(st);
    for (size_t i = 0; i < state.cavities_count(); i++) {
        auto neighbours = state.get_neighbours(i);

        for (auto cavity_id: neighbours) {
            if (i < cavity_id) {
                if (state.n(i) != 0) {
                    state.set_n(state.n(i) - 1, i);

                    res += set_qudit(state, state.n(cavity_id) + 1, 0, cavity_id) 
                            * state.get_gamma(i, cavity_id) * std::sqrt(state.n(i) + 1)
                            * std::sqrt(state.n(cavity_id) + 1);

                    state.set_n(state.n(i) + 1, i);
                }
    
                if (state.n(cavity_id) != 0) {
                    state.set_n(state.n(i) + 1, i);

                    res += set_qudit(state, state.n(cavity_id) - 1, 0, cavity_id)
                    * state.get_gamma(cavity_id, i) * std::sqrt(state.n(i))
                    * std::sqrt(state.n(cavity_id));

                    state.set_n(state.n(i) - 1, i);
                }

                //std::cout << "RES - " << res.to_string() << std::endl;
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
        //std::cout << state.get_qudit(0, i) << std::endl;
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

} // namespace QComputations
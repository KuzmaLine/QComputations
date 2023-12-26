#pragma once
#include <complex>
#include "state.hpp"
#include "config.hpp"
namespace QComputations {

namespace {
    using ValType = int;
    using COMPLEX = std::complex<double>;
}

class Formule {
    public:
        explicit Formule() = default;
        explicit Formule(const Basis_State& state) : states_(1, state) {}

        void operator+=(const Basis_State& state) { states_.insert(state); }
        std::set<Basis_State> get_states() const { return states_; }
    private:
        std::set<Basis_State> states_;
};

// ---------------------------- OPEATORS ---------------------------
Basis_State a_destroy(const Basis_State& state, size_t photon_index, size_t cavity_id = 0);
Basis_State a_create(const Basis_State& state, size_t photon_index, size_t cavity_id = 0);
Basis_State sigma_destroy(const Basis_State& state, size_t atom_index, size_t cavity_id = 0);
Basis_State sigma_create(const Basis_State& state, size_t atom_index, size_t cavity_id = 0);
Basis_State photons_count(const Basis_State& state, size_t photon_index, size_t cavity_id = 0);
Basis_State atoms_exc_count(const Basis_State& state, size_t atom_index, size_t cavity_id = 0);
Basis_State check(const Basis_State& state, ValType check_val, size_t qudit_index, size_t group_id = 0);

// ------------------------------ OLD_VERSION ---------------------------
// (!!!) ADD DESCRIPTION
COMPLEX self_energy_photon(const State& state_from, const State& state_to, COMPLEX h = QConfig::instance().h());
COMPLEX self_energy_atom(const State& state_from, const State& state_to, COMPLEX h = QConfig::instance().h());
COMPLEX excitation_atom(const State& state_from, const State& state_to, COMPLEX g = QConfig::instance().g());
COMPLEX de_excitation_atom(const State& state_from, const State& state_to, COMPLEX g = QConfig::instance().g());
COMPLEX photon_exchange(const State& state_from, const State& state_to, const State& grid);
COMPLEX photon_destroy(const State& state_from, const State& state_to, COMPLEX gamma = COMPLEX(1));
COMPLEX photon_create(const State& state_from, const State& state_to, COMPLEX gamma = COMPLEX(1));
COMPLEX JC_addition(const State& state_from, const State& state_to, COMPLEX g = QConfig::instance().g());

COMPLEX TCH_ADD(const State& state_from, const State& state_to, const State& grid);
COMPLEX TC_ADD(const State& state_from, const State& state_to, const State& grid);
COMPLEX JC_ADD(const State& state_from, const State& state_to, const State& grid);

} // namespace QComputations
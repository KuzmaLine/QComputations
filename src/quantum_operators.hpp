#pragma once
#include <complex>
#include "state.hpp"
#include "config.hpp"
namespace QComputations {

namespace {
    using COMPLEX = std::complex<double>;
}

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
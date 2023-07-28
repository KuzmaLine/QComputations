#pragma once
#include <complex>
#include "state.hpp"
#include "config.hpp"

namespace {
    using COMPLEX = std::complex<double>;
}

COMPLEX self_energy_photon(const State& state_from, const State& state_to, COMPLEX hw = config::h * config::w);
COMPLEX self_energy_atom(const State& state_from, const State& state_to, COMPLEX hw = config::h * config::w);
COMPLEX excitation_atom(const State& state_from, const State& state_to, COMPLEX g = config::g);
COMPLEX de_excitation_atom(const State& state_from, const State& state_to, COMPLEX g = config::g);
COMPLEX photon_exchange(const State& state_from, const State& state_to, const State& grid);
COMPLEX JC_addition(const State& state_from, const State& state_to, COMPLEX g = config::g);
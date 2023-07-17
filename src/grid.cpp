#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <complex>
#include "state.hpp"
#include "additional_operators.hpp"

State::State(const std::vector<E_LEVEL>& state) : x_size_(1), y_size_(1),z_size_(1), n_(0) {
    pols_with_atoms_.insert(0);
    grid_states_[0] = state;
}

State::State(const std::string& init_state) {
}

State::State(const State& state) : x_size_(state.x_size_), y_size_(state.y_size_),z_size_(state.z_size_), grid_states_(state.grid_states_), pols_with_atoms_(state.pols_with_atoms_), n_(state.n_) {}

size_t State::hash() const {
    std::hash<std::map<int, std::vector<int>>> grid_state_hash;

    return grid_state_hash(grid_states_);
}

std::string State::to_string() const {
    return "???";
}

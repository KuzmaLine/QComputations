#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <complex>
#include "state.hpp"
#include "additional_operators.hpp"
#include "functions.hpp"

namespace {
    using E_LEVEL = int;
    using vec_levels = std::vector<E_LEVEL>;

    constexpr int START_INDEX = 1;

    vec_levels index_to_state(size_t m, size_t state_in_num) {
        vec_levels state(m);

        for (int i = m - 1; i >= 0; i--) {
            state[i] = state_in_num % 2;
            state_in_num >>= 1;
        }

        return state;
    }
}

State::State(size_t n, size_t m, size_t state): n_(n), state_(index_to_state(m, state)) {}

State::State(const std::vector<E_LEVEL>& state) : n_(0), state_(state) {
    auto size = this->size();
    vector_of_atoms_state_ = vec_complex(std::pow(2, size), 0);
    vector_of_atoms_state_[get_index_from_state(state_)] = 1;
}

State::State(const std::string& str_state) {
    int i = START_INDEX;

    size_t n = 0;
    while(str_state.at(i) != '>') {
        n *= 10;
        n += str_state.at(i) - '0';
        i++;
    }

    n_ = n;

    if (i != str_state.length() - 1) {
        i += 2;

        while (str_state[i] != '>') {
            state_.emplace_back(str_state[i] - '0');
            i++;
        }
    }
}

size_t State::get_index() const {
    auto max_num_atoms = std::pow(2, this->size());

    return n_ * max_num_atoms + get_index_from_state(state_);
}

size_t State::get_index(const std::set<State>& basis) const {
    size_t index = 0;
    for (const auto& state: basis) {
        if (state == *this) return index;
        index++;
    }

    return -1;
}

size_t State::hash() const {
    std::hash<vec_levels> state_hash;
    return state_hash(state_) ^ n_;
}

std::string State::to_string() const {
    std::string str_state = "|" + std::to_string(n_) + ">|";
    for (const auto& bit: state_) {
        str_state += std::to_string(bit);
    }

    str_state += ">";
    return str_state;
}

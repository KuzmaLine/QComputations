#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <complex>
#include "cavity_state.hpp"
#include "additional_operators.hpp"
#include "functions.hpp"
#include "config.hpp"

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

namespace QComputations {

Cavity_State::Cavity_State(size_t n, size_t m, E_LEVEL e_levels_count): e_levels_count_(e_levels_count) {
    w_ph_ = Matrix<double>(C_STYLE, e_levels_count_, e_levels_count_, QConfig::instance().w());
    n_ = Matrix<size_t>(C_STYLE, e_levels_count_, e_levels_count_, 0);
    n_[0][1] = n;
    w_at_[0] = 0;
    state_ = index_to_state(m, 0);

    max_energy_ = get_energy();
}

Cavity_State::Cavity_State(const Matrix<size_t>& n, const std::vector<E_LEVEL>& state, E_LEVEL e_levels_count): n_(n),
                                    e_levels_count_(e_levels_count), w_at_(e_levels_count, QConfig::instance().w()) {
    w_ph_ = Matrix<double>(C_STYLE, e_levels_count_, e_levels_count_, QConfig::instance().w());
    w_at_[0] = 0;
    if (state.size() != 0) {
        state_ = state;
    }

    max_energy_ = get_energy();
}

Cavity_State::Cavity_State(size_t n, const std::vector<E_LEVEL>& state, E_LEVEL e_levels_count): e_levels_count_(e_levels_count), w_at_(e_levels_count_, QConfig::instance().w()) {
    w_ph_ = Matrix<double>(C_STYLE, e_levels_count_, e_levels_count_, QConfig::instance().w());
    n_ = Matrix<size_t>(C_STYLE, e_levels_count_, e_levels_count_, 0);
    n_[0][1] = n;
    w_at_[0] = 0;
    if (state.size() != 0) {
        state_ = state;
    }

    max_energy_ = get_energy();
}

Cavity_State::Cavity_State(const std::string& str_state, E_LEVEL e_levels_count): e_levels_count_(e_levels_count) {
    int i = START_INDEX;

    size_t n = 0;
    while(str_state.at(i) != '>') {
        n *= 10;
        n += str_state.at(i) - '0';
        i++;
    }

    n_ = Matrix<size_t>(C_STYLE, 2, 2, 0);
    n_[0][1] = n;

    if (i != str_state.length() - 1) {
        i += 2;

        while (str_state[i] != '>') {
            state_.emplace_back(str_state[i] - '0');
            i++;
        }
    }
}

size_t Cavity_State::up_count() const {
    size_t res = 0;
    for (const auto& st: state_) {
        res += st;
    }

    return res;
}

size_t Cavity_State::variants_of_state_count(size_t N) const {
    size_t m = this->m();
    size_t res = 0;

    for (long i = 0; i <= N; i++) {
        for (long k = std::max(long(0), i - long(m)); k <= i; k++) {
            res += Ck_n(i - k, m);
        }
    }

    return res;
}

size_t Cavity_State::get_atoms_index() const {
    return get_index_from_state(state_);
}

size_t Cavity_State::get_energy() const {
    size_t sum = 0;

    for (size_t i = 0; i < e_levels_count_; i++) {
        for (size_t j = i + 1; j < e_levels_count_; j++) {
            sum += n_[i][j];
        }
    }
    return sum + this->up_count();
}

// TMP
size_t Cavity_State::get_index() const {
    auto max_num_atoms = std::pow(2, this->m());

    return n_[0][1] * max_num_atoms + get_index_from_state(state_);
}

size_t Cavity_State::get_index(const std::set<Cavity_State>& basis) const {
    size_t index = 0;
    for (const auto& state: basis) {
        if (state == *this) return index;
        index++;
    }

    return -1;
}

bool Cavity_State::is_in_basis(const std::set<Cavity_State>& basis) const {
    for (const auto& state: basis) {
        if (state == *this) return true;
    }

    return false;
}

size_t Cavity_State::hash() const {
    std::hash<vec_levels> state_hash;
    auto res = state_hash(state_);

    for (size_t i = 0; i < e_levels_count_; i++) {
        for (size_t j = i + 1; j < e_levels_count_; j++) {
            res ^= n_[i][j];
        }
    }

    return res;
}

std::string Cavity_State::to_string() const {
    std::string str_state = "|" + std::to_string(n_[0][1]) + ">";

    if (state_.size() != 0) {
        str_state += "|";
        for (const auto& bit: state_) {
            str_state += std::to_string(bit);
        }

        str_state += ">";
    }
    return str_state;
}

} // namespace QComputations
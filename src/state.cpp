#include "state.hpp"
#include "functions.hpp"

// REWRITE TO REGEXP
State::State(const std::string& grid_state, const std::string& format) : gamma_(0, 0, 0) {
    size_t format_index = 0;
    size_t left_length = 0, middle_length = 0, right_length = 0;
    while (format[format_index] != 'n' and
           format[format_index] != 'N') {
            left_length++;
            format_index++;
    }

    N_ = 0;

    std::vector<CavityId> n;
    std::vector<std::vector<E_LEVEL>> m;
    if (format[format_index] == 'N') {
        char sep = format[format_index + 1];
        size_t Cavity_count = 0;
        size_t& index = left_length;

        while(grid_state[index] != sep) {
            while(!is_digit(grid_state[index])) { index++; }
            auto num = read_number<size_t>(grid_state, index);
            n.emplace_back(num);
            N_ += num;
            Cavity_count++;
            std::vector<E_LEVEL> tmp;
            m.emplace_back(0);
        }

        format_index++;
        middle_length = format_index;

        while(format[++format_index] != 'M') {}

        middle_length = format_index - middle_length;
        index += middle_length;

        char end = format[format_index + 1];
        size_t Cavity_index = 0;

        while(Cavity_index != Cavity_count) {
            if(!is_digit(grid_state[index])) {
                if (m[Cavity_index].size() != 0) {
                    cavities_with_atoms_.insert(Cavity_index);
                }

                grid_states_.emplace_back(n[Cavity_index], m[Cavity_index]);
                Cavity_index++;
            } else {
                if (grid_state[index] == '0') {
                    N_++;
                }

                m[Cavity_index].emplace_back(grid_state[index] - '0');
            }

            index++;
        }
    }
}

size_t State::get_index(const std::set<State>& basis) const {
    size_t index = 0;
    for (const auto& state: basis) {
        if (state == *this) return index;
        index++;
    }

    return -1;
}

size_t State::get_max_size() const {
    size_t res = 0;

    for (long i = grid_states_.size() - 1; i >= 0 ; i--) {
        auto tmp = grid_states_[i].variants_of_state_count(N_);
        res = res * tmp + tmp - 1;
    }

    return res;
}

State::State(size_t x_size, size_t y_size, size_t z_size) {
    grid_states_.reserve(x_size * y_size * z_size);
}

State::State(const Cavity_State& state) {
    grid_states_.emplace_back(state);
    x_size_ = y_size_ = z_size_ = 1;
    N_ = state.n() + state.up_count();
    if (state.m() != 0) {
        cavities_with_atoms_.insert(0);
    }

    gamma_ = Matrix<COMPLEX>(1, 1, 0);
}

size_t State::get_index() const {
    size_t index = 0;

    for (long i = grid_states_.size() - 1; i >= 0 ; i--) {
        index *= grid_states_[i].variants_of_state_count(N_);  
        index += grid_states_[i].get_index(); 
    }

    return index;
}

std::string State::to_string() const {
    std::string res = "|";

    for (size_t i = 0; i < grid_states_.size(); i++) {
        res += std::to_string(grid_states_[i].n());

        if (i != grid_states_.size() - 1) res += ",";
    }

    res += ";";

    for (size_t i = 0; i < grid_states_.size(); i++) {
        auto state = grid_states_[i].get_atoms_state();

        if (state.size() == 0) {
            res += "_";
        }

        for (const auto& st: state) {
            res += std::to_string(st);
        }

        if (i != grid_states_.size() - 1) res += ",";
    }

    res += ">";

    return res;
}

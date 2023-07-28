#include "state.hpp"
#include "functions.hpp"

// REWRITE TO REGEXP
State::State(const std::string& grid_state, const std::string& format) : gamma_(0, 0, 0) {
    size_t format_index = 0;
    size_t left_length = 0, middle_length = 0, right_length = 0;
    while (format[format_index] != 'N') {
            left_length++;
            format_index++;
    }

    max_N_ = 0;

    std::vector<CavityId> n;
    std::vector<std::vector<E_LEVEL>> m;
    if (format[format_index] == 'N') {
        char sep = format[format_index + 1];
        char end = format[format.size() - 1];
        size_t Cavity_count = 0;
        size_t& index = left_length;

        while(grid_state[index] != sep and grid_state[index] != end) {
            while(!is_digit(grid_state[index])) { index++; }
            auto num = read_number<size_t>(grid_state, index);
            n.emplace_back(num);
            max_N_ += num;
            Cavity_count++;
            std::vector<E_LEVEL> tmp;
            m.emplace_back(0);
        }

        format_index++;
        middle_length = format_index;

        bool is_continue = true;
        while(format[++format_index] != 'M') {
            if (format.size() == format_index) {
                is_continue = false;
                for (const auto& num: n) {
                    grid_states_.emplace_back(num);
                }
            }
        }

        if (is_continue) {
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
                    if (grid_state[index] == '1') {
                        max_N_++;
                    }

                    m[Cavity_index].emplace_back(grid_state[index] - '0');
                }

                index++;
            }
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
        std::cout << i << " " << max_N_ << " -> " << grid_states_[i].to_string() << " " << grid_states_[i].variants_of_state_count(max_N_) << std::endl;
        auto tmp = grid_states_[i].variants_of_state_count(max_N_);
        res = res * tmp + tmp - 1;
    }

    res += 1;
    return res;
}

size_t State::get_energy() const {
    size_t res = 0;

    for (const auto& state: grid_states_) {
        res += state.get_energy();
    }

    return res;
}

void State::set_state(CavityId id, const Cavity_State& state) {
    max_N_ -= grid_states_[id].get_energy();
    grid_states_[id] = state;

    cavities_with_atoms_.erase(id);

    max_N_ += state.get_energy();

    if (state.m() != 0) cavities_with_atoms_.insert(id);
}

State State::add_state(const Cavity_State& state) const {
    State res = (*this);
    res.max_N_ += state.get_energy();

    if (state.m() != 0) res.cavities_with_atoms_.insert(res.cavities_count());

    res.grid_states_.emplace_back(state);

    return res;
}

State::State(size_t x_size, size_t y_size, size_t z_size) {
    grid_states_.reserve(x_size * y_size * z_size);
}

State::State(const Cavity_State& state) {
    grid_states_.emplace_back(state);
    x_size_ = y_size_ = z_size_ = 1;
    max_N_ = state.n() + state.up_count();
    if (state.m() != 0) {
        cavities_with_atoms_.insert(0);
    }

    gamma_ = Matrix<COMPLEX>(1, 1, 0);
}

size_t State::get_index() const {
    size_t index = 0;

    for (long i = grid_states_.size() - 1; i >= 0 ; i--) {
        index *= grid_states_[i].variants_of_state_count(max_N_);  
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

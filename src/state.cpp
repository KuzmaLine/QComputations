#include "state.hpp"
#include "functions.hpp"


namespace QComputations {

namespace {
    using CavityId = size_t;

    std::vector<std::vector<CavityId>> update_neighbours(size_t x_size, size_t y_size, size_t z_size) {
        std::vector<std::vector<CavityId>> res(x_size * y_size * z_size);

        for (size_t z = 0; z < z_size; z++) {
            for (size_t y = 0; y < y_size; y++) {
                for (size_t x = 0; x < x_size; x++) {
                    auto index = z * y_size * x_size + y * x_size + x;
                    if ((x + 1) != x_size) {
                        res[index].emplace_back(index + 1);
                        res[index + 1].emplace_back(index);
                    }

                    if ((y + 1) != y_size) {
                        res[index].emplace_back(index + x_size);
                        res[index + x_size].emplace_back(index);
                    }
                    
                    if ((z + 1) != z_size) {
                        res[index].emplace_back(index + x_size * y_size);
                        res[index + x_size * y_size].emplace_back(index);
                    }
                }
            }
        }

        return res;
    }
}

size_t State::hash() const {
    std::hash<Cavity_State> state_hash;
    auto res = state_hash(grid_states_[0]);

    for (size_t i = 1; i < grid_states_.size(); i++) {
        res ^= state_hash(grid_states_[i]);
    }

    return res;
}


// REWRITE TO REGEXP
State::State(const std::string& grid_state, const std::string& format) : waveguides_(C_STYLE, 0, 0, std::make_pair(QConfig::instance().waveguides_amplitude(), QConfig::instance().waveguides_length())) {
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
            gamma_leak_cavities_.emplace_back(0);
            gamma_gain_cavities_.emplace_back(0);
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

void State::set_waveguide(double amplitude, double length) {
    waveguides_ = Matrix<std::pair<double, double>>(C_STYLE, grid_states_.size(), grid_states_.size(),
        std::make_pair(0, 0));
    for (size_t from_id = 0; from_id < grid_states_.size(); from_id++) {
        auto neighbours = neighbours_[from_id];

        for (const auto to_id: neighbours) {
            waveguides_[from_id][to_id] = std::make_pair(amplitude, length);
        }
    }
}

void State::reshape(size_t x_size, size_t y_size, size_t z_size) {
    assert(x_size * y_size * z_size == grid_states_.size());

    x_size_ = x_size;
    y_size_ = y_size;
    z_size_ = z_size;

    waveguides_ = Matrix<std::pair<double, double>>(C_STYLE, grid_states_.size(), grid_states_.size(),
            std::make_pair(QConfig::instance().waveguides_amplitude(), QConfig::instance().waveguides_length()));

    neighbours_ = update_neighbours(x_size_, y_size_, z_size_);
}

State::State(const std::vector<size_t>& grid_config, E_LEVEL e_levels_count): waveguides_(C_STYLE, grid_config.size(),
                                                                              grid_config.size(),
                                                                              std::make_pair(QConfig::instance().waveguides_amplitude(), QConfig::instance().waveguides_length())),
                                                                              e_levels_count_(e_levels_count) {
    //std::cout << grid_config << std::endl;
    x_size_ = grid_config.size();
    y_size_ = 1;
    z_size_ = 1;

    neighbours_ = update_neighbours(x_size_, y_size_, z_size_);
    for (size_t i = 0; i < grid_config.size(); i++) {
        std::vector<int> m(grid_config[i], 0);
        grid_states_.emplace_back(0, m, e_levels_count);
        gamma_leak_cavities_.emplace_back(0);
        gamma_gain_cavities_.emplace_back(0);
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

size_t State::get_grid_energy() const {
    size_t res = 0;

    for (const auto& state: grid_states_) {
        res += state.get_energy();
    }

    return res;
}

size_t State::get_energy(CavityId cavity_id) const {
    return grid_states_[cavity_id].get_energy();
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
    max_N_ = state.get_energy();
    if (state.m() != 0) {
        cavities_with_atoms_.insert(0);
    }

    waveguides_ = Matrix<std::pair<double, double>>(C_STYLE, 1, 1, std::make_pair(QConfig::instance().waveguides_amplitude(), QConfig::instance().waveguides_length()));
}

// Рудимент
size_t State::get_index() const {
    size_t index = 0;

    for (long i = grid_states_.size() - 1; i >= 0 ; i--) {
        index *= grid_states_[i].variants_of_state_count(max_N_);  
        index += grid_states_[i].get_index(); 
    }

    return index;
}

BigUInt State::to_uint() const {
    BigUInt res(0);

    for (size_t i = 0; i < grid_states_.size(); i++) {
        res <<= 32;
        res += BigUInt(this->n(i));
    }

    for (size_t i = 0; i < grid_states_.size(); i++) {
        auto state = grid_states_[i].get_atoms_state();

        for (const auto& st: state) {
            res <<= 1;
            res += BigUInt(st);
        }
    }

    return res;
}

void State::from_uint(const BigUInt& state_num) {
    size_t index = 0;
    for (long long i = grid_states_.size() - 1; i >= 0; i--) {
        long long m = grid_states_[i].m();
        //auto state = grid_states_[i].get_atoms_state();

        for (long long j = m - 1; j >= 0; j--) {
            this->set_qubit(i, j, state_num.get_bit(index++));
        }
    }

    auto n_num = state_num >> index;
    for (size_t i = 0; i < grid_states_.size(); i++) {
        this->set_n(n_num.get_num(grid_states_.size() - i - 1), i);
    }
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

} // namespace QComputations
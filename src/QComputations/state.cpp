#include "state.hpp"
#include "functions.hpp"
#include <regex>

namespace QComputations {

namespace {
    using CavityId = size_t;
    constexpr ValType INIT_VAL = 0;
}

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


// --------------------------- Basis_Sate -------------------------------------

Basis_State::Basis_State(size_t qudits_count, ValType max_val, size_t groups_count): qudits_(qudits_count, INIT_VAL), max_vals_(qudits_count, max_val) {
    assert(qudits_count % groups_count == 0);
    groups_ = std::vector<size_t>(groups_count);
    for (size_t i = 0; i < groups_count; ++i) {
        groups_[i] = (i + 1) * (qudits_count / groups_count) - 1;
    }
}

std::string Basis_State::to_string() const {
    std::string state_delimeter = QConfig::instance().state_delimeter();

    std::string res;

    size_t next_end = 0;
    size_t group_id = 0;
    size_t next_start = 0;
    for (size_t i = 0; i < qudits_.size(); i++) {
        if (i == next_start) {
            next_end = groups_[group_id++];
            next_start = next_end + 1;
            res += "|";
        }

        res += std::to_string(qudits_[i]);

        if (i == next_end) {
            res += ">";
        } else {
            res += state_delimeter;
        }
    }

    return res;
}

Basis_State Basis_State::get_group(size_t group_id) const {
    std::vector<ValType> qudits(this->get_group_size(group_id));

    for (size_t i = 0; i < this->get_group_size(group_id); i++) {
        qudits[i] = this->get_qudit(i, group_id);
    }

    return Basis_State(qudits, max_vals_);
}

size_t Basis_State::get_index(const std::set<Basis_State>& basis) const {
    size_t index = 0;
    for (const auto& state: basis) {
        if (state == *this) return index;
        index++;
    }

    return -1;
}

// --------------------------- CHE_State -------------------------------------

void CHE_State::set_waveguide(double amplitude, double length) {
    waveguides_ = Matrix<std::pair<double, double>>(C_STYLE, groups_.size(), groups_.size(),
        std::make_pair(0, 0));
    for (size_t from_id = 0; from_id < groups_.size(); from_id++) {
        auto neighbours = neighbours_[from_id];

        for (const auto to_id: neighbours) {
            waveguides_[from_id][to_id] = std::make_pair(amplitude, length);
        }
    }
}

void CHE_State::set_waveguide(size_t from_cavity_id, size_t to_cavity_id, double amplitude, double length) {
    waveguides_[from_cavity_id][to_cavity_id] = std::make_pair(amplitude, length);

    if (amplitude >= QConfig::instance().eps()) {
        if (!is_in_vector(neighbours_[from_cavity_id], to_cavity_id)) {
            neighbours_[from_cavity_id].emplace_back(to_cavity_id);
            neighbours_[to_cavity_id].emplace_back(from_cavity_id);
        }
    }
}

void CHE_State::reshape(size_t x_size, size_t y_size, size_t z_size) {
    assert(x_size * y_size * z_size == groups_.size());

    x_size_ = x_size;
    y_size_ = y_size;
    z_size_ = z_size;

    waveguides_ = Matrix<std::pair<double, double>>(C_STYLE, groups_.size(), groups_.size(),
            std::make_pair(QConfig::instance().waveguides_amplitude(), QConfig::instance().waveguides_length()));

    neighbours_ = update_neighbours(x_size_, y_size_, z_size_);
}

std::set<CavityId> CHE_State::get_cavities_with_leak() const {
    std::set<CavityId> cavity_set;

    for (size_t i = 0; i < this->cavities_count(); i++) {
        cavity_set.insert(i);
    }

    std::function<bool(CavityId)> func = {[&](CavityId cavity_id){
        return !is_zero(gamma_leak_cavities_[cavity_id]);
    }};

    return set_query<CavityId>(cavity_set, func);
}

std::set<CavityId> CHE_State::get_cavities_with_gain() const {
    std::set<CavityId> cavity_set;

    for (size_t i = 0; i < this->cavities_count(); i++) {
        cavity_set.insert(i);
    }

    std::function<bool(CavityId)> func = {[&](CavityId cavity_id){
        return !is_zero(gamma_gain_cavities_[cavity_id]);
    }};

    return set_query<CavityId>(cavity_set, func);
}

namespace {
    size_t vector_sum(const std::vector<size_t>& v) {
        size_t sum = 0;
        for (size_t i = 0; i < v.size(); i++) {
            sum += v[i];
        }

        return sum;
    }

    std::vector<size_t> che_groups_from_grid_config(const std::vector<size_t>& grid_config) {
        std::vector<size_t> groups;

        size_t qudits_prev_sum = 0;
        for (const auto& qubits_count: grid_config) {
            auto n = qubits_count + 1;
            groups.emplace_back(qudits_prev_sum + n - 1);
            qudits_prev_sum += n;
        }

        return groups;
    }
}

CHE_State::CHE_State(const std::vector<size_t>& grid_config): Basis_State(vector_sum(grid_config) + grid_config.size(), 1, che_groups_from_grid_config(grid_config)),
                                                              gamma_leak_cavities_(grid_config.size(), 0),
                                                              gamma_gain_cavities_(grid_config.size(), 0),
                                                              waveguides_(C_STYLE, grid_config.size(),
                                                              grid_config.size(),
                                                              std::make_pair(QConfig::instance().waveguides_amplitude(), QConfig::instance().waveguides_length())) {
    x_size_ = grid_config.size();
    y_size_ = 1;
    z_size_ = 1;

    neighbours_ = update_neighbours(x_size_, y_size_, z_size_);

    for (size_t i = 0; i < grid_config.size(); i++) {
        this->set_max_val(QConfig::instance().max_photons(), 0, i);
    }
}

size_t CHE_State::get_index(const std::set<CHE_State>& basis) const {
    size_t index = 0;
    for (const auto& state: basis) {
        if (state == *this) return index;
        index++;
    }

    return -1;
}

} // namespace QComputations
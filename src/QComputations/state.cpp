#include "state.hpp"
#include "functions.hpp"
#include <regex>
#include <mkl_cblas.h>

namespace QComputations {

void vector_normalize(std::vector<COMPLEX>& v) {
    double res = cblas_dznrm2(v.size(), v.data(), 1);

    for (size_t i = 0; i < v.size(); i++) {
        v[i] /= res;
    }
}

namespace {
    using CavityId = size_t;
    constexpr ValType INIT_VAL = 0;

    std::string state_delimeter = ";";
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

Basis_State::Basis_State(size_t qudits_count, ValType max_val, const std::vector<size_t>& groups): qudits_(qudits_count, 0),
                                                                                                    max_vals_(qudits_count, max_val) {
    size_t qudits_prev_sum = 0;
    for (const auto& qubits_count: groups) {
        auto n = qubits_count;
        groups_.emplace_back(qudits_prev_sum + n - 1);
        qudits_prev_sum += n;
    }
}

Basis_State::Basis_State(const std::vector<ValType>& qudits, ValType max_val, const std::vector<size_t>& groups): qudits_(qudits),
                                                                                                                    max_vals_(qudits.size(), max_val) {
    size_t qudits_prev_sum = 0;
    for (const auto& qubits_count: groups) {
        auto n = qubits_count;
        groups_.emplace_back(qudits_prev_sum + n - 1);
        qudits_prev_sum += n;
    }
}

Basis_State::Basis_State(const std::string& qudits_str, ValType max_val) {
    char state_delim = state_delimeter[0];

    ValType res = 0;
    for (size_t i = 0; i < qudits_str.size(); i++) {
        if (qudits_str[i] != '|' and qudits_str[i] != '>' and qudits_str[i] != state_delim) {
            res = res * 10 + (qudits_str[i] - '0');
        } else if (qudits_str[i] == state_delim or qudits_str[i] == '>') {
            assert(res <= max_val);
            max_vals_.emplace_back(max_val);
            qudits_.emplace_back(res);
            res = 0;

            if (qudits_str[i] == '>') {
                groups_.emplace_back(qudits_.size() - 1);
            }
        }
    }
}

Basis_State::Basis_State(const std::string& qudits_str, const std::vector<ValType>& max_vals): max_vals_(max_vals) {
    char state_delim = state_delimeter[0];

    ValType res = 0;
    for (size_t i = 0; i < qudits_str.size(); i++) {
        if (qudits_str[i] != '|' and qudits_str[i] != '>' and qudits_str[i] != state_delim) {
            res = res * 10 + (qudits_str[i] - '0');
        } else if (qudits_str[i] == state_delim or qudits_str[i] == '>') {
            assert(res <= max_vals_[qudits_.size() - 1]);
            qudits_.emplace_back(res);
            res = 0;

            if (qudits_str[i] == '>') {
                groups_.emplace_back(qudits_.size() - 1);
            }
        }
    }
}

//REWRITE TO REGEX
void Basis_State::set_state(const std::string& str_state) {
    size_t index_start = 0;
    for (size_t i = 0; i < this->groups_count(); i++) {
        this->set_group(i, str_state, index_start);

        index_start += this->group_size(i) * 2  + 1;
    }
}

void Basis_State::set_group(size_t group_id, const std::string& group_state, size_t index_start) {
    ValType res_num = 0;
    size_t qudit_index = 0;
    for (size_t i = 0; qudit_index < this->group_size(group_id); i++) {
        if (group_state[index_start + i] != '|' and group_state[index_start + i] != '>' and group_state[index_start + i] != ';') {
            res_num = res_num * 10 + (group_state[index_start + i] - '0');
        } else if (group_state[index_start + i] != '|') {
            this->set_qudit(res_num, qudit_index, group_id);
            res_num = 0;
            qudit_index++;
        }
    }
}

std::string Basis_State::group_to_string(size_t group_id) const {
    std::string state_delim = state_delimeter;

    std::string res;

    for (size_t i = this->get_group_start(group_id); i <= this->get_group_end(group_id); i++) {
        res += state_delimeter + std::to_string(qudits_[i]);
    }

    res[0] = '|';
    res += ">";

    return res;
}

std::string Basis_State::to_string() const {
    std::string res;

    for (size_t i = 0; i < this->get_groups_count(); i++) {
        res += this->group_to_string(i);
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

size_t Basis_State::get_index(const BasisType<Basis_State>& basis) const {
    size_t index = 0;
    for (auto state: basis) {
        if (*state == *this) return index;
        index++;
    }

    return -1;
}

// --------------------------- TCH_State -------------------------------------

void TCH_State::set_waveguide(double amplitude, double length) {
    waveguides_ = Matrix<std::pair<double, double>>(C_STYLE, groups_.size(), groups_.size(),
        std::make_pair(0, 0));
    for (size_t from_id = 0; from_id < groups_.size(); from_id++) {
        auto neighbours = neighbours_[from_id];

        for (const auto to_id: neighbours) {
            waveguides_[from_id][to_id] = std::make_pair(amplitude, length);
        }
    }
}

void TCH_State::set_waveguide(size_t from_cavity_id, size_t to_cavity_id, double amplitude, double length) {
    waveguides_[from_cavity_id][to_cavity_id] = std::make_pair(amplitude, length);

    if (amplitude >= QConfig::instance().eps()) {
        if (!is_in_vector(neighbours_[from_cavity_id], to_cavity_id)) {
            neighbours_[from_cavity_id].emplace_back(to_cavity_id);
            neighbours_[to_cavity_id].emplace_back(from_cavity_id);
        }
    }
}

void TCH_State::reshape(size_t x_size, size_t y_size, size_t z_size) {
    assert(x_size * y_size * z_size == groups_.size());

    x_size_ = x_size;
    y_size_ = y_size;
    z_size_ = z_size;

    waveguides_ = Matrix<std::pair<double, double>>(C_STYLE, groups_.size(), groups_.size(),
            std::make_pair(QConfig::instance().waveguides_amplitude(), QConfig::instance().waveguides_length()));

    neighbours_ = update_neighbours(x_size_, y_size_, z_size_);
}

std::set<CavityId> TCH_State::get_cavities_with_leak() const {
    std::set<CavityId> cavity_set;

    for (size_t i = 0; i < this->cavities_count(); i++) {
        cavity_set.insert(i);
    }

    std::function<bool(CavityId)> func = {[&](CavityId cavity_id){
        return !is_zero(gamma_leak_cavities_[cavity_id]);
    }};

    return set_bool_check<CavityId>(cavity_set, func);
}

std::set<CavityId> TCH_State::get_cavities_with_gain() const {
    std::set<CavityId> cavity_set;

    for (size_t i = 0; i < this->cavities_count(); i++) {
        cavity_set.insert(i);
    }

    std::function<bool(CavityId)> func = {[&](CavityId cavity_id){
        return !is_zero(gamma_gain_cavities_[cavity_id]);
    }};

    return set_bool_check<CavityId>(cavity_set, func);
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

    std::vector<size_t> add_one(const std::vector<size_t>& grid_config) {
        std::vector<size_t> res;

        for (const auto& qubits_count: grid_config) {
            res.emplace_back(qubits_count + 1);
        }

        return res;
    }
}

TCH_State::TCH_State(const std::vector<size_t>& grid_config): Basis_State(vector_sum(grid_config) + grid_config.size(), 1, add_one(grid_config)),
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

size_t TCH_State::get_index(const std::set<TCH_State>& basis) const {
    size_t index = 0;
    for (const auto& state: basis) {
        if (state == *this) return index;
        index++;
    }

    return -1;
}

} // namespace QComputations

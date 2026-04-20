#include "state.hpp"
#include "functions.hpp"
#include <regex>

#ifdef ENABLE_ONEAPI
#include <mkl_cblas.h>
#endif

namespace QComputations {

#ifdef ENABLE_ONEAPI
void vector_normalize(std::vector<COMPLEX>& v) {
    double res = cblas_dznrm2(v.size(), v.data(), 1);

    for (size_t i = 0; i < v.size(); i++) {
        v[i] /= res;
    }
}
#endif

namespace {
    using CavityId = size_t;
    constexpr ValType INIT_VAL = 0;

    std::string state_delimeter = ";";

    /*
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
    */
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

Basis_State::Basis_State(size_t qudits_count, ValType max_val, size_t groups_count): qudits_(qudits_count, INIT_VAL), max_vals_(std::make_shared<std::vector<ValType>>(std::vector<ValType>(qudits_count, max_val))) {
    assert(qudits_count % groups_count == 0);
    groups_ = std::make_shared<std::vector<size_t>>(std::vector<size_t>(groups_count));
    for (size_t i = 0; i < groups_count; ++i) {
        (*groups_)[i] = (i + 1) * (qudits_count / groups_count) - 1;
    }
}

Basis_State::Basis_State(size_t qudits_count, const std::vector<int>& max_val, size_t groups_count): qudits_(qudits_count, INIT_VAL), max_vals_(std::make_shared<std::vector<ValType>>(max_val)) {
    assert(qudits_count % groups_count == 0);
    groups_ = std::make_shared<std::vector<size_t>>(std::vector<size_t>(groups_count));
    for (size_t i = 0; i < groups_count; ++i) {
        (*groups_)[i] = (i + 1) * (qudits_count / groups_count) - 1;
    }
}

Basis_State::Basis_State(size_t qudits_count, ValType max_val, const std::vector<size_t>& groups): qudits_(qudits_count, 0),
                                                                                                    max_vals_(std::make_shared<std::vector<ValType>>(std::vector<ValType>(qudits_count, max_val))) {
    groups_ = std::make_shared<std::vector<size_t>>(std::vector<size_t>());
    size_t qudits_prev_sum = 0;
    for (const auto& qubits_count: groups) {
        auto n = qubits_count;
        groups_->emplace_back(qudits_prev_sum + n - 1);
        qudits_prev_sum += n;
    }
}

Basis_State::Basis_State(const std::vector<ValType>& qudits, ValType max_val, const std::vector<size_t>& groups): qudits_(qudits),
                                                                                                                    max_vals_(std::make_shared<std::vector<ValType>>(std::vector<ValType>(qudits.size(), max_val))) {
    groups_ = std::make_shared<std::vector<size_t>>(std::vector<size_t>());                                                                                                                    
    size_t qudits_prev_sum = 0;
    for (const auto& qubits_count: groups) {
        auto n = qubits_count;
        groups_->emplace_back(qudits_prev_sum + n - 1);
        qudits_prev_sum += n;
    }
}

Basis_State::Basis_State(const std::string& qudits_str, ValType max_val) {
    char state_delim = state_delimeter[0];
    groups_ = std::make_shared<std::vector<size_t>>(std::vector<size_t>());
    max_vals_ = std::make_shared<std::vector<ValType>>(std::vector<ValType>());        
    ValType res = 0;
    for (size_t i = 0; i < qudits_str.size(); i++) {
        if (qudits_str[i] != '|' and qudits_str[i] != '>' and qudits_str[i] != state_delim) {
            res = res * 10 + (qudits_str[i] - '0');
        } else if (qudits_str[i] == state_delim or qudits_str[i] == '>') {
            assert(res <= max_val);
            max_vals_->emplace_back(max_val);
            qudits_.emplace_back(res);
            res = 0;

            if (qudits_str[i] == '>') {
                groups_->emplace_back(qudits_.size() - 1);
            }
        }
    }
}

Basis_State::Basis_State(const std::string& qudits_str, const std::vector<ValType>& max_vals): max_vals_(std::make_shared<std::vector<ValType>>(max_vals)) {
    char state_delim = state_delimeter[0];

    ValType res = 0;
    for (size_t i = 0; i < qudits_str.size(); i++) {
        if (qudits_str[i] != '|' and qudits_str[i] != '>' and qudits_str[i] != state_delim) {
            res = res * 10 + (qudits_str[i] - '0');
        } else if (qudits_str[i] == state_delim or qudits_str[i] == '>') {
            assert(res <= (*max_vals_)[qudits_.size() - 1]);
            qudits_.emplace_back(res);
            res = 0;

            if (qudits_str[i] == '>') {
                groups_->emplace_back(qudits_.size() - 1);
            }
        }
    }
}

void Basis_State::set_groups(const std::vector<size_t>& groups) {
    groups_->resize(0);
    size_t qudits_prev_sum = 0;
    for (const auto& qubits_count: groups) {
        auto n = qubits_count;
        groups_->emplace_back(qudits_prev_sum + n - 1);
        qudits_prev_sum += n;
    }

    hash_cached_ = false;
}

//REWRITE TO REGEX
void Basis_State::set_state(const std::string& str_state) {
    size_t index_start = 0;
    for (size_t i = 0; i < this->groups_count(); i++) {
        this->set_group(i, str_state, index_start);

        index_start += this->group_size(i) * 2  + 1;
    }

    hash_cached_ = false;
}

void Basis_State::set_group(size_t group_id, const std::string& group_state, size_t index_start) {
    //std::string state_delimeter = state_delim;

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
    hash_cached_ = false;
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

Basis_State Basis_State::reduce(const std::vector<size_t>& qudits) const {
    Basis_State res(*this);

    res.qudits_.resize(qudits.size());
    res.max_vals_->resize(qudits.size());
    int prev_group_id = -1;

    std::vector<size_t> new_groups;

    for (int i = 0; i < qudits.size(); i++) { 
        res.qudits_[i] = this->get_qudit(qudits[i]);
        (*(res.max_vals_))[i] = this->get_max_val(qudits[i]);
        if (this->get_group_id(qudits[i]) != prev_group_id) {
            if (prev_group_id != -1) {
                new_groups.emplace_back(i - 1);
            }
            prev_group_id = this->get_group_id(qudits[i]);
        }
    }

    new_groups.emplace_back(qudits.size() - 1);
    res.groups_ = std::make_shared<std::vector<size_t>>(new_groups);

    return res;
}

Basis_State Basis_State::get_group(size_t group_id) const {
    std::vector<ValType> qudits(this->get_group_size(group_id));
    std::vector<ValType> max_vals(this->get_group_size(group_id));

    for (size_t i = 0; i < this->get_group_size(group_id); i++) {
        qudits[i] = this->get_qudit(i, group_id);
        max_vals[i] = this->get_max_val(i, group_id);
    }

    return Basis_State(qudits, max_vals);
}

// size_t Basis_State::get_index(const BasisType<Basis_State>& basis) const {
//     size_t index = 0;
//     for (auto state: basis) {
//         if (*state == *this) return index;
//         index++;
//     }

//     return -1;
// }

size_t Basis_State::get_qudit_index(size_t qudit_index, size_t group_index) const {
    size_t res = 0;
    for (int i = 0; i < group_index; i++) {
        res += this->get_group_size(i);
    }

    return res + qudit_index;
}

// --------------------------- TCH_State -------------------------------------

/*
size_t TCH_State::hash() const {
    std::hash<Cavity_State> state_hash;
    auto res = state_hash(grid_states_[0]);

    for (size_t i = 1; i < grid_states_.size(); i++) {
        res ^= state_hash(grid_states_[i]);
    }

    return res;
}
*/

// REWRITE TO REGEXP
/*
TCH_State::TCH_State(const std::string& grid_state, const std::string& format,
             const std::string& del, bool is_freq_display) : waveguides_(C_STYLE, 0, 0,
                                                                        std::make_pair(QConfig::instance().waveguides_amplitude(),
                                                                        QConfig::instance().waveguides_length())) {
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
*/

void TCH_State::set_waveguide(double amplitude, double length) {
    waveguides_ = Matrix<std::pair<double, double>>(C_STYLE, groups_->size(), groups_->size(),
        std::pair<double, double>(0, 0));
    for (size_t from_id = 0; from_id < groups_->size(); from_id++) {
        auto neighbours = neighbours_[from_id];

        for (const auto to_id: neighbours) {
            waveguides_[from_id][to_id] = std::pair<double, double>(amplitude, length);
        }
    }
}

void TCH_State::set_waveguide(size_t from_cavity_id, size_t to_cavity_id, double amplitude, double length) {
    waveguides_[from_cavity_id][to_cavity_id] = std::pair<double, double>(amplitude, length);
    waveguides_[to_cavity_id][from_cavity_id] = std::pair<double, double>(amplitude, length);

    if (amplitude >= QConfig::instance().eps()) {
        if (!is_in_vector(neighbours_[from_cavity_id], to_cavity_id)) {
            neighbours_[from_cavity_id].emplace_back(to_cavity_id);
            neighbours_[to_cavity_id].emplace_back(from_cavity_id);
        }
    }
}

/*
void TCH_State::set_waveguide(const Matrix<std::pair<double, double>>& A) {
    waveguides_ = A;
    for (size_t from_id = 0; from_id < groups_.size(); from_id++) {
        auto neighbours = neighbours_[from_id];

        for (const auto to_id: neighbours) {
            waveguides_[from_id][to_id] = std::make_pair(amplitude, length);
        }
    }
}
*/

void TCH_State::reshape(size_t x_size, size_t y_size, size_t z_size) {
    assert(x_size * y_size * z_size == groups_->size());

    x_size_ = x_size;
    y_size_ = y_size;
    z_size_ = z_size;

    waveguides_ = Matrix<std::pair<double, double>>(C_STYLE, groups_->size(), groups_->size(),
            std::pair<double, double>(QConfig::instance().waveguides_amplitude(), QConfig::instance().waveguides_length()));

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

    std::vector<size_t> add_photons(const std::vector<size_t>& grid_config, size_t levels_count) {
        std::vector<size_t> res;
        auto ph_types_count = photons_types_count(levels_count);

        for (const auto& qubits_count: grid_config) {
            res.emplace_back(qubits_count + ph_types_count);
        }

        return res;
    }
}

TCH_State::TCH_State(const std::vector<size_t>& grid_config, size_t levels_count): Basis_State(vector_sum(grid_config) + photons_types_count(levels_count) * grid_config.size(), levels_count - 1, add_photons(grid_config, levels_count)),
                                                              gamma_leak_cavities_(grid_config.size(), 0),
                                                              gamma_gain_cavities_(grid_config.size(), 0),
                                                              waveguides_(C_STYLE, grid_config.size(),
                                                              grid_config.size(),
                                                              std::pair<double, double>(QConfig::instance().waveguides_amplitude(), QConfig::instance().waveguides_length())),
                                                              freq_levels_(levels_count),
                                                              g_(C_STYLE, levels_count, levels_count, COMPLEX(0)) {
    is_move_ = std::vector<std::vector<std::vector<bool>>>(qudits_.size(), std::vector<std::vector<bool>>(levels_count, std::vector<bool>(levels_count, false)));
    freq_levels_[0] = 0;
    freq_levels_[1] = QConfig::instance().w();
    g_[0][1] = QConfig::instance().g();
    g_[1][0] = std::conj(QConfig::instance().g());

    for (size_t i = 0; i < qudits_.size(); i++) {
        is_move_[i][0][1] = true;
        is_move_[i][1][0] = true;
    }
    //std::cout << grid_config << std::endl;
    x_size_ = grid_config.size();
    y_size_ = 1;
    z_size_ = 1;

    neighbours_ = update_neighbours(x_size_, y_size_, z_size_);

    int index = 0;
    for (size_t i = 0; i < grid_config.size(); i++) {
        for (size_t w_from = 0; w_from < levels_count; w_from++ ) {
            for (size_t w_to = w_from + 1; w_to < levels_count; w_to++) {
                this->set_max_val(QConfig::instance().max_photons(), index++, i);
            }
        }

        index = 0;
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

/*
TCH_State TCH_State::get_state_in_cavity(CavityId cavity_id) const {
    auto a = this->get_group_start(cavity_id);
    auto b = this->get_group_end(cavity_id);

    std::vector<ValType> qudits(b - a + 1);
    std::vector<ValType> max_vals(b - a + 1);

    std::copy(qudits_.begin() + a, qudits_.begin() + b + 1, qudits.begin());    
    std::copy(max_vals_.begin() + a, max_vals_.begin() + b + 1, max_vals.begin());

    Basis_State b_state(qudits, max_vals);

    TCH_State res(b_state);

    res.set_leak_for_cavity(0, this->get_leak_gamma(0));
    res.set_gain_for_cavity(0, this->get_gain_gamma(0));

    return res;
}
*/

/*
size_t TCH_State::get_max_size() const {
    size_t res = 0;

    for (long i = grid_states_.size() - 1; i >= 0 ; i--) {
        std::cout << i << " " << max_N_ << " -> " << grid_states_[i].to_string() << " " << grid_states_[i].variants_of_state_count(max_N_) << std::endl;
        auto tmp = grid_states_[i].variants_of_state_count(max_N_);
        res = res * tmp + tmp - 1;
    }

    res += 1;
    return res;
}

size_t TCH_State::get_grid_energy() const {
    size_t res = 0;

    for (const auto& state: grid_states_) {
        res += state.get_energy();
    }

    return res;
}

size_t TCH_State::get_energy(CavityId cavity_id) const {
    return grid_states_[cavity_id].get_energy();
}

void TCH_State::set_state(CavityId id, const Cavity_State& state) {
    max_N_ -= grid_states_[id].get_energy();
    grid_states_[id] = state;

    cavities_with_atoms_.erase(id);

    max_N_ += state.get_energy();

    if (state.m() != 0) cavities_with_atoms_.insert(id);
}

TCH_State TCH_State::add_state(const Cavity_State& state) const {
    TCH_State res = (*this);
    res.max_N_ += state.get_energy();

    if (state.m() != 0) res.cavities_with_atoms_.insert(res.cavities_count());

    res.grid_states_.emplace_back(state);

    return res;
}

TCH_State::TCH_State(size_t x_size, size_t y_size, size_t z_size) {
    grid_states_.reserve(x_size * y_size * z_size);
}

TCH_State::TCH_State(const Cavity_State& state) {
    grid_states_.emplace_back(state);
    x_size_ = y_size_ = z_size_ = 1;
    max_N_ = state.get_energy();
    if (state.m() != 0) {
        cavities_with_atoms_.insert(0);
    }

    waveguides_ = Matrix<std::pair<double, double>>(C_STYLE, 1, 1, std::make_pair(QConfig::instance().waveguides_amplitude(), QConfig::instance().waveguides_length()));
}

// Рудимент
size_t TCH_State::get_index() const {
    size_t index = 0;

    for (long i = grid_states_.size() - 1; i >= 0 ; i--) {
        index *= grid_states_[i].variants_of_state_count(max_N_);  
        index += grid_states_[i].get_index(); 
    }

    return index;
}

BigUInt TCH_State::to_uint() const {
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

void TCH_State::from_uint(const BigUInt& state_num) {
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
*/

/*
std::string TCH_State::to_string() const {
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
*/

} // namespace QComputations

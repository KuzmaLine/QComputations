#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <complex>
#include <cassert>
#include <set>
#include "matrix.hpp"
//#include "big_uint.hpp"
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <unordered_set>

/* modified: 
    State<StateType> */

namespace QComputations {

#ifdef ENABLE_ONEAPI
void vector_normalize(std::vector<COMPLEX>& v);
#endif

namespace {
    inline size_t sum_of_arithmetic_progression(size_t start, size_t end, size_t n) {
        return size_t(double(start + end) / 2 * n);
    }

    // ????
    inline size_t photons_types_count(size_t levels_count) {
        return size_t(double(levels_count) / 2 * (levels_count - 1));
    }

    using ValType = int;
    using CavityId = size_t;
    using COMPLEX = std::complex<double>;
    std::complex<double> gamma(double amplitude, double length, double w_ph) {
        return amplitude * std::exp(std::complex<double>(0, -1) * length * w_ph / QConfig::instance().h());
    }

    const std::vector<std::string> subscript_numbers = {
        "\u2080", "\u2081", "\u2082", "\u2083", "\u2084", "\u2085", "\u2086",
        "\u2087", "\u2088", "\u2089"};

    std::vector<size_t> split_to_groups(size_t system_size, size_t parts_count) {
        assert(system_size % parts_count == 0);

        std::vector<size_t> res;

        auto part_size = system_size / parts_count;
        for (size_t i = 0; i < parts_count; i++) {
            res.emplace_back(i * part_size + part_size - 1);
        }

        return res;
    }

}

std::vector<std::vector<CavityId>> update_neighbours(size_t x_size, size_t y_size, size_t z_size = 1);

struct State_Comparator {
    template<typename StateType>
    bool operator()(const std::shared_ptr<StateType> a, const std::shared_ptr<StateType> b) const {
        return *b < *a;
    }
};

// template<typename StateType>
//     // using BasisType = std::set<std::shared_ptr<StateType>, State_Comparator>;
//     using BasisType = std::unordered_set<std::shared_ptr<StateType>>;

struct State_Hash {
    template<typename StateType>
    size_t operator()(const std::shared_ptr<StateType>& a) const {
        std::hash<std::string> hasher;
        return hasher(a->to_string());
    }
};

struct State_Equal {
    template<typename StateType>
    bool operator()(const std::shared_ptr<StateType>& a, 
                   const std::shared_ptr<StateType>& b) const {
        return *a == *b;
    }
};

template<typename StateType>
using BasisType = std::unordered_set<std::shared_ptr<StateType>, 
                                     State_Hash, State_Equal>;

template<typename StateType>
using StateBasisMap = std::unordered_map<std::shared_ptr<StateType>, size_t, State_Hash, State_Equal>;


template<typename StateType>
std::vector<std::shared_ptr<StateType>> sort_basis(const BasisType<StateType>& basis) {
    std::vector<std::shared_ptr<StateType>> res(basis.begin(), basis.end());
    std::sort(res.begin(), res.end(), State_Comparator());
    return res;
}

class Basis_State {
    public:
        // инициализация пустого состояния
        explicit Basis_State() = default;
        Basis_State(const Basis_State& other) = default;

        virtual size_t hash() const {
            if (!hash_cached_) {
                std::hash<std::string> hasher;
                cached_hash_ = hasher(this->to_string());
                hash_cached_ = true;
            }

            return cached_hash_;
        }

        // groups_count делит кудиты на равные по размеру группы
        explicit Basis_State(size_t qudits_count, ValType max_val = 1, size_t groups_count = 1); // !!!!!!
        // добавление поддержки для разных кудитов
        explicit Basis_State(size_t qudits_count, const std::vector<ValType>& max_vals,
                            size_t groups_count = 1); // !!!!!!

        // qudit_count - чилсо кудитов. max_val - максимальное значение их всех. groups - количество элементов в каждой группе. 
        // В сумме должно получиться qudits_count, иначе ошибка.
        // ВНИМАНИЕ!!! - groups потом хранится в другом виде. А именно, в нём хранятся индексы конца групп. Так сделано в целях удобства индексации.
        // Для получения размера группы есть метод get_group_size.
        explicit Basis_State(size_t qudits_count, ValType max_val, const std::vector<size_t>& groups); // !!!!!!
        // инициализация значений
        explicit Basis_State(const std::vector<ValType>& qudits, ValType max_vals = 1, size_t groups_count = 1); // !!!!!
        explicit Basis_State(const std::vector<ValType>& qudits, ValType max_vals, const std::vector<size_t>& groups); // !!!!
        // инициализация значений с поддержкой разных кудитов
        explicit Basis_State(const std::vector<ValType>& qudits, const std::vector<ValType>& max_vals,
                            size_t groups_count = 1): qudits_(qudits), max_vals_(std::make_shared<std::vector<ValType>>(max_vals)), groups_(std::make_shared<std::vector<size_t>>(split_to_groups(qudits.size(), groups_count))) {}
        // инициализация значений с поддержкой разных кудитов + разных групп
        explicit Basis_State(const std::vector<ValType>& qudits,  const std::vector<ValType>& max_vals,
                            const std::vector<size_t>& groups_sizes); // !!!!!

        explicit Basis_State(const std::string& qudits, ValType max_vals = 1); // !!!!!
        explicit Basis_State(const std::string& qudits, const std::vector<ValType>& max_vals); // !!!!!

        void set_state(const std::string& str_state); // !!!!
        inline void set_qudit(ValType val, size_t qudit_index = 0, size_t group_id = 0) { assert(val <= (*max_vals_)[qudit_index]);
                                    qudits_[this->get_group_start(group_id) + qudit_index] = val; hash_cached_ = false;}
        inline ValType get_qudit(size_t qudit_index = 0, size_t group_id = 0) const { return qudits_[this->get_group_start(group_id) + qudit_index]; }
        inline void append_qudit(ValType init_val = 0, ValType max_val = 1) { groups_->emplace_back(qudits_.size());
                                                                    qudits_.emplace_back(init_val);
                                                                    max_vals_->emplace_back(max_val);
                                                                    hash_cached_ = false; }
        inline std::vector<ValType> qudits() const { return qudits_;}
        inline const std::vector<ValType>& qudits_ref() const { return qudits_; }
        inline ValType* qudits_data() { return qudits_.data();}
        inline const ValType* qudits_data() const { return qudits_.data();}

        inline bool is_empty() const { return qudits_.size() == 0;}

        //Basis_State operator*(const COMPLEX& c) const { auto res = *this; res.set_coef(this->get_coef() * c); return res;}\
        // deprecated
        inline std::string get_info() const { return info_; }
        inline void set_info(const std::string& str) { info_ = str; }


        inline size_t qudits_count() const { return qudits_.size();}

        Basis_State reduce(const std::vector<size_t>& qudits) const;

        inline std::vector<size_t> get_groups() const { return *groups_; }
        // inline std::vector<size_t> get_groups() const { return groups_; }

        size_t get_group_id(size_t qudit_index) const { 
            size_t res = 0; 
            while (get_group_end(res) > qudit_index) { 
                res++; 
            } 

            return res;
        }

        inline size_t get_group_start(size_t group_id) const { return ((group_id == 0) ? 0 : (*groups_)[group_id - 1] + 1);}
        inline size_t group_start(size_t group_id) const { return ((group_id == 0) ? 0 : (*groups_)[group_id - 1] + 1);}
        inline size_t get_group_end(size_t group_id) const { return (*groups_)[group_id]; } // Включительно
        inline size_t group_end(size_t group_id) const { return (*groups_)[group_id]; }
        inline size_t get_groups_count() const { return groups_->size(); }
        inline size_t groups_count() const { return groups_->size(); }
        inline size_t get_group_size(size_t group_id) const { return this->get_group_end(group_id) - this->get_group_start(group_id) + 1; }
        inline size_t group_size(size_t group_id) const { return this->get_group_size(group_id); }
        Basis_State get_group(size_t group_id) const; // !!!!
        void set_group(size_t group_id, const std::string& group_state, size_t index_start = 0); // !!!!
        void set_groups(const std::vector<size_t>& groups); /// !!!

        void reset_cache() { hash_cached_ = false; }

        virtual std::string group_to_string(size_t group_id) const; // !!!
        virtual std::string to_string() const; // !!!!
        inline virtual bool operator==(const Basis_State& other) const { return qudits_ == other.qudits_; }
        inline virtual bool operator<(const Basis_State& other) const { return this->to_string() > other.to_string(); }
        
        // inline void set_max_val(ValType val, size_t qudit_index, size_t group_id = 0) { max_vals_[this->get_group_start(group_id) + qudit_index] = val; }
        // inline ValType get_max_val(size_t qudit_index, size_t group_id = 0) const { return max_vals_[this->get_group_start(group_id) + qudit_index]; }
        // std::vector<ValType> max_vals() const { return max_vals_; }

        inline void set_max_val(ValType val, size_t qudit_index, size_t group_id = 0) { (*max_vals_)[this->get_group_start(group_id) + qudit_index] = val; }
        inline ValType get_max_val(size_t qudit_index, size_t group_id = 0) const { return (*max_vals_)[this->get_group_start(group_id) + qudit_index]; }
        std::vector<ValType> max_vals() const { return *max_vals_; }

        size_t get_index(const BasisType<Basis_State>& basis) const; // !!!
        size_t get_qudit_index(size_t qudit_index, size_t group_index) const; // !!!

        // inline void clear() { qudits_.resize(0); max_vals_.resize(0); groups_.resize(0); }
        inline void clear() { qudits_.resize(0); max_vals_->resize(0); groups_->resize(0); hash_cached_ = false; }
        inline void set_zero() { qudits_ = std::vector<ValType>(qudits_.size(), 0); hash_cached_ = false;}
        // Переписать
        inline bool is_all_qudits_zero() const { if (qudits_ == std::vector<ValType>(qudits_.size(), 0)) return true; return false;}
    protected:
        // deprecated
        std::string info_;

        std::vector<ValType> qudits_;
        // !!!!!!!!!!!!!!!!! new !!!!!!!!!!!!!!!!
        // std::vector<ValType> max_vals_;
        // std::vector<size_t> groups_;
        std::shared_ptr<std::vector<ValType>> max_vals_;
        std::shared_ptr<std::vector<size_t>> groups_;
        mutable size_t cached_hash_ = 0;
        mutable bool hash_cached_ = false;
};

template<typename StateType>
size_t get_index_state_in_basis(const StateType& state, const BasisType<StateType>& basis) {
    size_t res = 0;
    for (auto p: basis) {
        if (state == *p) {
            return res;
        }

        res++;
    }

    assert(false); // Элемента нет в базисе
    return 0;
}

class TCH_State: public Basis_State {
    using E_LEVEL = int;
    //using CavityId = size_t;
    using AtomId = size_t;

    public:
        TCH_State() = default;
        TCH_State(const Basis_State& base): Basis_State(base), x_size_(base.get_groups_count()), y_size_(1), z_size_(1), neighbours_(update_neighbours(x_size_, y_size_, z_size_)) {}
        //TCH_State(size_t x_size = 1, size_t y_size = 1, size_t z_size = 1);
        TCH_State(const TCH_State& state) = default;
        TCH_State(const std::vector<size_t>& grid_config, size_t levels_count = 2);
        //explicit TCH_State(const std::string&, const std::string& format = QConfig::instance().state_format(),
        //            const std::string& del = QConfig::instance().state_delimeter(),
        //            bool is_freq_display = QConfig::instance().is_freq_display());

        size_t x_size() const { return x_size_; }
        size_t y_size() const { return y_size_; }
        size_t z_size() const { return z_size_; }

        // size_t n(CavityId id = 0, E_LEVEL e_from = 0, E_LEVEL e_to = 1) const { return grid_states_[id].n(e_from, e_to); } // get amount of photons in cavity with id = id
        // void set_n(size_t n, CavityId id = 0, E_LEVEL e_from = 0, E_LEVEL e_to = 1) { grid_states_[id].set_n(n, e_from, e_to); } // set n photons in cavity with id = id
        // size_t m(CavityId id) const { return grid_states_[id].m(); } // get amount of atoms in cavity with id = id

        // change grid shapes
        void reshape(size_t x_size, size_t y_size, size_t z_size);

        // TMP realizations
        void set_waveguide(double amplitude, double length);
        //void set_waveguide(const Matrix<std::pair<double, double>>& A);
        void set_waveguide(size_t from_cavity_id, size_t to_cavity_id, double amplitude, double length = QConfig::instance().waveguides_length());
        // set entire state in cavity with id = id
        void set_state(CavityId id, const TCH_State& state);

        // add cavity to grid (Don't safe, be careful)
        //TCH_State add_state(const TCH_State& state) const;


        inline void set_g_level(COMPLEX g, int level_from = 0, int level_to = 1) {
            g_[level_from][level_to] = g;
            g_[level_to][level_from] = std::conj(g);
        }

        inline COMPLEX g_level(int level_from, int level_to) const {
            return g_[level_from][level_to];
        }

        inline void set_move(bool val, size_t atom_index, CavityId id, int level_from, int level_to) {
            is_move_[this->get_qudit_index(atom_index + ph_types_count(), id)][level_from][level_to] = val;
            is_move_[this->get_qudit_index(atom_index + ph_types_count(), id)][level_to][level_from] = val;
        }

        inline bool is_move(size_t atom_index, CavityId id, int level_from, int level_to) {
            return is_move_[this->get_qudit_index(atom_index + ph_types_count(), id)][level_from][level_to];
        }

        COMPLEX g(size_t atom_index = 0, CavityId id = 0, int level_from = 0, int level_to = 1) const {
            if (is_move_[this->get_qudit_index(atom_index + ph_types_count(), id)][level_from][level_to]) {
                return g_[level_from][level_to];
            } else {
                return 0;
            }
        }

        double w_move(int level_from = 0, int level_to = 1) const {
            return std::abs(freq_levels_[level_to] - freq_levels_[level_from]);
        }

        double w(int level) const {
            return freq_levels_[level];
        }

        void set_w(double w, int level) {
            freq_levels_[level] = w;
        }

        size_t freqs_count() const { return freq_levels_.size();}
        size_t levels_count() const { return freq_levels_.size();}

        size_t cavities_count() const { return this->groups_count(); }
        size_t cavity_atoms_count(CavityId id) const { return this->get_group_size(id) - this->ph_types_count(); }
        //size_t cavity_size(CavityId id) const { return cavity_atoms_count(id) + 1; }
        size_t m(CavityId id) const { return cavity_atoms_count(id); }
        ValType n(CavityId id, int level_from = 0, int level_to = 1) const { 
            auto levels_count = freq_levels_.size();
            assert(level_from != level_to);
            if (level_from > level_to) {
                auto tmp = level_from;
                level_from = level_to;
                level_to = tmp;
            }

            return this->get_qudit(sum_of_arithmetic_progression(levels_count - 1, levels_count - level_from, level_from) + level_to - level_from - 1, id);
        }

        void set_n(ValType n, CavityId id, int level_from = 0, int level_to = 1) { 
            auto levels_count = freq_levels_.size();
            assert(level_from != level_to);
            if (level_from > level_to) {
                auto tmp = level_from;
                level_from = level_to;
                level_to = tmp;
            }

            this->set_qudit(n, sum_of_arithmetic_progression(levels_count - 1, levels_count - level_from, level_from) + level_to - level_from - 1, id); 
        }

        size_t ph_types_count() const {
            return photons_types_count(freq_levels_.size());
        }

        void set_atom(ValType val, size_t atom_index = 0, CavityId cavity_id = 0) { assert(this->m(cavity_id) > atom_index); this->set_qudit(val, atom_index + ph_types_count(), cavity_id); }

        ValType get_atom(size_t atom_index = 0, CavityId id = 0) const {
            return this->get_qudit(atom_index + ph_types_count(), id);
        }

        // Return state vector from cavity
        TCH_State get_state_in_cavity(CavityId cavity_id) const { return TCH_State(this->get_group(cavity_id)); }
        TCH_State operator[](CavityId cavity_id) const { return TCH_State(this->get_group(cavity_id)); }

        CavityId get_index_of_cavity(size_t x, size_t y = 0, size_t z = 0) const { return z * y_size_ * x_size_ + y * x_size_ + x; }
        
        // Рудимент
        // size_t get_index() const;
        // void set_term(size_t atom_index, double term, CavityId cavity_id) { grid_states_[cavity_id].set_term(atom_index, term); }
        // double get_term(size_t atom_index, CavityId cavity_id) const { return grid_states_[cavity_id].get_term(atom_index); }

        // Get index of state in basis
        size_t get_index(const std::set<TCH_State>& basis) const;
        size_t get_max_size() const;

        // return energy in state (photons + atoms in state one)
        size_t get_grid_energy() const;

        size_t get_energy(CavityId cavity_id) const;

        double get_leak_gamma(CavityId id) const { return gamma_leak_cavities_[id]; }
        double get_gain_gamma(CavityId id) const { return gamma_gain_cavities_[id]; }

        void set_leak_for_cavity(CavityId id, double gamma) { gamma_leak_cavities_[id] = gamma;}
        void set_gain_for_cavity(CavityId id, double gamma) { gamma_gain_cavities_[id] = gamma;}

        std::set<CavityId> get_cavities_with_leak() const;
        std::set<CavityId> get_cavities_with_gain() const;

        // Matrix<COMPLEX> get_gamma() const { return gamma_; }
        COMPLEX get_gamma(CavityId from_id, CavityId to_id) const {
            bool is_conj = false;
            if (from_id > to_id) {
                auto tmp = from_id;
                from_id = to_id;
                to_id = tmp;
                is_conj = true;
            }
            auto res = gamma(waveguides_[from_id][to_id].first, waveguides_[from_id][to_id].second, QConfig::instance().w());

            if (is_conj) {
                res = std::conj(res);
            }

            return res;
        }

        std::set<CavityId> get_cavities_with_atoms() const { return cavities_with_atoms_; }

        // Like a hash
        //BigUInt to_uint() const;

        // Change state to with BigUint = state_num
        //void from_uint(const BigUInt& state_num);

        std::vector<CavityId> get_neighbours(CavityId cavity_id) const { return neighbours_[cavity_id]; }

        virtual std::string group_to_string(size_t group_id) const override {
            std::string res = "|";
            for (int w_from = 0; w_from < this->freqs_count(); w_from++) {
                for (int w_to = w_from + 1; w_to < this->freqs_count(); w_to++) {
                    res += std::to_string(this->n(group_id, w_from, w_to)) + "_" + std::to_string(w_from) + std::to_string(w_to);

                    if (w_from != this->freqs_count() - 2) {
                        res += ",";
                    } else {
                        res += ";";
                    }
                }
            }

            if (this->m(group_id) != 0) {
                for (int i = 0; i < this->m(group_id); i++) {
                    res += std::to_string(this->get_atom(i, group_id));

                    if (i != this->m(group_id) - 1) {
                        res += ",";
                    } else {
                        res += ">";
                    }
                }
            } else {
                res[res.size() - 1] = '>';
            }

            return res;
        }
    private:
        size_t x_size_;
        size_t y_size_;
        size_t z_size_;
        std::vector<double> freq_levels_;
        Matrix<COMPLEX> g_;
        std::vector<std::vector<std::vector<bool>>> is_move_;

        std::set<CavityId> cavities_with_atoms_;
        Matrix<std::pair<double, double>> waveguides_;
        std::vector<std::vector<CavityId>> neighbours_;
        std::vector<double> gamma_leak_cavities_;
        std::vector<double> gamma_gain_cavities_;
};

// ------------------------------ State ---------------------------------

template<typename StateType>
class State {
    public:
        explicit State() = default;
        State(const State<StateType>& state) = default;
        State(const StateType& state, COMPLEX c = COMPLEX(1, 0)) {
            if (!state.is_empty()) {
                auto st_pt = std::make_shared<StateType>(state);
                state_vec_.emplace_back(c);
                // state_basis_.insert(st_pt);
                state_map_[st_pt] = 0;
                // hash_map_[st_pt->hash()] = 0;
                sorted_basis_.emplace_back(st_pt);
                is_sorted_ = true;
            }
        }

        State(const std::shared_ptr<StateType>& state, COMPLEX c = COMPLEX(1, 0)) {
            if (!(state->is_empty())) {
                state_vec_.emplace_back(c);
                // state_basis_.insert(st_pt);
                state_map_[state] = 0;
                // hash_map_[st_pt->hash()] = 0;
                sorted_basis_.emplace_back(state);
                is_sorted_ = true;
            }
        }

        State(const StateType& state, const BasisType<StateType>& basis) {
            size_t index = 0;
            for (auto st: basis) {
                //state_components_.insert(std::shared_ptr<StateType>(new StateType(*st)));
                // state_basis_.insert(st);
                state_map_[st] = index;
                // hash_map_[st->hash()] = index++;

                if (*st == state) {
                    state_vec_.emplace_back(1, 0);
                } else {
                    state_vec_.emplace_back(0, 0);
                }
            }
        }

        State(const BasisType<StateType>& basis) {
            size_t index = 0;
            for (auto st: basis) {
                //state_components_.insert(std::shared_ptr<StateType>(new StateType(*st)));
                // state_basis_.insert(st);
                state_vec_.emplace_back(0, 0);
                state_map_[st] = index++;
                // hash_map_[st->hash()] = index++;
            }
        }

        State(const std::vector<std::shared_ptr<StateType>>& basis, bool is_sorted = true): is_sorted_(is_sorted) {
            size_t index = 0;
            for (auto st: basis) {
                //state_components_.insert(std::shared_ptr<StateType>(new StateType(*st)));
                // state_basis_.insert(st);
                state_vec_.emplace_back(0, 0);
                state_map_[st] = index++;
                // hash_map_[st->hash()] = index++;
            }

            if (is_sorted) {
                sorted_basis_ = basis;
            }
        }

        State(const std::vector<std::shared_ptr<StateType>>& basis, const std::vector<COMPLEX>& state_vec, bool is_sorted = true): state_vec_(state_vec), is_sorted_(is_sorted) {
            size_t index = 0;
            for (auto st: basis) {
                //state_components_.insert(std::shared_ptr<StateType>(new StateType(*st)));
                // state_basis_.insert(st);
                // state_vec_.emplace_back(state_vec[index]);
                state_map_[st] = index++;
                // hash_map_[st->hash()] = index++;
            }

            if (is_sorted) {
                sorted_basis_ = basis;
            }
        }

        /*
        operator State<Basis_State>() {
            State<Basis_State> res;
            res.state_vec_ = this->state_vec_;

            for (const auto& st: this->state_components_) {
                res.state_components_.insert(Basis_State(st));
            }

            return res;
        }
        */

        // Работает не типично. Возвращает other, если this не пустое
        State<StateType> operator*(const State<StateType>& other) const { 
            return (this->is_empty() ? State<StateType>() : other);  
        }

        bool is_empty() const { return state_vec_.size() == 0; }

        State<StateType> operator*(const COMPLEX& c) const {
            State<StateType> res(*this);
            for (auto& num: res.state_vec_) {
                num *= c;
            }

            return res;
        }

        void operator*=(const COMPLEX& c) {
            for (auto& st: this->state_vec_) {
                st *= c;
            }
        }

        void operator/=(const COMPLEX& c) {
            for (auto& st: this->state_vec_) {
                st /= c;
            }
        }

#ifdef ENABLE_ONEAPI
        void normalize() {
            vector_normalize(this->state_vec_);
        }
#endif

        void operator+=(const State<StateType>& other) {
            for (const auto& p : other.state_map_) {
                auto it = state_map_.find(p.first);
                if (it == state_map_.end()) {
                    state_map_[p.first] = state_vec_.size();
                    state_vec_.emplace_back(other.state_vec_[p.second]);
                    is_sorted_ = false;
                } else {
                    state_vec_[it->second] += other.state_vec_[p.second];
                }
            }
        }

        void operator-=(const State<StateType>& other) {
            for (const auto& p : other.state_map_) {
                auto it = state_map_.find(p.first);
                if (it == state_map_.end()) {
                    state_map_[p.first] = state_vec_.size();
                    state_vec_.emplace_back(-other.state_vec_[p.second]);
                    is_sorted_ = false;
                } else {
                    state_vec_[it->second] -= other.state_vec_[p.second];
                }
            }
        }

        inline bool is_in_state(const std::shared_ptr<StateType>& state) const {
            return state_map_.find(state) != state_map_.end();
        }

        inline bool is_in_state(const StateType& state) const {
            return is_in_state(std::make_shared<StateType>(state));
        }

        inline COMPLEX& operator[](size_t index) { return state_vec_[index];}
        inline COMPLEX operator[](size_t index) const { return state_vec_[index];}
        inline COMPLEX& operator[](const StateType& st) { return (*this)[std::make_shared<StateType>(st)];}
        inline COMPLEX operator[](const StateType& st) const { return (*this)[std::make_shared<StateType>(st)];}
        inline COMPLEX& operator[](const std::shared_ptr<StateType>& st) { return state_vec_[(state_map_.find(st))->second];}
        inline COMPLEX operator[](const std::shared_ptr<StateType>& st) const { return state_vec_[(state_map_.find(st))->second];}
        inline std::shared_ptr<StateType> operator()(size_t index) const { this->sort(); return sorted_basis_[index]; }
        inline size_t size() const { return state_vec_.size(); }
        inline bool is_sorted() const { return is_sorted_; }
        inline void set_sorted(bool is_sorted) { is_sorted_ = is_sorted; }

        // size_t get_index(const StateType& state) const {
        //     size_t index = 0;
        //     for (auto st: state_components_) {
        //         if (*st == state) {
        //             return index;
        //         }

        //         index++;
        //     }

        //     return index;
        // }

        // inline size_t get_index(const StateType&)

        void insert(const StateType& state, const COMPLEX& amplitude = COMPLEX(0, 0)) {
            auto st_pt = std::make_shared<StateType>(state);
            insert(st_pt, amplitude);
        }

        void insert(const std::shared_ptr<StateType>& state, const COMPLEX& amplitude = COMPLEX(0, 0)) {
            if (!is_in_state(state)) {
                // state_components_.insert(state);
                state_map_[state] = state_vec_.size();
                state_vec_.emplace_back(amplitude);
                is_sorted_ = false;
            } else {
                if (amplitude != COMPLEX(0, 0)) {
                    state_vec_[state_map_[state]] = amplitude;
                }
            }
        }

        // void set_state_components(const BasisType<StateType>& st) { state_components_ = st; is_sorted_ = false; }
        // void set_vector(const std::vector<COMPLEX>& v) { state_vec_ = v; }
        inline void set_all(COMPLEX a) {
            state_vec_.assign(state_vec_.size(), a);
        }

        inline void set_zero() {
            set_all(COMPLEX(0, 0));
        }

        // BasisType<StateType> get_state_components() const { return state_components_; }
        // BasisType<StateType> state_components() const { return state_components_; }
        inline StateBasisMap<StateType> state_map() const { this->sort(); return state_map_; }
        inline StateBasisMap<StateType> get_map() const { this->sort(); return state_map_; }
        inline std::vector<std::shared_ptr<StateType>> get_basis() const { this->sort(); return sorted_basis_; }
        inline std::vector<COMPLEX> get_vector() const { this->sort(); return state_vec_;}
        inline std::vector<COMPLEX> vector() const { this->sort(); return state_vec_;}
        inline void set_vector(const std::vector<COMPLEX>& v) { state_vec_ = v; }

        std::vector<double> to_probs() const {
            if (!is_sorted_) this->sort();
            std::vector<double> res(state_vec_.size());

            // !!!! state_map !!!!!!!!!!!!!
            // for (size_t i = 0; i < state_vec_.size(); ++i) {
            //     auto tmp = std::abs(state_vec_[i]);
            //     res[i] = tmp * tmp;
            // }

            for (auto& p: state_map_) {
                auto tmp = std::abs(state_vec_[p.second]);
                res[p.second] = tmp * tmp;
            }

            return res;
        }

        void init_state_by_func(const std::function<COMPLEX(const StateType&)>& func) {
            // if (!is_sorted_) this->sort();
            for (auto& p: state_map_) {
                state_vec_[p.second] = func(*(p.first));
            }
        }

        std::string to_string() const {
            std::string res;

            size_t index = 0;
            for (auto& p: state_map_) {
                res += "(" + std::to_string(state_vec_[p.second].real()) + " + " + std::to_string(state_vec_[p.second].imag()) + "j)";

                res += " * ";
                res += (p.first)->to_string();

                if (++index != state_map_.size()) {
                    res += " + ";
                }
            }

            return res;
        }

        State<StateType> fit_to_basis(const BasisType<StateType>& basis) const {
            State<StateType> res;
            std::vector<std::shared_ptr<StateType>> basis_vector(basis.begin(), basis.end());
            std::sort(basis_vector.begin(), basis_vector.end(), State_Comparator());
            this->sort();

            // //res.state_vec_ = std::vector<COMPLEX>(basis.size(), 0);
            // res.set_vector(std::vector<COMPLEX>(basis.size(), 0));
            // //res.state_components_ = basis;
            // res.set_state_components(basis);

            // // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            // size_t index = 0;
            // for (auto state: basis_vector) {
            //     size_t my_index = 0;
            //     for (auto my_state: this->state_components_) {
            //         if ((*state) == (*my_state)) {
            //             res[index] = this->state_vec_[my_index];
            //             break;
            //         }

            //         my_index++;
            //     }

            //     index++;
            // }

            // return res;

            size_t cur_index = 0;
            for (size_t i = 0; i < basis_vector.size(); i++) {
                if ((*(basis_vector[i])).Basis_State::operator==(Basis_State(*(sorted_basis_[cur_index])))) {
                    res.insert(basis_vector[i], state_vec_[cur_index]);
                    cur_index++;
                } else {
                    res.insert(basis_vector[i], COMPLEX(0, 0));
                }
            }

            return res;
        }

        inline size_t get_index(const StateType& state) const {
            this->sort(); return state_map_[std::make_shared<StateType>(state)];
        }

        inline size_t get_index(const std::shared_ptr<StateType>& state) const {
            this->sort(); return state_map_[state];
        }

        State<Basis_State> fit_to_basis_state(const BasisType<Basis_State>& basis) const {
            State<Basis_State> res;
            std::vector<std::shared_ptr<Basis_State>> basis_vector(basis.begin(), basis.end());
            std::sort(basis_vector.begin(), basis_vector.end(), State_Comparator());
            this->sort();

            size_t cur_index = 0;
            for (size_t i = 0; i < basis_vector.size(); i++) {
                if (cur_index != sorted_basis_.size() && (*(basis_vector[i])).Basis_State::operator==(Basis_State(*(sorted_basis_[cur_index])))) {
                    res.insert(basis_vector[i], state_vec_[cur_index]);
                    cur_index++;
                } else {
                    res.insert(basis_vector[i], COMPLEX(0, 0));
                }
            }

            return res;
        }

        State<Basis_State> fit_to_basis_state(const std::vector<std::shared_ptr<Basis_State>>& basis_vector) const {
            State<Basis_State> res;
            this->sort();

            size_t cur_index = 0;
            for (size_t i = 0; i < basis_vector.size(); i++) {
                if (cur_index != sorted_basis_.size() && (*basis_vector[i])==(Basis_State(*(sorted_basis_[cur_index])))) {
                    res.insert(basis_vector[i], state_vec_[cur_index]);
                    cur_index++;
                } else {
                    res.insert(basis_vector[i], COMPLEX(0, 0));
                }
            }

            return res;
        }

        std::vector<size_t> fit_indexes_to_basis(const std::vector<std::shared_ptr<StateType>>& basis_vector) const {
            std::vector<size_t> idxs;
            this->sort();

            size_t cur_idx = 0;
            for (auto state: basis_vector) {
                if (cur_idx != sorted_basis_.size() && *state == *(sorted_basis_[cur_idx])) {
                    idxs.emplace_back(cur_idx);
                }

                cur_idx++;
            }

            assert(idxs.size() == sorted_basis_.size());

            return idxs;
        }

        // // Полное копирование. Память выделяется ещё раз
        // State<StateType> copy() const {
        //     State<StateType> res;
        //     for (auto (st, index): this->state_map_) {
        //         res.insert(st, state_vec_[index]);
        //     }

        //     return res;
        // }

        void sort() const {
            if (is_sorted_) return;
            sorted_basis_.clear();
            std::vector<COMPLEX> new_state_vec(state_vec_.size());

            for (auto p: state_map_) {
                sorted_basis_.emplace_back(p.first);
            }

            std::sort(sorted_basis_.begin(), sorted_basis_.end(), State_Comparator());

            for (size_t i = 0; i < sorted_basis_.size(); i++) {
                new_state_vec[i] = state_vec_[state_map_[sorted_basis_[i]]];
                state_map_[sorted_basis_[i]] = i;
            }

            state_vec_ = new_state_vec;
            is_sorted_ = true;
        }

        void clear() {
            state_vec_.clear();

            //for (auto p: state_components_) {
            //    delete p;
            //}

            // state_components_.clear();
            state_map_.clear();
            sorted_basis_.clear();
            is_sorted_ = false;
        }
    private:
        mutable std::vector<COMPLEX> state_vec_;
        // BasisType<StateType> state_basis_;
        // std::vector<std::shared_ptr<StateType>> sorted_components_;
        mutable StateBasisMap<StateType> state_map_;
        // std::unordered_map<size_t, std::vector<size_t>> hash_map_;
        mutable std::vector<std::shared_ptr<StateType>> sorted_basis_;
        mutable bool is_sorted_ = false;
};

// <a|b>
template<typename StateType>
COMPLEX scalar_product(const State<StateType>& a, const State<StateType>& b) {
    // auto a_map = a.state_map();
    // auto b_map = b.state_map();
    auto a_basis = a.get_basis();
    auto b_basis = b.get_basis();

    COMPLEX res = 0;
    if (a_basis.size() < b_basis.size()) {
        for (size_t i = 0; i < a_basis.size(); i++) {
            if (b.is_in_state(a_basis[i])) {
                res += std::conj(a[i]) * b[a_basis[i]];
            }
        }
    } else {
        for (size_t i = 0; i < b_basis.size(); i++) {
            if (a.is_in_state(b_basis[i])) {
                res += std::conj(a[b_basis[i]]) * b[i];
            }
        }
    }

    return res;
}

} // namespace QComputations

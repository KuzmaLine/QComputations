#pragma once
#include <algorithm>
#include <cassert>
#include <complex>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "matrix.hpp"

namespace QComputations {

    void vector_normalize(std::vector<COMPLEX>& v);

    namespace {
        using ValType = int;
        using CavityId = size_t;
        using COMPLEX = std::complex<double>;
        std::complex<double> gamma(double amplitude, double length, double w_ph) {
            return amplitude * std::exp(std::complex<double>(0, -1) * length * w_ph / QConfig::instance().h());
        }

        const std::vector<std::string> subscript_numbers =
            {"\u2080", "\u2081", "\u2082", "\u2083", "\u2084", "\u2085", "\u2086", "\u2087", "\u2088", "\u2089"};

        std::vector<size_t> split_to_groups(size_t system_size, size_t parts_count) {
            assert(system_size % parts_count == 0);

            std::vector<size_t> res;

            auto part_size = system_size / parts_count;
            for (size_t i = 0; i < parts_count; i++) {
                res.emplace_back(i * part_size + part_size - 1);
            }

            return res;
        }

    }  // namespace

    std::vector<std::vector<CavityId>> update_neighbours(size_t x_size, size_t y_size, size_t z_size = 1);

    struct State_Comparator {
        template <typename StateType>
        bool operator()(const std::shared_ptr<StateType> a, const std::shared_ptr<StateType> b) const {
            return *a < *b;
        }
    };

    template <typename StateType>
    using BasisType = std::set<std::shared_ptr<StateType>, State_Comparator>;

    class Basis_State {
       public:
        // инициализация пустого состояния
        explicit Basis_State() = default;
        Basis_State(const Basis_State& other) = default;

        // groups_count делит кудиты на равные по размеру группы
        explicit Basis_State(size_t qudits_count, ValType max_val = 1, size_t groups_count = 1);
        // добавление поддержки для разных кудитов
        explicit Basis_State(size_t qudits_count, const std::vector<ValType>& max_vals, size_t groups_count = 1);

        // qudit_count - чилсо кудитов. max_val - максимальное значение их всех.
        // groups - количество элементов в каждой группе. В сумме должно получиться
        // qudits_count, иначе ошибка. ВНИМАНИЕ!!! - groups потом хранится в другом
        // виде. А именно, в нём хранятся индексы начала групп. Так сделано в целях
        // удобства индексации. Для получения размера группы есть метод
        // get_group_size.
        explicit Basis_State(size_t qudits_count, ValType max_val, const std::vector<size_t>& groups);
        // инициализация значений
        explicit Basis_State(const std::vector<ValType>& qudits, ValType max_vals = 1, size_t groups_count = 1);
        explicit Basis_State(const std::vector<ValType>& qudits, ValType max_vals, const std::vector<size_t>& groups);
        // инициализация значений с поддержкой разных кудитов
        explicit Basis_State(const std::vector<ValType>& qudits,
                             const std::vector<ValType>& max_vals,
                             size_t groups_count = 1)
            : qudits_(qudits), max_vals_(max_vals), groups_(split_to_groups(qudits.size(), groups_count)) {}
        // инициализация значений с поддержкой разных кудитов + разных групп
        explicit Basis_State(const std::vector<ValType>& qudits,
                             const std::vector<ValType>& max_vals,
                             const std::vector<size_t>& groups_sizes);

        explicit Basis_State(const std::string& qudits, ValType max_vals = 1);
        explicit Basis_State(const std::string& qudits, const std::vector<ValType>& max_vals);

        void set_state(const std::string& str_state);
        void set_qudit(ValType val, size_t qudit_index, size_t group_id = 0) {
            assert(qudit_index < this->get_group_size(group_id));
            assert(val <= max_vals_[qudit_index]);
            qudits_[this->get_group_start(group_id) + qudit_index] = val;
        }
        void set_atom(ValType val, size_t atom_index, size_t group_id) {
            this->set_qudit(val, atom_index + 1, group_id);
        }
        ValType get_qudit(size_t qudit_index, size_t group_id = 0) const {
            assert(qudit_index < this->get_group_size(group_id));
            return qudits_[this->get_group_start(group_id) + qudit_index];
        }
        void append_qudit(ValType init_val = 0, ValType max_val = 1) {
            groups_.emplace_back(qudits_.size());
            qudits_.emplace_back(init_val);
            max_vals_.emplace_back(max_val);
        }
        bool is_empty() const { return qudits_.size() == 0; }

        std::string get_info() const { return info_; }
        void set_info(const std::string& str) { info_ = str; }

        size_t qudits_count() const { return qudits_.size(); }

        std::vector<size_t> get_groups() const { return groups_; }
        size_t get_group_start(size_t group_id) const {
            assert(group_id < this->get_groups_count());
            return ((group_id == 0) ? 0 : groups_[group_id - 1] + 1);
        }
        size_t group_start(size_t group_id) const { return this->get_group_start(group_id); }
        size_t get_group_end(size_t group_id) const {
            assert(group_id < this->get_groups_count());
            return groups_[group_id];
        }
        size_t group_end(size_t group_id) const { return this->get_group_end(group_id); }
        size_t get_groups_count() const { return groups_.size(); }
        size_t groups_count() const { return this->get_groups_count(); }
        size_t get_group_size(size_t group_id) const {
            return this->get_group_end(group_id) - this->get_group_start(group_id) + 1;
        }
        size_t group_size(size_t group_id) const {
            return this->get_group_end(group_id) - this->get_group_start(group_id) + 1;
        }
        Basis_State get_group(size_t group_id) const;
        void set_group(size_t group_id, const std::string& group_state, size_t index_start = 0);

        virtual std::string group_to_string(size_t group_id) const;
        virtual std::string to_string() const;
        virtual bool operator==(const Basis_State& other) const {
            assert(max_vals_ == other.max_vals_ and groups_ == other.groups_);
            return qudits_ == other.qudits_;
        }
        virtual bool operator<(const Basis_State& other) const { return this->to_string() > other.to_string(); }

        void set_max_val(ValType val, size_t qudit_index, size_t group_id = 0) {
            assert(qudit_index < this->get_group_size(group_id));
            max_vals_[this->get_group_start(group_id) + qudit_index] = val;
        }
        ValType get_max_val(size_t qudit_index, size_t group_id = 0) const {
            assert(qudit_index < this->get_group_size(group_id));
            return max_vals_[this->get_group_start(group_id) + qudit_index];
        }
        std::vector<ValType> max_vals() const { return max_vals_; }

        size_t get_index(const BasisType<Basis_State>& basis) const;

        void clear() {
            qudits_.resize(0);
            max_vals_.resize(0);
            groups_.resize(0);
        }
        void set_zero() { qudits_ = std::vector<ValType>(qudits_.size(), 0); }

       protected:
        std::string info_;
        std::vector<ValType> qudits_;
        std::vector<ValType> max_vals_;
        std::vector<size_t> groups_;
    };

    template <typename StateType>
    size_t get_index_state_in_basis(const StateType& state, const BasisType<StateType>& basis) {
        size_t res = 0;
        for (auto p : basis) {
            if (state == *p) {
                return res;
            }

            res++;
        }

        assert(false);  // Элемента нет в базисе
        return 0;
    }

    class TCH_State : public Basis_State {
        using E_LEVEL = int;
        // using CavityId = size_t;
        using AtomId = size_t;

       public:
        TCH_State() = default;
        TCH_State(const Basis_State& base)
            : Basis_State(base),
              x_size_(base.get_groups_count()),
              y_size_(1),
              z_size_(1),
              neighbours_(update_neighbours(x_size_, y_size_, z_size_)) {}
        TCH_State(const TCH_State& state) = default;
        TCH_State(const std::vector<size_t>& grid_config);

        size_t x_size() const { return x_size_; }
        size_t y_size() const { return y_size_; }
        size_t z_size() const { return z_size_; }

        // change grid shapes
        void reshape(size_t x_size, size_t y_size, size_t z_size);

        // TMP realizations
        void set_waveguide(double amplitude, double length);
        // void set_waveguide(const Matrix<std::pair<double, double>>& A);
        void set_waveguide(size_t from_cavity_id,
                           size_t to_cavity_id,
                           double amplitude,
                           double length = QConfig::instance().waveguides_length());
        // set entire state in cavity with id = id
        void set_state(CavityId id, const TCH_State& state);

        // add cavity to grid (Don't safe, be careful)
        TCH_State add_state(const TCH_State& state) const;

        size_t cavities_count() const { return groups_.size(); }
        size_t cavity_atoms_count(CavityId id) const { return this->get_group_end(id) - this->get_group_start(id); }
        // size_t cavity_size(CavityId id) const { return cavity_atoms_count(id) +
        // 1;
        // }
        size_t m(CavityId id) const { return cavity_atoms_count(id); }
        ValType n(CavityId id) const { return qudits_[get_group_start(id)]; }
        void set_n(ValType n, CavityId id) { qudits_[get_group_start(id)] = n; }

        // Return state vector from cavity
        TCH_State get_state_in_cavity(CavityId cavity_id) const { return TCH_State(this->get_group(cavity_id)); }
        TCH_State operator[](CavityId cavity_id) const { return TCH_State(this->get_group(cavity_id)); }

        CavityId get_index_of_cavity(size_t x, size_t y = 0, size_t z = 0) const {
            return z * y_size_ * x_size_ + y * x_size_ + x;
        }

        // Get index of state in basis
        size_t get_index(const std::set<TCH_State>& basis) const;
        size_t get_max_size() const;

        // return energy in state (photons + atoms in state one)
        size_t get_grid_energy() const;

        size_t get_energy(CavityId cavity_id) const;

        double get_leak_gamma(CavityId id) const { return gamma_leak_cavities_[id]; }
        double get_gain_gamma(CavityId id) const { return gamma_gain_cavities_[id]; }

        void set_leak_for_cavity(CavityId id, double gamma) { gamma_leak_cavities_[id] = gamma; }
        void set_gain_for_cavity(CavityId id, double gamma) { gamma_gain_cavities_[id] = gamma; }

        std::set<CavityId> get_cavities_with_leak() const;
        std::set<CavityId> get_cavities_with_gain() const;

        COMPLEX get_gamma(CavityId from_id, CavityId to_id) const {
            bool is_conj = false;
            if (from_id > to_id) {
                auto tmp = from_id;
                from_id = to_id;
                to_id = tmp;
                is_conj = true;
            }
            auto res =
                gamma(waveguides_[from_id][to_id].first, waveguides_[from_id][to_id].second, QConfig::instance().w());

            if (is_conj) {
                res = std::conj(res);
            }

            return res;
        }

        std::set<CavityId> get_cavities_with_atoms() const { return cavities_with_atoms_; }

        std::vector<CavityId> get_neighbours(CavityId cavity_id) const { return neighbours_[cavity_id]; }

        size_t hash() const;

       private:
        size_t x_size_;
        size_t y_size_;
        size_t z_size_;

        std::set<CavityId> cavities_with_atoms_;
        Matrix<std::pair<double, double>> waveguides_;
        std::vector<std::vector<CavityId>> neighbours_;
        std::vector<double> gamma_leak_cavities_;
        std::vector<double> gamma_gain_cavities_;
    };

    // ------------------------------ State ---------------------------------

    template <typename StateType>
    class State {
       public:
        explicit State() = default;
        State(const State<StateType>& state) = default;
        State(const StateType& state, COMPLEX c = COMPLEX(1, 0)) {
            if (!state.is_empty()) {
                state_vec_.emplace_back(c);
                state_components_.insert(std::shared_ptr<StateType>(new StateType(state)));
            }
        }

        State(const StateType& state, const BasisType<StateType>& basis) {
            for (auto st : basis) {
                state_components_.insert(std::shared_ptr<StateType>(new StateType(*st)));

                if (*st == state) {
                    state_vec_.emplace_back(1, 0);
                } else {
                    state_vec_.emplace_back(0, 0);
                }
            }
        }

        State<StateType> operator*(const COMPLEX& c) const {
            State<StateType> res(*this);

            size_t index = 0;

            for (auto st : res.state_components_) {
                res.state_vec_[index] *= c;

                index++;
            }

            return res;
        }

        // Работает не типично. Возвращает other, если this не пустое
        State<StateType> operator*(const State<StateType>& other) const {
            return (this->is_empty() ? State<StateType>() : other);
        }

        bool is_empty() const { return state_vec_.size() == 0; }

        void operator*=(const COMPLEX& c) {
            for (auto& st : this->state_vec_) {
                st *= c;
            }
        }

        void normalize() { vector_normalize(this->state_vec_); }

        void operator+=(const State<StateType>& st) {
            for (auto component : st.get_state_components()) {
                if (!is_in_state(*component)) {
                    this->insert(*component, st[st.get_index(*component)]);
                } else {
                    (*this)[this->get_index(*component)] += st[st.get_index(*component)];
                }
            }
        }

        void operator-=(const State<StateType>& st) {
            for (auto component : st.get_state_components()) {
                if (!is_in_state(*component)) {
                    this->insert(*component, -st[st.get_index(*component)]);
                } else {
                    (*this)[this->get_index(*component)] -= st[st.get_index(*component)];
                }
            }
        }

        bool is_in_state(const StateType& state) {
            for (auto st : state_components_) {
                if (*st == state) {
                    return true;
                }
            }

            return false;
        }

        COMPLEX& operator[](size_t index) { return state_vec_[index]; }
        COMPLEX operator[](size_t index) const { return state_vec_[index]; }
        COMPLEX& operator[](const StateType& st) { return state_vec_[this->get_index(st)]; }
        COMPLEX operator[](const StateType& st) const { return state_vec_[this->get_index(st)]; }
        std::shared_ptr<StateType> operator()(size_t index) const {
            return get_state_from_basis(this->state_components_, index);
        }
        size_t size() const { return state_vec_.size(); }

        size_t get_index(const StateType& state) const {
            size_t index = 0;
            for (auto st : state_components_) {
                if (*st == state) {
                    return index;
                }

                index++;
            }

            return index;
        }

        void insert(const StateType& state, const COMPLEX& amplitude = COMPLEX(0, 0)) {
            if (!is_in_state(state)) {
                state_components_.insert(std::shared_ptr<StateType>(new StateType(state)));
                state_vec_.insert(state_vec_.begin() + this->get_index(state), amplitude);
            }
        }

        void set_state_components(const BasisType<StateType>& st) { state_components_ = st; }
        void set_vector(const std::vector<COMPLEX>& v) { state_vec_ = v; }
        BasisType<StateType> get_state_components() const { return state_components_; }
        BasisType<StateType> state_components() const { return state_components_; }
        std::vector<COMPLEX> get_vector() const { return state_vec_; }
        std::vector<COMPLEX> vector() const { return state_vec_; }

        std::string to_string() const {
            std::string res;

            size_t index = 0;
            for (auto st : state_components_) {
                res += "(" + std::to_string(state_vec_[index].real()) + " + " +
                       std::to_string(state_vec_[index].imag()) + "j)";
                index++;

                res += " * ";
                res += st->to_string();

                if (index != state_components_.size()) {
                    res += " + ";
                }
            }

            return res;
        }

        State<StateType> fit_to_basis(const BasisType<StateType>& basis) const {
            State<StateType> res;

            //res.state_vec_ = std::vector<COMPLEX>(basis.size(), 0);
            res.set_vector(std::vector<COMPLEX>(basis.size(), 0));
            //res.state_components_ = basis;
            res.set_state_components(basis);

            size_t index = 0;
            for (auto state: basis) {
                size_t my_index = 0;
                for (auto my_state: this->state_components_) {
                    if ((*state) == (*my_state)) {
                        res[index] = this->state_vec_[my_index];
                        break;
                    }

                    my_index++;
                }

                index++;
            }

            return res;
        }

        State<Basis_State> fit_to_basis_state(const BasisType<Basis_State>& basis) const {
            State<Basis_State> res;

            //res.state_vec_ = std::vector<COMPLEX>(basis.size(), 0);
            res.set_vector(std::vector<COMPLEX>(basis.size(), 0));
            //res.state_components_ = basis;
            res.set_state_components(basis);

            size_t index = 0;
            for (auto state : basis) {
                size_t my_index = 0;
                for (auto my_state : this->state_components_) {
                    if ((*state).Basis_State::operator==(Basis_State(*my_state))) {
                        res[index] = this->state_vec_[my_index];
                        break;
                    }

                    my_index++;
                }

                index++;
            }

            return res;
        }

        // Полное копирование. Память выделяется ещё раз
        State<StateType> copy() const {
            State<StateType> res;
            for (auto st : this->state_components_) {
                res.insert(*st, (*this)[*st]);
            }

            return res;
        }

        void clear() {
            state_vec_.resize(0);

            state_components_.clear();
        }

       private:
        std::vector<COMPLEX> state_vec_;
        BasisType<StateType> state_components_;
    };

}  // namespace QComputations

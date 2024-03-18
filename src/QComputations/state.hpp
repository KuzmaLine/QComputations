    #pragma once
    #include <iostream>
    #include <vector>
    #include <string>
    #include <complex>
    #include <cassert>
    #include <set>
    #include "cavity_state.hpp"
    #include "matrix.hpp"
    #include "big_uint.hpp"
    #include <algorithm>

    namespace QComputations {

    namespace {
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

    const std::string PHOTONS_STR = "$N";
    const std::string ATOMS_STR = "$M";
    const std::string FREQ_STR = "$W";
    const std::string DEL_STR = "$!";

    /*
    Инструкция к строковому формату.
    $N - место задания фотонов
    $M - место задания атомов
    $W - место отображения частот перехода с 1 уровня на другой, если QConfig::instance().is_freq_display() == true
    $! - разделитель, между состояниями фотонов и атомами (смысл см. в примерах)

    Пример:
    Для формата |$N>$W$!|$M> примером состояние будет
    |0>[0-1]|1>[1-2]|2>[0-2]|0112>, то есть,
    0 фотонов для переход с 0 уровня на 1
    1 -''- с 1 на 2
    2 -''- с 0 на 2
    и состояния атомов 0112

    $W можно опустить, если QConfig::instance().is_freq_display() == false

    В последствии метод to_string() числа переходов с уровня на уровень переделает их в нижние индексы.
    Метод find_states_in_string() - будет согласно формату искать в строке состояния.
    */

    class Basis_State {
        public:
            // инициализация пустого состояния
            explicit Basis_State() = default;
            // groups_count делит кудиты на равные по размеру группы
            explicit Basis_State(size_t qudits_count, ValType max_val = 1, size_t groups_count = 1);
            // добавление поддержки для разных кудитов
            explicit Basis_State(size_t qudits_count, const std::vector<ValType>& max_vals,
                                size_t groups_count = 1);
            explicit Basis_State(size_t qudits_count, ValType max_val, const std::vector<size_t>& groups): qudits_(qudits_count, 0),
                                                                                                               max_vals_(qudits_count, max_val),
                                                                                                               groups_(groups) {}
            // инициализация значений
            explicit Basis_State(const std::vector<ValType>& qudits, ValType max_vals = 1, size_t groups_count = 1);
            // инициализация значений с поддержкой разных кудитов
            explicit Basis_State(const std::vector<ValType>& qudits, const std::vector<ValType>& max_vals,
                                size_t groups_count = 1): qudits_(qudits), max_vals_(max_vals), groups_(split_to_groups(qudits.size(), groups_count)) {}
            // инициализация значений с поддержкой разных кудитов + разных групп
            explicit Basis_State(const std::vector<ValType>& qudits,  const std::vector<ValType>& max_vals,
                                const std::vector<size_t>& groups_sizes);

            void set_qudit(ValType val, size_t qudit_index, size_t group_id = 0) { assert(val <= max_vals_[qudit_index]);
                                        qudits_[this->get_group_start(group_id) + qudit_index] = val;}
            void set_atom(ValType val, size_t atom_index, size_t group_id) { this->set_qudit(val, atom_index  + 1, group_id); }
            ValType get_qudit(size_t qudit_index, size_t group_id = 0) const { return qudits_[this->get_group_start(group_id) + qudit_index]; }
            void append_qudit(ValType init_val = 0, ValType max_val = 1) { groups_.emplace_back(qudits_.size());
                                                                        qudits_.emplace_back(init_val);
                                                                        max_vals_.emplace_back(max_val);}
            bool is_empty() const { return qudits_.size() == 0;}

            //Basis_State operator*(const COMPLEX& c) const { auto res = *this; res.set_coef(this->get_coef() * c); return res;}
            std::string get_info() const { return info_; }
            void set_info(const std::string& str) { info_ = str; }


            std::vector<size_t> get_groups() const { return groups_; }
            size_t get_group_start(size_t group_id) const { return ((group_id == 0) ? 0 : groups_[group_id - 1] + 1);}
            size_t get_group_end(size_t group_id) const { return groups_[group_id]; }
            size_t get_groups_count() const { return groups_.size(); }
            size_t get_group_size(size_t group_id) const { return this->get_group_end(group_id) - this->get_group_start(group_id) + 1; }
            Basis_State get_group(size_t group_id) const;

            std::string to_string() const;
            bool operator==(const Basis_State& other) const { assert(max_vals_ == other.max_vals_ and groups_ == other.groups_); return qudits_ == other.qudits_; }
            bool operator<(const Basis_State& other) const { return this->to_string() > other.to_string(); }
            
            void set_max_val(ValType val, size_t qudit_index, size_t group_id = 0) { max_vals_[this->get_group_start(group_id) + qudit_index] = val; }
            ValType get_max_val(size_t qudit_index, size_t group_id = 0) const { return max_vals_[this->get_group_start(group_id) + qudit_index]; }
            std::vector<ValType> max_vals() const { return max_vals_; }

            size_t get_index(const std::set<Basis_State>& basis) const;

            //COMPLEX get_coef() const { return coef_; }
            //void set_coef(COMPLEX coef) { coef_ = coef; }

            void clear() { qudits_.resize(0); max_vals_.resize(0); groups_.resize(0); }
        protected:
            std::string info_;
            //COMPLEX coef_ = COMPLEX(1, 0);
            std::vector<ValType> qudits_;
            std::vector<ValType> max_vals_;
            std::vector<size_t> groups_;
    };

    class CHE_State: public Basis_State {
        using E_LEVEL = int;
        //using CavityId = size_t;
        using AtomId = size_t;

        public:
            CHE_State() = default;
            CHE_State(const Basis_State& base): Basis_State(base), x_size_(base.get_groups_count()), y_size_(1), z_size_(1), neighbours_(update_neighbours(x_size_, y_size_, z_size_)) {}
            //CHE_State(size_t x_size = 1, size_t y_size = 1, size_t z_size = 1);
            CHE_State(const CHE_State& state) = default;
            CHE_State(const std::vector<size_t>& grid_config);
            //explicit CHE_State(const std::string&, const std::string& format = QConfig::instance().state_format(),
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
            void set_waveguide(const Matrix<std::pair<double, double>>& A) {waveguides_ = A;}
            void set_waveguide(size_t from_cavity_id, size_t to_cavity_id, double amplitude, double length = QConfig::instance().waveguides_length());
            // set entire state in cavity with id = id
            void set_state(CavityId id, const CHE_State& state);

            // add cavity to grid (Don't safe, be careful)
            CHE_State add_state(const CHE_State& state) const;

            size_t cavities_count() const { return groups_.size(); }
            size_t cavity_atoms_count(CavityId id) const { return this->get_group_end(id) - this->get_group_start(id); }
            //size_t cavity_size(CavityId id) const { return cavity_atoms_count(id) + 1; }
            size_t m(CavityId id) const { return cavity_atoms_count(id); }
            ValType n(CavityId id) const { return qudits_[get_group_start(id)]; }
            void set_n(ValType n, CavityId id) { qudits_[get_group_start(id)] = n; }

            // Return state vector from cavity
            CHE_State get_state_in_cavity(CavityId cavity_id) const { return CHE_State(this->get_group(cavity_id)); }
            CHE_State operator[](CavityId cavity_id) const { return CHE_State(this->get_group(cavity_id)); }

            CavityId get_index_of_cavity(size_t x, size_t y = 0, size_t z = 0) const { return z * y_size_ * x_size_ + y * x_size_ + x; }
            
            // Рудимент
            // size_t get_index() const;
            // void set_term(size_t atom_index, double term, CavityId cavity_id) { grid_states_[cavity_id].set_term(atom_index, term); }
            // double get_term(size_t atom_index, CavityId cavity_id) const { return grid_states_[cavity_id].get_term(atom_index); }

            // Get index of state in basis
            size_t get_index(const std::set<CHE_State>& basis) const;
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
            void from_uint(const BigUInt& state_num);

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

    class EXC_State: public Basis_State {
        
    };

    // ------------------------------ State ---------------------------------

    template<typename StateType>
    class State {
        public:
            explicit State() = default;
            State(const State<StateType>& state) = default;
            State(const StateType& state) {
                if (!state.is_empty()) {
                    state_vec_.emplace_back(1, 0);
                    state_components_.insert(state);
                }
            }

            explicit State(const std::string& state_string, ValType max_val = 1);
            explicit State(const std::string& state_string, const std::vector<ValType>& max_vals);

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

            State<StateType> operator*(const COMPLEX& c) const {
                State<StateType> res(*this);

                size_t index = 0;

                for (StateType st: res.state_components_) {
                    res.state_vec_[index] *= c;
                    
                    index++;
                }

                return res;
            }

            void operator+=(const State<StateType>& st) {
                for (const auto& component: st.get_state_components()) {
                    if (!is_in_state(component)) {
                        this->insert(component, st[st.get_index(component)]);
                    } else {
                        (*this)[this->get_index(component)] += st[st.get_index(component)];
                    }
                }
            }

            bool is_in_state(const StateType& state) {
                auto it = std::find(state_components_.begin(), state_components_.end(), state);

                return it != state_components_.end();
            }

            COMPLEX& operator[](size_t index) { return state_vec_[index];}
            COMPLEX operator[](size_t index) const { return state_vec_[index];}
            size_t size() const { return state_vec_.size(); }

            size_t get_index(const StateType& state) const {
                auto it = std::find(state_components_.begin(), state_components_.end(), state);
                return std::distance(state_components_.begin(), it);
            }

            void insert(const StateType& state, const COMPLEX& amplitude = COMPLEX(0, 0)) {
                if (std::find(state_components_.begin(), state_components_.end(), state) == state_components_.end()) {
                    state_components_.insert(state);
                    state_vec_.insert(state_vec_.begin() + this->get_index(state), amplitude);
                }
            }

            void set_state_components(const std::set<StateType>& st) { state_components_ = st; }
            void set_vector(const std::vector<COMPLEX>& v) { state_vec_ = v; }
            std::set<StateType> get_state_components() const { return state_components_; }
            std::vector<COMPLEX> get_vector() const { return state_vec_;}

            std::string to_string() const {
                std::string res;

                size_t index = 0;
                for (const auto& st: state_components_) {
                    res += "(" + std::to_string(state_vec_[index].real()) + " + " + std::to_string(state_vec_[index].imag()) + "j)";
                    index++;

                    res += " * ";
                    res += st.to_string();

                    if (index != state_components_.size()) {
                        res += " + ";
                    }
                }

                return res;
            }

            State<Basis_State> fit_to_basis(const std::set<Basis_State>& basis) const {
                State<Basis_State> res;

                //res.state_vec_ = std::vector<COMPLEX>(basis.size(), 0);
                res.set_vector(std::vector<COMPLEX>(basis.size(), 0));
                //res.state_components_ = basis;
                res.set_state_components(basis);

                size_t index = 0;
                for (const auto& state: basis) {
                    size_t my_index = 0;
                    for (const auto& my_state: this->state_components_) {
                        if (state == Basis_State(my_state)) {
                            res[index] = this->state_vec_[my_index];
                            break;
                        }

                        my_index++;
                    }

                    index++;
                }

                return res;
            }
        private:
            std::vector<COMPLEX> state_vec_;
            std::set<StateType> state_components_;
    };

    } // namespace QComputations
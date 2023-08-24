#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <complex>
#include <cassert>
#include "cavity_state.hpp"
#include "matrix.hpp"
#include "big_uint.hpp"

class State {
    using COMPLEX = std::complex<double>;
    using matrix = std::vector<std::vector<COMPLEX>>;
    using E_LEVEL = int;
    using CavityId = size_t;
    using AtomId = size_t;

    public:
        State(size_t x_size = 1, size_t y_size = 1, size_t z_size = 1);
        State(const Cavity_State& state);
        State(const State& state) = default;
        State(const std::vector<size_t>& grid_config);
        explicit State(const std::string&, const std::string& format = "|N;M>");

        size_t x_size() const { return x_size_; }
        size_t y_size() const { return y_size_; }
        size_t z_size() const { return z_size_; }

        size_t max_N() const { return max_N_; }
        void set_max_N(size_t N) { max_N_ = N; }
        size_t min_N() const { return min_N_; }
        void set_min_N(size_t N) { min_N_ = N; }
        size_t n(CavityId id = 0) const { return grid_states_[id].n(); } // TMP
        void set_n(size_t n, CavityId id = 0) { grid_states_[id].set_n(n); } // TMP
        size_t m(CavityId id) const { return grid_states_[id].m(); }

        void reshape(size_t x_size, size_t y_size, size_t z_size) {
            assert(x_size * y_size * z_size == grid_states_.size());

            x_size_ = x_size;
            y_size_ = y_size;
            z_size_ = z_size;

            gamma_ = Matrix<COMPLEX>(C_STYLE, grid_states_.size(), grid_states_.size(), 0);
            is_init_gamma_ = false;
        }

        void set_gamma(COMPLEX gamma) { gamma_ = Matrix<COMPLEX>(C_STYLE, grid_states_.size(),
                                                                 grid_states_.size(),
                                                                 gamma);
                                                                 is_init_gamma_ = true; }
        void set_gamma(const Matrix<COMPLEX>& A) {gamma_ = A;
                                                  is_init_gamma_ = true;}
        void set_gamma(size_t from_id, size_t to_id, COMPLEX gamma);

        E_LEVEL get_qubit(CavityId pol_id, AtomId atom_index) const { return grid_states_[pol_id].get_qubit(atom_index); }

        void set_qubit(CavityId pol_id, AtomId atom_index, E_LEVEL level) {
            grid_states_[pol_id].set_qubit(atom_index, level);
        }

        void set_state(CavityId id, const Cavity_State& state);
        State add_state(const Cavity_State& state) const;

        std::string to_string() const;


        size_t cavities_count() const { return grid_states_.size(); }
        size_t cavity_atoms_count(CavityId id) const { return grid_states_.at(id).m(); }
        size_t cavity_max_size(CavityId id) const { return grid_states_[id].variants_of_state_count(max_N_); }

        bool operator==(const State& other) const { return grid_states_ == other.grid_states_; }
        bool operator<(const State& other) const { return this->to_string() > other.to_string(); }

        Cavity_State get_state_in_pol(CavityId pol_id) const { return grid_states_[pol_id]; }
        Cavity_State operator[](CavityId pol_id) const { return grid_states_[pol_id]; }
        
        size_t get_index() const;
        size_t get_index(const std::set<State>& basis) const;
        size_t get_max_size() const;
        size_t get_energy() const;
    
        std::set<CavityId> get_cavities_with_leak() const { return leak_cavities_; }
        COMPLEX get_leak_gamma(CavityId id) const { return gamma_leak_cavities_[id]; }
        std::set<CavityId> get_cavities_with_gain() const { return gain_cavities_; }
        COMPLEX get_gain_gamma(CavityId id) const { return gamma_gain_cavities_[id]; }

        void set_leak_for_cavity(CavityId id, COMPLEX gamma) { leak_cavities_.insert(id);
                                                               gamma_leak_cavities_[id] = gamma;}
        void set_gain_for_cavity(CavityId id, COMPLEX gamma) { gain_cavities_.insert(id);
                                                               gamma_gain_cavities_[id] = gamma;}

        Matrix<COMPLEX> get_gamma() const { return gamma_; }
        COMPLEX get_gamma(CavityId from_id, CavityId to_id) const { return gamma_[from_id][to_id]; }
        std::set<CavityId> get_cavities_with_atoms() const { return cavities_with_atoms_; }

        size_t hash() const;
    private:
        size_t max_N_;
        size_t min_N_;
        size_t x_size_;
        size_t y_size_;
        size_t z_size_;

        std::set<CavityId> cavities_with_atoms_;
        std::vector<Cavity_State> grid_states_;
        Matrix<COMPLEX> gamma_;
        std::vector<COMPLEX> gamma_leak_cavities_;
        std::vector<COMPLEX> gamma_gain_cavities_;
        std::set<CavityId> leak_cavities_;
        std::set<CavityId> gain_cavities_;
        bool is_init_gamma_ = false;
};

/*
|00;00>   |0>|0>|0>|0>
|00;01>   |0>|0>|0>|1>
|00;10>   |0>|0>|1>|0>
|00;11>   |0>|0>|1>|1>
|01;00>   |0>|1>|0>|0>
|01;01>   |0>|1>|0>|1>
...

*/

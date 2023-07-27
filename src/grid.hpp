#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <complex>
#include <cassert>
#include "state.hpp"
#include "matrix.hpp"

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
        explicit State(const std::string&, const std::string& format = "|N;M>");

        size_t x_size() const { return x_size_; }
        size_t y_size() const { return y_size_; }
        size_t z_size() const { return z_size_; }

        size_t n() const { return grid_states_[0].n(); }
        void set_n(size_t n) { grid_states_[0].set_n(n); }

        void reshape(size_t x_size, size_t y_size, size_t z_size) {
            assert(x_size * y_size * z_size == grid_states_.size());

            x_size_ = x_size;
            y_size_ = y_size;
            z_size_ = z_size;

            gamma_ = Matrix<COMPLEX>(grid_states_.size(), grid_states_.size(), 0);
            is_init_gamma_ = false;
        }

        void set_gamma(COMPLEX gamma) { gamma_ = Matrix<COMPLEX>(grid_states_.size(),
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

        std::string to_string() const;

        size_t amount_of_states() const { return grid_states_.size(); }
        size_t pol_size(CavityId pol_id) const { return grid_states_.at(pol_id).m(); }

        bool operator==(const State& other) const { return grid_states_ == other.grid_states_; }
        bool operator<(const State& other) const { return this->to_string() > other.to_string(); }

        Cavity_State get_state_in_pol(CavityId pol_id) const { return grid_states_[pol_id]; }
        Cavity_State operator[](CavityId pol_id) const { return grid_states_[pol_id]; }
        
        size_t get_index() const;
        size_t get_index(const std::set<State>& basis) const;

        Matrix<COMPLEX> get_gamma() const { return gamma_; };
        std::set<CavityId> get_pols_with_atoms() const { return cavities_with_atoms_; }

        size_t hash() const;
    private:
        size_t N_;
        size_t x_size_;
        size_t y_size_;
        size_t z_size_;

        std::set<CavityId> cavities_with_atoms_;
        std::vector<Cavity_State> grid_states_;
        Matrix<COMPLEX> gamma_;
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

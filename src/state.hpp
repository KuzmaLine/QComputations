#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <complex>
#include <cassert>
#include "cavity_state.hpp"
#include "matrix.hpp"
#include "big_uint.hpp"

namespace QComputations {

namespace {
    std::complex<double> gamma(double length, double w_ph) {
        return std::exp(std::complex<double>(0, -1) * length * w_ph / QConfig::instance().h());
    }
}

class State {
    using COMPLEX = std::complex<double>;
    using E_LEVEL = int;
    using CavityId = size_t;
    using AtomId = size_t;

    public:
        State(size_t x_size = 1, size_t y_size = 1, size_t z_size = 1);
        State(const Cavity_State& state);
        State(const State& state) = default;
        State(const std::vector<size_t>& grid_config, E_LEVEL e_levels_count = 2);
        explicit State(const std::string&, const std::string& format = "|N;M>");

        size_t x_size() const { return x_size_; }
        size_t y_size() const { return y_size_; }
        size_t z_size() const { return z_size_; }

        size_t max_N() const { return max_N_; }  // Get maximum considering energy on grid
        void set_max_N(size_t N) { max_N_ = N; } // Set maximum considering energy on grid
        size_t min_N() const { return min_N_; }  // Get minimum considering energy on grid
        void set_min_N(size_t N) { min_N_ = N; } // Set minimum considering energy on grid

        size_t n(CavityId id = 0, E_LEVEL e_from = 0, E_LEVEL e_to = 1) const { return grid_states_[id].n(e_from, e_to); } // get amount of photons in cavity with id = id
        void set_n(size_t n, CavityId id = 0, E_LEVEL e_from = 0, E_LEVEL e_to = 1) { grid_states_[id].set_n(n, e_from, e_to); } // set n photons in cavity with id = id
        size_t m(CavityId id) const { return grid_states_[id].m(); } // get amount of atoms in cavity with id = id

        // change grid shapes
        void reshape(size_t x_size, size_t y_size, size_t z_size);

        // TMP realizations
        void set_waveguide(double length) { waveguides_length_ = Matrix<double>(C_STYLE, grid_states_.size(),
                                                                 grid_states_.size(),
                                                                 length); }
        void set_waveguide(const Matrix<double>& A) {waveguides_length_ = A;}
        void set_waveguide(size_t from_cavity_id, size_t to_cavity_id, double waveguide_length) { waveguides_length_[from_cavity_id][to_cavity_id] = waveguide_length; }

        // get qubit in cavity with atom_index
        E_LEVEL get_qubit(CavityId pol_id, AtomId atom_index) const { return grid_states_[pol_id].get_qubit(atom_index); }

        // set qubit in cavity with atom_index
        void set_qubit(CavityId pol_id, AtomId atom_index, E_LEVEL level) {
            grid_states_[pol_id].set_qubit(atom_index, level);
        }

        // set entire state in cavity with id = id
        void set_state(CavityId id, const Cavity_State& state);

        // add cavity to grid (Don't safe, be careful)
        State add_state(const Cavity_State& state) const;

        std::string to_string() const;

        size_t cavities_count() const { return grid_states_.size(); }
        size_t cavity_atoms_count(CavityId id) const { return grid_states_.at(id).m(); }
        
        // Рудимент
        size_t cavity_max_size(CavityId id) const { return grid_states_[id].variants_of_state_count(max_N_); }

        bool operator==(const State& other) const { return grid_states_ == other.grid_states_; }
        bool operator<(const State& other) const { return this->to_string() > other.to_string(); }

        // Return state vector from cavity
        Cavity_State get_state_in_pol(CavityId pol_id) const { return grid_states_[pol_id]; }
        Cavity_State operator[](CavityId pol_id) const { return grid_states_[pol_id]; }
        
        // Рудимент
        size_t get_index() const;

        // Get index of state in basis
        size_t get_index(const std::set<State>& basis) const;
        size_t get_max_size() const;

        // return energy in state (photons + atoms in state one)
        size_t get_grid_energy() const;

        size_t get_energy(CavityId cavity_id) const;
        size_t get_max_energy(CavityId cavity_id) const { return grid_states_[cavity_id].get_max_energy(); }
    
        std::set<CavityId> get_cavities_with_leak() const { return leak_cavities_; }
        COMPLEX get_leak_gamma(CavityId id) const { return gamma_leak_cavities_[id]; }
        std::set<CavityId> get_cavities_with_gain() const { return gain_cavities_; }
        COMPLEX get_gain_gamma(CavityId id) const { return gamma_gain_cavities_[id]; }

        void set_leak_for_cavity(CavityId id, COMPLEX gamma) { leak_cavities_.insert(id);
                                                               gamma_leak_cavities_[id] = gamma;}
        void set_gain_for_cavity(CavityId id, COMPLEX gamma) { gain_cavities_.insert(id);
                                                               gamma_gain_cavities_[id] = gamma;}

        // Matrix<COMPLEX> get_gamma() const { return gamma_; }
        COMPLEX get_gamma(CavityId from_id, CavityId to_id, E_LEVEL e_from = 0, E_LEVEL e_to = 1) const {
            return gamma(waveguides_length_[from_id][to_id], grid_states_[from_id].w_ph(e_from, e_to));
        }
        std::set<CavityId> get_cavities_with_atoms() const { return cavities_with_atoms_; }

        // Like a hash
        BigUInt to_uint() const;

        // Change state to with BigUint = state_num
        void from_uint(const BigUInt& state_num);

        size_t e_levels_count() const { return e_levels_count_; }

        std::vector<CavityId> get_neighbours(CavityId cavity_id) const { return neighbours_[cavity_id]; }

        size_t hash() const;
    private:
        size_t max_N_;
        size_t min_N_;
        size_t x_size_;
        size_t y_size_;
        size_t z_size_;
        size_t e_levels_count_;

        std::set<CavityId> cavities_with_atoms_;
        std::vector<Cavity_State> grid_states_;
        Matrix<double> waveguides_length_;
        std::vector<std::vector<CavityId>> neighbours_;
        std::vector<COMPLEX> gamma_leak_cavities_;
        std::vector<COMPLEX> gamma_gain_cavities_;
        std::set<CavityId> leak_cavities_;
        std::set<CavityId> gain_cavities_;
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

} // namespace QComputations
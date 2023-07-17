#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <complex>
#include "basis.hpp"

class Grid {
    using COMPLEX = std::complex<double>;
    using matrix = std::vector<std::vector<COMPLEX>>;
    using E_LEVEL = int;
    using PolId = int;
    using AtomId = int;

    public:
        Grid(size_t x_size = 1, size_t y_size = 1, size_t z_size = 1);
        explicit Grid(const std::vector<Basis>& state);
        explicit Grid(const std::string&);
        Grid(const Grid&) = default;

        size_t x_size() const { return x_size_; }
        size_t y_size() const { return y_size_; }
        size_t z_size() const { return z_size_; }

        void set_size(size_t x_size, size_t y_size, size_t z_size) {
            if (x_size * y_size * z_size != grid_states_.size()) {
                std::cout << "SIZE ERROR\n";
                exit(1);
            }

            x_size_ = x_size;
            y_size_ = y_size;
            z_size_ = z_size;
        }

        E_LEVEL get_qubit(PolId pol_id, AtomId atom_index) const { return grid_states_[pol_id].get_qubit(atom_index); }

        void set_qubit(PolId pol_id, AtomId atom_index, E_LEVEL level) {
            grid_states_[pol_id].set_qubit(atom_index, level);
        }

        std::string to_string() const;

        size_t amount_of_states() const { return grid_states_.size(); }
        size_t pol_size(PolId pol_id) const { return grid_states_.at(pol_id).size(); }

        bool operator==(const Grid& other) const { return grid_states_ == other.grid_states_ and x_size_ == other.x_size_ and y_size_ == other.y_size_ and z_size_ == other.z_size_; }

        Basis get_state_in_pol(PolId pol_id) { return grid_states_[pol_id]; }
        matrix get_full_matrix() const;

        std::set<PolId> pols_with_atoms() const { return pols_with_atoms_; }
        size_t hash() const;
    private:
        void init_gamma();

        size_t x_size_;
        size_t y_size_;
        size_t z_size_;
        std::vector<Basis> grid_states_;
        std::set<PolId> pols_with_atoms_;
        std::vector<std::vector<COMPLEX>> gamma;
};

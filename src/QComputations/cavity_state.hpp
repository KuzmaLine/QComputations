#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <complex>
#include <map>
#include <set>
#include <cassert>
#include "matrix.hpp"

namespace QComputations {

// Состояния в полостях
class Cavity_State {
    using COMPLEX = std::complex<double>;
    using vec_complex = std::vector<COMPLEX>;
    using E_LEVEL = int;
    using vec_levels = std::vector<E_LEVEL>;
    using AtomId = size_t;

    public:
        explicit Cavity_State() {};

        // n - photons, m - num of atoms, state - vector of state in 2 numerical system
        Cavity_State(size_t n, size_t m = 0, E_LEVEL e_levels_count = QConfig::instance().E_LEVELS_COUNT());
        Cavity_State(const Matrix<size_t>& n, const std::vector<E_LEVEL>& state, E_LEVEL e_levels_count = QConfig::instance().E_LEVELS_COUNT());
        Cavity_State(size_t n, const std::vector<E_LEVEL>& state, E_LEVEL e_levels_count = QConfig::instance().E_LEVELS_COUNT());

        //format |n>|m> 
        // (!!!) NEED CHANGE TO |n;m>
        explicit Cavity_State(const std::string&, E_LEVEL e_levels_count = QConfig::instance().E_LEVELS_COUNT());
        Cavity_State(const Cavity_State&) = default;

        size_t n(E_LEVEL level_from = 0, E_LEVEL level_to = 1) const { return n_[level_from][level_to]; }

        void set_n(const Matrix<size_t>& n);
        void set_n(size_t new_n, E_LEVEL level_from = 0, E_LEVEL level_to = 1) {
            assert(level_from != level_to);
            if (level_from > level_to) {
                auto tmp = level_from;
                level_from = level_to;
                level_to = tmp;
            }

            n_[level_from][level_to] = new_n;
        }

        // Num of atoms with state = 1
        size_t up_count() const;
        E_LEVEL get_qubit(AtomId atom_index) const { return state_[atom_index]; }
        void set_qubit(AtomId atom_index, E_LEVEL level) {
            state_[atom_index] = level;
        }

        // (!!!) NEED CHANGE TO |n;m>
        std::string to_string() const;

        size_t m() const { return state_.size(); }
        size_t size() const { return state_.size(); }

        // Return count of every possible states of cavity with max energy = n_max
        size_t variants_of_state_count(size_t n_max) const;
        std::vector<E_LEVEL> get_atoms_state() const { return state_; }

        // Return n_ * state_ (converted from 2 numerical system)
        size_t get_index() const;
        size_t get_index(const std::set<Cavity_State>& basis) const;
        bool is_in_basis(const std::set<Cavity_State>& basis) const;

        // Return state_ (converted from 2 numerical system)
        size_t get_atoms_index() const;

        // return sum of n_ and ones in state_
        size_t get_energy() const;
        
        double w_at(E_LEVEL e_level = 1) const { return w_at_[e_level]; }
        void set_w_at(double w_at, E_LEVEL e_level = 1) { w_at_[e_level] = w_at; }

        double w_ph(E_LEVEL level_from = 0, E_LEVEL level_to = 1) const {
            assert(level_from != level_to);
            if (level_from > level_to) {
                auto tmp = level_from;
                level_from = level_to;
                level_to = tmp;
            }

            return w_ph_[level_from][level_to];
        }

        void set_w_ph(double w_ph, E_LEVEL level_from = 0, E_LEVEL level_to = 1) {
            assert(level_from != level_to);
            if (level_from > level_to) {
                auto tmp = level_from;
                level_from = level_to;
                level_to = tmp;
            }

            w_ph_[level_from][level_to] = w_ph;
        }

        size_t get_max_energy() const { return max_energy_; }

        bool operator==(const Cavity_State& other) const { return state_ == other.state_ and n_ == other.n_; }
        //bool operator<(const Cavity_State& other) const { return n_ > other.n_ or get_index_from_state(state_) < get_index_from_state(other.state_); }
        bool operator<(const Cavity_State& other) const { return this->to_string() > other.to_string(); }
        size_t hash() const;
    private:
        E_LEVEL e_levels_count_ = QConfig::instance().E_LEVELS_COUNT();
        Matrix<size_t> n_;
        std::vector<double> w_at_;
        Matrix<double> w_ph_;
        std::vector<E_LEVEL> state_;

        size_t max_energy_;
};

} // namespace QComputations
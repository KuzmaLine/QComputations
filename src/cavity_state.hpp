#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <complex>
#include <map>
#include <set>

class Cavity_State {
    using COMPLEX = std::complex<double>;
    using vec_complex = std::vector<COMPLEX>;
    using E_LEVEL = int;
    using vec_levels = std::vector<E_LEVEL>;
    using CavityId = size_t;
    using AtomId = size_t;

    public:
        explicit Cavity_State() {};
        Cavity_State(size_t n, size_t m = 0, size_t state = 0);
        Cavity_State(size_t n, const std::vector<E_LEVEL>& state);
        explicit Cavity_State(const std::string&);
        Cavity_State(const Cavity_State&) = default;

        size_t n() const { return n_; }

        void set_n(size_t new_n) { n_ = new_n; }

        size_t up_count() const;
        E_LEVEL get_qubit(AtomId atom_index) const { return state_[atom_index]; }

        void set_qubit(AtomId atom_index, E_LEVEL level) {
            state_[atom_index] = level;
        }

        std::string to_string() const;

        size_t m() const { return state_.size(); }
        size_t variants_of_state_count(size_t n_max) const { return n_max * std::pow(2, this->m());}
        std::vector<E_LEVEL> get_atoms_state() const { return state_; }

        size_t get_index() const;
        size_t get_index(const std::set<Cavity_State>& basis) const;
        bool is_in_basis(const std::set<Cavity_State>& basis) const;
        size_t get_atoms_index() const;
        size_t get_energy() const;

        bool operator==(const Cavity_State& other) const { return state_ == other.state_ and n_ == other.n_; }
        //bool operator<(const Cavity_State& other) const { return n_ > other.n_ or get_index_from_state(state_) < get_index_from_state(other.state_); }
        bool operator<(const Cavity_State& other) const { return this->to_string() > other.to_string(); }
        size_t hash() const;
    private:
        size_t n_;
        std::vector<E_LEVEL> state_;
};

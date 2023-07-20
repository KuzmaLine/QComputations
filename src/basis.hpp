#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <complex>
#include <map>
#include <set>

class State {
    using COMPLEX = std::complex<double>;
    using vec_complex = std::vector<COMPLEX>;
    using E_LEVEL = int;
    using vec_levels = std::vector<E_LEVEL>;
    using PolId = size_t;
    using AtomId = size_t;

    public:
        explicit State() {};
        explicit State(size_t n, size_t m, size_t state);
        explicit State(const std::vector<E_LEVEL>& state);
        explicit State(const std::string&);
        State(const State&) = default;

        size_t n() const { return n_; }

        void set_n(size_t new_n) { n_ = new_n; }

        E_LEVEL get_qubit(AtomId atom_index) const { return state_[atom_index]; }

        void set_qubit(AtomId atom_index, E_LEVEL level) {
            state_[atom_index] = level;
        }

        std::string to_string() const;

        size_t size() const { return state_.size(); }
        std::vector<E_LEVEL> get_vector_notaion() const { return state_; }
        std::vector<COMPLEX> get_vector_atoms_state() const { return vector_of_atoms_state_; };

        size_t get_index() const;

        bool operator==(const State& other) const { return state_ == other.state_ and n_ == other.n_; }
        //bool operator<(const State& other) const { return n_ > other.n_ or get_index_from_state(state_) < get_index_from_state(other.state_); }
        bool operator<(const State& other) const { return this->to_string() > other.to_string(); }
        size_t hash() const;
    private:
        size_t n_;
        std::vector<E_LEVEL> state_;
        std::vector<COMPLEX> vector_of_atoms_state_;
};

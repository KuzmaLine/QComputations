#pragma once
#include "additional_operators.hpp"
#include <iostream>
#include <unordered_map>
#include <queue>
#include <unordered_set>
#include <set>
#include "state.hpp"


namespace QComputations {
// DON'T TOUCH
// NEED REWORK

// Рудимент
class State_Graph {
    public:
        explicit State_Graph(const Cavity_State& init_state, bool with_loss_photons = false, bool GAIN_PHOTONS = false, size_t N = 1);
        void show() const;

        std::set<Cavity_State> get_basis() const { return basis_; }
    private:
        std::set<Cavity_State> basis_;
        std::unordered_map<Cavity_State, std::unordered_set<Cavity_State>> to_;
        std::unordered_map<Cavity_State, std::unordered_set<Cavity_State>> from_;
};

// Это тебе знать необязательно
class Quantum_Neural_Network {
    public:
        Quantum_Neural_Network(const State& init_grid, const Matrix<COMPLEX>& H);
        void show() const;
    private:
        std::set<State> basis_;
        std::unordered_map<BigUInt, std::unordered_set<BigUInt>> to_;
        std::unordered_map<BigUInt, std::unordered_set<BigUInt>> from_;
        Matrix<COMPLEX> H_;
};

} // namespace QComputations
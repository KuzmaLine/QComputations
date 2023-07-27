#pragma once
#include "additional_operators.hpp"
#include <iostream>
#include <unordered_map>
#include <queue>
#include <unordered_set>
#include <set>
#include "state.hpp"


class State_Graph {
    public:
        explicit State_Graph(const Cavity_State& init_state, bool with_loss_photons = false);
        void show() const;

        std::set<Cavity_State> get_basis() const { return basis_; }
    private:
        std::set<Cavity_State> basis_;
        std::unordered_map<Cavity_State, std::unordered_set<Cavity_State>> to_;
        std::unordered_map<Cavity_State, std::unordered_set<Cavity_State>> from_;
};

class Quantum_Neural_Network : public State_Graph {

};
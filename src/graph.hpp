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
        explicit State_Graph(const State& init_state, bool with_loss_photons = false);
        void show() const;

        std::set<State> get_basis() const { return basis_; }
    private:
        std::set<State> basis_;
        std::queue<State> state_queue_;
        std::unordered_map<State, std::unordered_set<State>> to_;
        std::unordered_map<State, std::unordered_set<State>> from_;
};

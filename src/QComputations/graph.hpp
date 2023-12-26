#pragma once
#include "additional_operators.hpp"
#include <iostream>
#include <unordered_map>
#include <queue>
#include <unordered_set>
#include <set>
#include "state.hpp"
#include "quantum_operators.hpp"


namespace QComputations {

class State_Graph {
    public:
        explicit State_Graph(const CHE_State& init_state);
        explicit State_Graph(const State& init_state,
                        std::function<Formule(const Basis_State&)> func,
                        std::function<Formule(const Basis_State&)> func_decoherence)
        void show() const;

        std::set<Basis_State> get_basis() const { return basis_; }
    private:
        std::set<Basis_State> basis_;
        //std::unordered_map<Basis_State, std::unordered_set<Basis_State>> to_;
        //std::unordered_map<Basis_State, std::unordered_set<Basis_State>> from_;
};

} // namespace QComputations
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

template<typename StateType>
bool is_in_basis(const std::set<StateType>& basis, const StateType& state) {
    return std::find(basis.begin(), basis.end(), state) != basis.end();
}

template<typename StateType>
class State_Graph {
    public:
        //explicit State_Graph(const CHE_State& init_state);
        explicit State_Graph(const State<StateType>& init_state,
                        const Operator<StateType>& A_op,
                        const std::vector<Operator<StateType>>& operator_decoherence = {});
        void show() const;

        std::set<StateType> get_basis() const { return basis_; }
    private:
        std::set<StateType> basis_;
        //std::set<CHE_State> ch_basis_;
        //std::unordered_map<Basis_State, std::unordered_set<Basis_State>> to_;
        //std::unordered_map<Basis_State, std::unordered_set<Basis_State>> from_;
};

template<typename StateType>
State_Graph<StateType>::State_Graph(const State<StateType>& init_state,
                         const Operator<StateType>& A_op,
                         const std::vector<Operator<StateType>>& operator_decoherence) {
    auto state_components = init_state.get_state_components();
    std::queue<StateType> state_queue;
    for (const auto& state: state_components) {
        basis_.insert(state);
        state_queue.push(state);
    }

    while (!state_queue.empty()) {
        for (const auto& b: basis_) {
            std::cout << b.to_string() << " ";
        }
        std::cout << std::endl;
        auto state = state_queue.front();
        state_queue.pop();
        std::cout << "RUN" << " " << State<StateType>(state).to_string() << std::endl;
        auto res = A_op.run(State<StateType>(state));
        std::cout << "RUNED\n";

        for (const auto& st: res.get_state_components()) {
            if (!is_in_basis<StateType>(basis_, st)) {
                basis_.insert(st);
                state_queue.push(st);
            }
        }

        for (const auto& op: operator_decoherence) {
            res = op.run(State<StateType>(state));

            for (const auto& st: res.get_state_components()) {
                if (!is_in_basis(basis_, st)) {
                    basis_.insert(st);
                    state_queue.push(st);
                }
            }
        }
    }
}

} // namespace QComputations
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
inline bool is_in_basis(const BasisType<StateType>& basis, std::shared_ptr<StateType> state) {
    // for (auto st: basis) {
    //     if ((*st) == (*state)) {
    //         return true;
    //     }
    // }
    // return false;

    return basis.find(state) != basis.end();
}

template<typename StateType>
inline bool is_in_basis(const BasisType<StateType>& basis, const StateType& state) {
    return is_in_basis(basis, std::make_shared<StateType>(state));
}

template<typename StateType>
class State_Graph {
    public:
        explicit State_Graph(const State<StateType>& init_state,
                        const Operator<StateType>& A_op,
                        const std::vector<Operator<StateType>>& operator_decoherence = {});
        void show() const;

        BasisType<StateType> get_basis() const { return basis_; }
    private:
        BasisType<StateType> basis_;
};

template<typename StateType>
State_Graph<StateType>::State_Graph(const State<StateType>& init_state,
                         const Operator<StateType>& A_op,
                         const std::vector<Operator<StateType>>& operator_decoherence) {
    auto state_map = init_state.get_map();
    std::queue<std::shared_ptr<StateType>> state_queue;
    for (auto p: state_map) {
        basis_.insert(p.first);
        state_queue.push(p.first);
    }

    while (!state_queue.empty()) {
        auto state = state_queue.front();
        state_queue.pop();
        auto res = A_op.run(State<StateType>(state));


        for (auto p: res.get_map()) {
            if (!is_in_basis(basis_, p.first)) {
                basis_.insert(p.first);
                //basis_.insert(st);
                state_queue.push(p.first);
            }
        }

        for (const auto& op: operator_decoherence) {
            auto new_res = op.run(State<StateType>(state));

            for (auto p: new_res.get_map()) {
                if (!is_in_basis(basis_, p.first)) {
                    basis_.insert(p.first);
                    //basis_.insert(st);
                    state_queue.push(p.first);
                }
            }
        }
    }
}

} // namespace QComputations
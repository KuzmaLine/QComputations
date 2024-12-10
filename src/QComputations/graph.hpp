#pragma once
#include <iostream>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "additional_operators.hpp"
#include "quantum_operators.hpp"
#include "state.hpp"

namespace QComputations {

    template <typename StateType>
    bool is_in_basis(const BasisType<StateType>& basis, std::shared_ptr<StateType> state) {
        for (auto st : basis) {
            if ((*st) == (*state)) {
                return true;
            }
        }

        return false;
    }

    template <typename StateType>
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

    template <typename StateType>
    State_Graph<StateType>::State_Graph(const State<StateType>& init_state,
                                        const Operator<StateType>& A_op,
                                        const std::vector<Operator<StateType>>& operator_decoherence) {
        auto state_components = init_state.get_state_components();
        std::queue<StateType> state_queue;
        for (auto state : state_components) {
            basis_.insert(state);
            state_queue.push(*state);
        }

        while (!state_queue.empty()) {
            auto state = state_queue.front();
            state_queue.pop();
            auto res = A_op.run(State<StateType>(state));

            for (auto st : res.get_state_components()) {
                if (!is_in_basis(basis_, st)) {
                    basis_.insert(std::shared_ptr<StateType>(new StateType(*st)));
                    state_queue.push(*st);
                }
            }

            for (const auto& op : operator_decoherence) {
                auto new_res = op.run(State<StateType>(state));

                for (auto st : new_res.get_state_components()) {
                    if (!is_in_basis(basis_, st)) {
                        basis_.insert(std::shared_ptr<StateType>(new StateType(*st)));
                        state_queue.push(*st);
                    }
                }
            }
        }
    }

}  // namespace QComputations
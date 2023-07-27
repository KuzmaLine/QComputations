#include "graph.hpp"
#include <algorithm>
#include <functional>

namespace {
    constexpr int DOWN = 0;
    constexpr int UP = 1;
}

State_Graph::State_Graph(const Cavity_State& init_state, bool with_loss_photons) {
    basis_.insert(init_state);
    std::queue<Cavity_State> state_queue_;
    state_queue_.push(init_state);

    while(!state_queue_.empty()) {
        auto cur_state = state_queue_.front();
        auto cur_n = cur_state.n();
        auto tmp_state = cur_state;
        state_queue_.pop();

        if (with_loss_photons and cur_n != 0) {
            tmp_state.set_n(cur_n - 1);
            if (std::find(basis_.begin(), basis_.end(), tmp_state) == basis_.end()) {
                basis_.insert(tmp_state);
                state_queue_.push(tmp_state);
            }

            from_[cur_state].insert(tmp_state);
            tmp_state.set_n(cur_n);
        }

        for (size_t i = 0; i < cur_state.m(); i++) {
            if (cur_state.get_qubit(i) == UP) {
                tmp_state.set_qubit(i, DOWN);
                tmp_state.set_n(cur_n + 1);

                if (std::find(basis_.begin(), basis_.end(), tmp_state) == basis_.end()) {
                    basis_.insert(tmp_state);
                    state_queue_.push(tmp_state);
                }

                from_[cur_state].insert(tmp_state);
                to_[tmp_state].insert(cur_state);
                tmp_state.set_qubit(i, UP);
                tmp_state.set_n(cur_n);
            }
        }

        if (cur_n != 0) {
            for (size_t i = 0; i < cur_state.m(); i++) {
                if (cur_state.get_qubit(i) == DOWN) {
                    tmp_state.set_qubit(i, UP);
                    tmp_state.set_n(cur_n - 1);

                    if (std::find(basis_.begin(), basis_.end(), tmp_state) == basis_.end()) {
                        basis_.insert(tmp_state);
                        state_queue_.push(tmp_state);
                    }

                    from_[cur_state].insert(tmp_state);
                    to_[tmp_state].insert(cur_state);
                    tmp_state.set_qubit(i, DOWN);
                    tmp_state.set_n(cur_n);
                }
            }
        }
    }
}

void State_Graph::show() const {
    for (const auto& state: basis_) {
        std::cout << state.to_string() << " : ";
        if (state.get_index() != 0) {
            for (const auto& to_state: from_.at(state)) {
                std::cout << to_state.to_string() << " ";
            }
        }
        std::cout << std::endl;
    }
}

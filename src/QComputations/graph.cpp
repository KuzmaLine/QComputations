#include "graph.hpp"
#include <algorithm>
#include <functional>
#include "functions.hpp"

namespace QComputations {

bool is_in_basis(const std::set<Basis_State>& basis, const Basis_State& state) {
    return std::find(basis.begin(), basis.end(), state) != basis.end();
}

bool is_in_basis(const std::set<State>& basis, const State& state) {
    return std::find(basis.begin(), basis.end(), state) != basis.end();
}

State_Graph::State_Graph(const State& init_state,
                        std::function<Formule(const Basis_State&)> func,
                        std::function<Formule(const Basis_State&)> func_decoherence) {
    auto state_components = init_state.get_state_components();
    std::queue<Basis_State> state_queue;
    for (const auto& state: state_components) {
        basis_.insert(state);
        state_queue.push(state);
    }

    while (!state_queue.empty()) {
        auto state = state_queue.front();
        state_queue.pop();
        auto res = func(state);
        auto res_decoherence = func_decoherence(state);

        for (const auto& state: res.get_states()) {
            if (!is_in_basis(basis_, state)) {
                basis_.insert(state);
                state_queue_.push(state);
            }
        }

        for (const auto& state: res.get_states()) {
            if (!is_in_basis(basis_, state)) {
                basis_.insert(state);
                state_queue_.push(state);
            }
        }
    }
}

/*
State_Graph::State_Graph(const State& init_state) {
    basis_.insert(init_state);
    std::queue<State> state_queue_;
    state_queue_.push(init_state);
    auto cavities_count = init_state.cavities_count();
    auto e_levels_count = init_state.e_levels_count();

    while(!state_queue_.empty()) {
        auto cur_state = state_queue_.front();
        auto tmp_state = cur_state;
        state_queue_.pop();

        for (size_t cavity_id = 0; cavity_id < cavities_count; cavity_id++) {
            for (size_t e_from = 0; e_from < e_levels_count; e_from++) {
                for (size_t e_to = e_from + 1; e_to < e_levels_count; e_to++) {
                    auto cur_n = cur_state.n(cavity_id, e_from, e_to);
                    if ((!is_zero(cur_state.get_leak_gamma(cavity_id))) and cur_n != 0 and cur_state.get_grid_energy() > cur_state.min_N()) {
                        tmp_state.set_n(cur_n - 1, cavity_id, e_from, e_to);
                        if (std::find(basis_.begin(), basis_.end(), tmp_state) == basis_.end()) {
                            basis_.insert(tmp_state);
                            state_queue_.push(tmp_state);
                        }

                        from_[cur_state].insert(tmp_state);
                        tmp_state.set_n(cur_n, cavity_id, e_from, e_to);
                    }

                    if (!is_zero(cur_state.get_gain_gamma(cavity_id)) and (cur_state.get_grid_energy() < cur_state.max_N())) {
                        tmp_state.set_n(cur_n + 1, cavity_id, e_from, e_to);
                        if (std::find(basis_.begin(), basis_.end(), tmp_state) == basis_.end()) {
                            basis_.insert(tmp_state);
                            state_queue_.push(tmp_state);
                        }

                        from_[cur_state].insert(tmp_state);
                        tmp_state.set_n(cur_n, cavity_id, e_from, e_to);
                    }

                    for (auto to_cavity: cur_state.get_neighbours(cavity_id)) {
                        if (!is_zero(cur_state.get_gamma(cavity_id, to_cavity)) and cur_n != 0) {
                            tmp_state.set_n(tmp_state.n(to_cavity, e_from, e_to) + 1, to_cavity, e_from, e_to);
                            tmp_state.set_n(cur_n - 1, cavity_id, e_from, e_to);
                            if (std::find(basis_.begin(), basis_.end(), tmp_state) == basis_.end()) {
                                basis_.insert(tmp_state);
                                state_queue_.push(tmp_state);
                            }

                            tmp_state.set_n(tmp_state.n(to_cavity, e_from, e_to) - 1, to_cavity, e_from, e_to);
                            tmp_state.set_n(cur_n, cavity_id, e_from, e_to); 
                        }
                    }

                    for (size_t i = 0; i < cur_state.m(cavity_id); i++) {
                        if (cur_state.get_qubit(cavity_id, i) == e_to) {
                            tmp_state.set_qubit(cavity_id, i, e_from);
                            tmp_state.set_n(cur_n + 1, cavity_id, e_from, e_to); 

                            if (std::find(basis_.begin(), basis_.end(), tmp_state) == basis_.end()) {
                                basis_.insert(tmp_state);
                                state_queue_.push(tmp_state);
                            }

                            from_[cur_state].insert(tmp_state);
                            to_[tmp_state].insert(cur_state);
                            tmp_state.set_qubit(cavity_id, i, e_to);
                            tmp_state.set_n(cur_n, cavity_id, e_from, e_to); 
                        }
                    }

                    if (cur_n != 0) {
                        for (size_t i = 0; i < cur_state.m(cavity_id); i++) {
                            if (cur_state.get_qubit(cavity_id, i) == e_from) {
                                tmp_state.set_qubit(cavity_id, i, e_to);
                                tmp_state.set_n(cur_n - 1, cavity_id, e_from, e_to); 

                                if (std::find(basis_.begin(), basis_.end(), tmp_state) == basis_.end()) {
                                    basis_.insert(tmp_state);
                                    state_queue_.push(tmp_state);
                                }

                                from_[cur_state].insert(tmp_state);
                                to_[tmp_state].insert(cur_state);
                                tmp_state.set_qubit(cavity_id, i, e_from);
                                tmp_state.set_n(cur_n, cavity_id, e_from, e_to); 
                            }
                        }
                    }
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
*/

} // namespace QComputations
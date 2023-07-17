#include "graph.hpp"
#include <algorithm>
#include <functional>

namespace {
    constexpr int DOWN = 0;
    constexpr int UP = 1;
}

Basis_Graph::Basis_Graph(const Basis& init_state, bool with_loss_photons) {
    basis_.insert(init_state);
    basis_queue_.push(init_state);

    while(!basis_queue_.empty()) {
        auto cur_basis = basis_queue_.front();
        auto cur_n = cur_basis.n();
        auto tmp_basis = cur_basis;
        basis_queue_.pop();

        if (with_loss_photons and cur_n != 0) {
            tmp_basis.set_n(cur_n - 1);
            if (std::find(basis_.begin(), basis_.end(), tmp_basis) == basis_.end()) {
                basis_.insert(tmp_basis);
                basis_queue_.push(tmp_basis);
            }

            from_[cur_basis].insert(tmp_basis);
            tmp_basis.set_n(cur_n);
        }

        for (size_t i = 0; i < cur_basis.size(); i++) {
            if (cur_basis.get_qubit(i) == UP) {
                tmp_basis.set_qubit(i, DOWN);
                tmp_basis.set_n(cur_n + 1);

                if (std::find(basis_.begin(), basis_.end(), tmp_basis) == basis_.end()) {
                    basis_.insert(tmp_basis);
                    basis_queue_.push(tmp_basis);
                }

                from_[cur_basis].insert(tmp_basis);
                to_[tmp_basis].insert(cur_basis);
                tmp_basis.set_qubit(i, UP);
                tmp_basis.set_n(cur_n);
            }
        }

        if (cur_n != 0) {
            for (size_t i = 0; i < cur_basis.size(); i++) {
                if (cur_basis.get_qubit(i) == DOWN) {
                    tmp_basis.set_qubit(i, UP);
                    tmp_basis.set_n(cur_n - 1);

                    if (std::find(basis_.begin(), basis_.end(), tmp_basis) == basis_.end()) {
                        basis_.insert(tmp_basis);
                        basis_queue_.push(tmp_basis);
                    }

                    from_[cur_basis].insert(tmp_basis);
                    to_[tmp_basis].insert(cur_basis);
                    tmp_basis.set_qubit(i, DOWN);
                    tmp_basis.set_n(cur_n);
                }
            }
        }
    }
}

void Basis_Graph::show() const {
    for (const auto& state: basis_) {
        std::cout << state.to_string() << " : ";
        for (const auto& to_state: from_.at(state)) {
            std::cout << to_state.to_string() << " ";
        }
        std::cout << std::endl;
    }
}

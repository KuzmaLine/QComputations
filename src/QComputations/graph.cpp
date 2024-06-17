#include "graph.hpp"

namespace QComputations {

template<>
bool is_in_basis<TCH_State>(const BasisType<TCH_State>& basis, std::shared_ptr<TCH_State> state) {
    for (auto st: basis) {
        if (*st == *state) {
            return true;
        }
    }

    return false;
}

} // namespace QComputations
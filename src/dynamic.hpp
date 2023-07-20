#pragma once
#include <vector>
#include <memory>
#include "state.hpp"
#include "hamiltonian.hpp"

namespace Evolution {
    using COMPLEX = std::complex<double>;
    using Probs = Matrix<double>;

    Probs evolution(const State& init_state, const Hamiltonian& H, const std::vector<double>& time_vec);
}

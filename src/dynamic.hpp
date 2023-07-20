#pragma once
#include <vector>
#include <memory>
#include "state.hpp"
#include "hamiltonian.hpp"

namespace Evolution {
    using COMPLEX = std::complex<double>;
    using Probs = std::vector<std::vector<double>>;

    Probs evolution(const Basis& init_base, Hamiltonian* H, const std::vector<double>& time_vec);
}

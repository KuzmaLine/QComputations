#pragma once
#include <vector>
#include <memory>
#include "basis.hpp"
#include "hamiltonian.hpp"

namespace Evolution {
    using COMPLEX = std::complex<double>;
    using Probs = std::vector<std::vector<double>>;

    Probs evolution(const Basis& init_base, Hamiltonian* H, const std::vector<double>& time_vec);
}

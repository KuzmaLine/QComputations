#pragma once
#include <vector>
#include <memory>
#include "state.hpp"
#include "hamiltonian.hpp"

namespace Evolution {
    using COMPLEX = std::complex<double>;
    using Probs = Matrix<double>;

    Probs evol(const std::vector<COMPLEX>& init_state, Hamiltonian& H, const std::vector<double>& time_vec);
}

#include "dynamic.hpp"
#include "additional_operators.hpp"
#include "functions.hpp"

namespace {
    size_t get_index_in_set(const std::set<State>& basis, const State& state) {
        return std::distance(basis.begin(), basis.find(state));
    }
}

namespace Evolution {
    Probs evolution(const std::vector<COMPLEX>& init_state, Hamiltonian& H, const std::vector<double>& time_vec) {
        auto p = H.eigen();
        auto eigen_values = p.first;
        auto eigen_vectors = p.second;
        size_t n = eigen_values.size();

        std::vector<COMPLEX> lambda;
        for (size_t i = 0; i < eigen_values.size(); i++) {
            lambda.emplace_back(scalar_product(eigen_vectors.col(i), init_state)); // <KSI(0)|PHI_i> 
        }

        Probs probs(eigen_values.size(), time_vec.size());
        size_t time_index = 0;

        eigen_vectors = eigen_vectors.transpose();
        for (const auto& t: time_vec) {
            std::vector<COMPLEX> psi_t(eigen_values.size(), 0);

            for (size_t i = 0; i < eigen_values.size(); i++) {
                for (size_t j = 0; j < psi_t.size(); j++) {
                    psi_t[j] += lambda[i] * std::exp(COMPLEX(0, 1 / config::h * eigen_values[i] * t)) * eigen_vectors[i][j];
                }
            }

            for (size_t i = 0; i < eigen_values.size(); i++) {
                double tmp = std::abs(psi_t[i]);
                probs[i][time_index] = tmp * tmp;
            }
            time_index++;
        }

        return probs;
    }
}


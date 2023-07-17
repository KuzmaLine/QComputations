#include "dynamic.hpp"

namespace {
    COMPLEX scalar_product(const std::vector<COMPLEX>& a, const std::vector<COMPLEX>& b) {
        COMPLEX res(0);

        for (size_t i = 0; i < a.size(); i++) {
            res += std::conj(a[i]) * b[i];
        }

        return res;
    }

    size_t get_index_in_set(const std::set<Basis>& bases, const Basis& basis) {
        return std::distance(bases.begin(), bases.find(basis));
    }
}

namespace Evolution {
    Probs evolution(const Basis& init_basis, Hamiltonian* H, const std::vector<double>& time_vec) {
        H->Reduce_H(init_basis);
        auto p = H->reduced_eigen();
        auto eigen_values = p.first;
        auto eigen_vectors = p.second;
        size_t n = eigen_values.size();
        std::vector<COMPLEX> tmp_init_state(n, 0);

        tmp_init_state[get_index_in_set(H->get_bases(), init_basis)] = COMPLEX(1);

        std::vector<COMPLEX> lambda;
        for (size_t i = 0; i < eigen_values.size(); i++) {
            lambda.emplace_back(scalar_product(eigen_vectors[i], tmp_init_state));
        }

        Probs probs(eigen_values.size(), std::vector<double>(time_vec.size()));
        size_t time_index = 0;

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


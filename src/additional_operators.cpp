#include "additional_operators.hpp"

double scalar_product(const std::vector<double>& a, const std::vector<double>& b) {
    double res = 0;

    for (size_t i = 0; i < a.size(); i++) {
        res += a[i] * b[i];
    }

    return res;
}

COMPLEX scalar_product(const std::vector<COMPLEX>& a, const std::vector<COMPLEX>& b) {
    COMPLEX res = 0;

    for (size_t i = 0; i < a.size(); i++) {
        COMPLEX tmp(b[i].real(), -(b[i].imag()));

        res += a[i] * tmp;
    }

    return res;
}


double norm(const std::vector<COMPLEX>& v) {
    double res = 0;

    for (const auto& num: v) {
        auto tmp = std::abs(num);
        res += tmp * tmp;
    }

    return std::sqrt(res);
}

/*
void show_matrix(const matrix& A) {
    for (const auto& v: A) {
        for (const auto& num: v) {
            std::cout << std::setw(25) << num << " ";
        }
        std::cout << std::endl;
    }
}
*/

void show_vector (const std::vector<COMPLEX>& v) {
    for (const auto& num: v) {
        std::cout << std::setw(15) << num << " ";
    }

    std::cout << std::endl;
}

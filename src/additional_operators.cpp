#include "additional_operators.hpp"

COMPLEX quantum::operator | (const std::vector<COMPLEX>& a, const std::vector<COMPLEX>& b) { 
    COMPLEX res = 0;

    for (size_t i = 0; i < a.size(); i++) {
        res += std::conj(a[i]) * b[i];
    }

    return res;
}

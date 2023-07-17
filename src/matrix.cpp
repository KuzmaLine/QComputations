#include "matrix.hpp"
#include <iostream>

namespace {
    using COMPLEX = std::complex<double>;
}

template<>
template<>
Matrix<COMPLEX> Matrix<COMPLEX>::operator* <double>(const Matrix<double>& A) const {
    assert(m_ == A.n());
    Matrix<COMPLEX> res(n_, A.m(), 0);

    for (size_t i = 0; i < n_; i++) {
        for (size_t j = 0; j < A.m(); j++) {
            for (size_t k = 0; k < m_; k++) {
                res.mass_[res.get_index(i, j)] += mass_[this->get_index(i, k)] * A[k][j];
            }
        }
    }

    return res;
}

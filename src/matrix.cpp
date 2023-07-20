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
                //res.mass_[res.get_index(i, j)] += std::conj(mass_[this->get_index(i, k)]) * A[k][j];
                res.mass_[res.get_index(i, j)] += mass_[this->get_index(i, k)] * A[k][j];
            }
        }
    }

    return res;
}

/*
template<>
Matrix<COMPLEX> Matrix<COMPLEX>::operator* <COMPLEX>(const Matrix<COMPLEX>& A) const {
    assert(m_ == A.n_);
    Matrix<COMPLEX> res(n_, A.m_, 0);

    for (size_t i = 0; i < n_; i++) {
        for (size_t j = 0; j < A.m_; j++) {
            for (size_t k = 0; k < m_; k++) {
                res.mass_[res.get_index(i, j)] += std::conj(mass_[this->get_index(i, k)]) * A.mass_[A.get_index(k, j)];
            }
        }
    }

    return res;
}

template<>
std::vector<COMPLEX> Matrix<COMPLEX>::operator* <COMPLEX>(const std::vector<COMPLEX>& v) const {
    assert(m_ == v.size());
    std::vector<COMPLEX> res(v.size(), COMPLEX(0, 0));

    size_t n = v.size();

    for (size_t i = 0; i < n; i++) {
        for (size_t k = 0; k < n; k++) {
            res[i] += std::conj(mass_[this->get_index(i, k)]) * v[k];
        }
    }

    return res;
}
*/
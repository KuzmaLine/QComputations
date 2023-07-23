#include "matrix.hpp"
#include <iostream>
#include <mkl_cblas.h>
#include <mkl_blas.h>

namespace {
    using COMPLEX = std::complex<double>;
}

template<>
Matrix<double> Matrix<double>::operator* <double>(const Matrix<double>& A) const {
    assert(m_ == A.n_);
    Matrix<double> res(n_, A.m_);

    double alpha = 1.0;
    double betta = 0.0;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_, A.m_, m_, alpha, mass_.data(), n_, A.mass_.data(), A.n_, betta, res.mass_.data(), n_);
    return res;
}

template<>
Matrix<int> Matrix<int>::operator* <int>(const Matrix<int>& A) const {
    Matrix<double> tmp_A(A);
    Matrix<double> tmp_this(*this);

    Matrix<double> tmp_res = tmp_this * tmp_A;

    return Matrix<int>(tmp_res);
}

template<>
Matrix<COMPLEX> Matrix<COMPLEX>::operator* <COMPLEX>(const Matrix<COMPLEX>& A) const {
    assert(m_ == A.n_);
    Matrix<COMPLEX> res(n_, A.m_);

    double alpha = 1.0;
    double betta = 0.0;
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_, A.m_, m_, &alpha, mass_.data(), n_, A.mass_.data(), A.n_, &betta, res.mass_.data(), n_);
    return res;
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

namespace {
    lapack_complex_double make_complex(const double a, const double b) {
        lapack_complex_double c;

        c.real = a;
        c.imag = b;

        return c;
    }
}

template<>
lapack_complex_double* Matrix<COMPLEX>::to_upper_lapack() const {
    lapack_complex_double* a;
    size_t index = 0;
    a = new lapack_complex_double [n_ * m_];
    for (size_t i = 0; i < n_; i++) {
        for (size_t j = i; j < m_; j++) {
            a[get_index(i, j)] = make_complex(mass_[get_index(i, j)].real(), mass_[get_index(i, j)].imag());
            //a[index++] = make_complex(mass_[get_index(i, j)].real(), mass_[get_index(i, j)].imag());
        }
    }

    return a;
}

template<>
lapack_complex_double* Matrix<COMPLEX>::to_lapack() const {
    lapack_complex_double* a;

    a = new lapack_complex_double [n_ * m_];
    for (size_t i = 0; i < n_; i++) {
        for (size_t j = 0; j < m_; j++) {
            a[get_index(i, j)] = make_complex(mass_[get_index(i, j)].real(), mass_[get_index(i, j)].imag());
        }
    }

    return a;
}
#define MKL_Complex16 std::complex<double>

#include "matrix.hpp"

#include <mkl_blas.h>
#include <mkl_cblas.h>

#include <chrono>
#include <fstream>
#include <iostream>

#include "functions.hpp"
#include "mpi_functions.hpp"

namespace QComputations {

namespace {
using COMPLEX = std::complex<double>;
}

template <>
Matrix<double> Matrix<double>::operator*(const Matrix<double>& A) const {
    assert(m_ == A.n_);
    Matrix<double> res(this->get_matrix_style(), n_, A.m_);

    double alpha = 1.0;
    double betta = 0.0;

    auto type = CblasRowMajor;
    if (!(this->is_c_style())) type = CblasColMajor;
    cblas_dgemm(type, CblasNoTrans, CblasNoTrans, n_, A.m_, m_, alpha, mass_.data(), this->LD(), A.mass_.data(), A.LD(),
                betta, res.mass_.data(), res.LD());
    return res;
}

template <>
Matrix<double>& Matrix<double>::operator*=(const Matrix<double>& A) {
    assert(m_ == A.n_);

    double alpha = 1.0;
    double betta = 0.0;

    auto type = CblasRowMajor;
    if (!(this->is_c_style())) type = CblasColMajor;
    cblas_dgemm(type, CblasNoTrans, CblasNoTrans, n_, A.m_, m_, alpha, mass_.data(), this->LD(), A.mass_.data(), A.LD(),
                betta, mass_.data(), this->LD());

    return *this;
}

template <>
Matrix<int> Matrix<int>::operator*(const Matrix<int>& A) const {
    Matrix<double> tmp_A(A);
    Matrix<double> tmp_this(*this);

    Matrix<double> tmp_res = tmp_this * tmp_A;

    return Matrix<int>(tmp_res);
}

template <>
Matrix<COMPLEX> Matrix<COMPLEX>::operator*(const Matrix<COMPLEX>& A) const {
    assert(m_ == A.n_);
    Matrix<COMPLEX> res(this->get_matrix_style(), n_, A.m_);

    COMPLEX alpha(1, 0);
    COMPLEX betta(0, 0);

    auto type = CblasRowMajor;
    if (!(this->is_c_style())) type = CblasColMajor;
    cblas_zgemm(type, CblasNoTrans, CblasNoTrans, n_, A.m_, m_, &alpha, mass_.data(), this->LD(), A.mass_.data(),
                A.LD(), &betta, res.mass_.data(), res.LD());
    return res;
}

template <>
Matrix<COMPLEX>& Matrix<COMPLEX>::operator*=(const Matrix<COMPLEX>& A) {
    assert(m_ == A.n_);

    COMPLEX alpha(1, 0);
    COMPLEX betta(0, 0);

    auto type = CblasRowMajor;
    if (!(this->is_c_style())) type = CblasColMajor;
    cblas_zgemm(type, CblasNoTrans, CblasNoTrans, n_, A.m_, m_, &alpha, mass_.data(), this->LD(), A.mass_.data(),
                A.LD(), &betta, mass_.data(), this->LD());
    return *this;
}

template <>
Matrix<double>& Matrix<double>::operator*=(double num) {
    ILP_TYPE iONE = 1;

    ILP_TYPE size = this->n() * this->m();
    dscal(&size, &num, this->data(), &iONE);

    return *this;
}

template <>
Matrix<COMPLEX>& Matrix<COMPLEX>::operator*=(COMPLEX num) {
    ILP_TYPE iONE = 1;

    ILP_TYPE size = this->n() * this->m();
    zscal(&size, &num, this->data(), &iONE);

    return *this;
}

template <>
Matrix<double>& Matrix<double>::operator/=(double num) {
    ILP_TYPE iONE = 1;
    double new_num = double(1) / num;

    ILP_TYPE size = this->n() * this->m();
    dscal(&size, &new_num, this->data(), &iONE);

    return *this;
}

template <>
Matrix<COMPLEX>& Matrix<COMPLEX>::operator/=(COMPLEX num) {
    ILP_TYPE iONE = 1;
    COMPLEX new_num = COMPLEX(1, 0) / num;

    ILP_TYPE size = this->n() * this->m();
    zscal(&size, &new_num, this->data(), &iONE);

    return *this;
}

template <>
void Matrix<double>::write_to_csv_file(const std::string& filename) const {
    std::ofstream file(filename);

    size_t max_number_size = QConfig::instance().csv_max_number_size();
    size_t num_accuracy = QConfig::instance().csv_num_accuracy();

    const int delimiter_size = 1;
    const int one_elem_size = max_number_size + 1;
    const char char_delimiter = ',';

    for (size_t i = 0; i < this->n(); i++) {
        for (size_t j = 0; j < this->m(); j++) {
            auto cur_index = i * this->m() + j;

            if ((j + 1) != this->m()) {
                file << to_string_double_with_precision(this->elem(i, j), num_accuracy, max_number_size)
                     << char_delimiter;
            } else {
                file << to_string_double_with_precision(this->elem(i, j), num_accuracy, max_number_size) << "\n";
            }
        }
    }

    file.close();
}

template <>
void Matrix<COMPLEX>::write_to_csv_file(const std::string& filename) const {
    std::ofstream file(filename);

    size_t max_number_size = QConfig::instance().csv_max_number_size();
    size_t num_accuracy = QConfig::instance().csv_num_accuracy();

    const int delimiter_size = 1;
    const int one_elem_size = max_number_size + 1;
    const char char_delimiter = ',';

    for (size_t i = 0; i < this->n(); i++) {
        for (size_t j = 0; j < this->m(); j++) {
            auto cur_index = this->get_index(i, j);

            if ((j + 1) != this->m()) {
                file << to_string_complex_with_precision(this->elem(i, j), num_accuracy, max_number_size)
                     << char_delimiter;
            } else {
                file << to_string_complex_with_precision(this->elem(i, j), num_accuracy, max_number_size) << "\n";
            }
        }
    }

    file.close();
}

/* ---------------- FUNCTIONS ------------------------------ */

namespace {
CBLAS_TRANSPOSE char_to_trans(char ch_trans) {
    switch (ch_trans) {
        case 'N':
            return CblasNoTrans;
        case 'T':
            return CblasTrans;
        case 'C':
            return CblasConjTrans;
        default:
            assert(false);  // Некорректная операция транспозиции при перемножении
    }
}
}  // namespace

template <>
void optimized_multiply(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C, double alpha, double betta,
                        char trans_A, char trans_B) {
    assert(A.m() == B.n());
    assert(A.n() == C.n() and B.m() == C.m());

    auto type = CblasRowMajor;
    if (!(A.is_c_style())) type = CblasColMajor;
    cblas_dgemm(type, char_to_trans(trans_A), char_to_trans(trans_B), A.n(), B.m(), A.m(), alpha, A.data(), A.LD(),
                B.data(), B.LD(), betta, C.data(), C.LD());
}

template <>
void optimized_multiply(const Matrix<COMPLEX>& A, const Matrix<COMPLEX>& B, Matrix<COMPLEX>& C, COMPLEX alpha,
                        COMPLEX betta, char trans_A, char trans_B) {
    assert(A.m() == B.n());
    assert(A.n() == C.n() and B.m() == C.m());

    auto type = CblasRowMajor;
    if (!(A.is_c_style())) type = CblasColMajor;
    cblas_zgemm(type, char_to_trans(trans_A), char_to_trans(trans_B), A.n(), B.m(), A.m(), &alpha, A.data(), A.LD(),
                B.data(), B.LD(), &betta, C.data(), C.LD());
}

std::vector<Matrix<COMPLEX>> OPT_Runge_Kutt_4(const std::vector<double>& x, const Matrix<COMPLEX>& y0,
                                              std::function<void(double, const Matrix<COMPLEX>&, Matrix<COMPLEX>&)> f) {
    size_t len = x.size();
    size_t dim = y0.n();
    std::vector<Matrix<COMPLEX>> y(len, Matrix<COMPLEX>(y0.matrix_style(), dim, dim));
    y[0] = y0;

    Matrix<COMPLEX> k1(y0.matrix_style(), dim, dim);
    Matrix<COMPLEX> k2(y0.matrix_style(), dim, dim);
    Matrix<COMPLEX> k3(y0.matrix_style(), dim, dim);

    for (size_t i = 0; i < len - 1; i++) {
        double h = x[i + 1] - x[i];

        f(x[i], y[i], k1);
        k1 *= (h / 2.0);
        k1 += y[i];
        f(x[i] + h / 2.0, k1, k2);
        k2 *= (h / 2.0);
        k2 += y[i];
        f(x[i] + h / 2.0, k2, k3);
        k3 *= h;
        k3 += y[i];
        f(x[i] + h, k3, y[i + 1]);
        k1 -= y[i];
        k2 -= y[i];
        k3 -= y[i];

        y[i + 1] *= (h / 6.0);
        y[i + 1] += y[i];

        k1 /= 3.0;
        y[i + 1] += k1;
        k2 *= (double(2) / double(3));
        y[i + 1] += k2;
        k3 /= 3.0;
        y[i + 1] += k3;
    }

    return y;
}

std::vector<Matrix<COMPLEX>> OPT_Runge_Kutt_2(const std::vector<double>& x, const Matrix<COMPLEX>& y0,
                                              std::function<void(double, const Matrix<COMPLEX>&, Matrix<COMPLEX>&)> f) {
    size_t len = x.size();
    size_t dim = y0.n();
    std::vector<Matrix<COMPLEX>> y(len, Matrix<COMPLEX>(y0.matrix_style(), dim, dim));
    y[0] = y0;

    Matrix<COMPLEX> k1(y0.matrix_style(), dim, dim);

    for (size_t i = 0; i < len - 1; i++) {
        double h = x[i + 1] - x[i];

        f(x[i], y[i], k1);
        k1 *= h;
        k1 += y[i];
        f(x[i] + h, k1, y[i + 1]);
        k1 -= y[i];
        k1 /= COMPLEX(2.0, 0);
        y[i + 1] *= COMPLEX(h / 2.0, 0);
        y[i + 1] += k1;
        y[i + 1] += y[i];
    }

    return y;
}

/* --------------------- DON'T TOUCH ------------------------*/

namespace {
lapack_complex_double make_complex(const double a, const double b) {
    lapack_complex_double c(a, b);

    return c;
}
}  // namespace

template <>
lapack_complex_double* Matrix<COMPLEX>::to_upper_lapack() const {
    lapack_complex_double* a;
    size_t index = 0;
    a = new lapack_complex_double[n_ * m_];
    for (size_t i = 0; i < n_; i++) {
        for (size_t j = i; j < m_; j++) {
            a[get_index(i, j)] = make_complex(mass_[get_index(i, j)].real(), mass_[get_index(i, j)].imag());
        }
    }

    return a;
}

template <>
lapack_complex_double* Matrix<COMPLEX>::to_lapack() const {
    lapack_complex_double* a;

    a = new lapack_complex_double[n_ * m_];
    for (size_t i = 0; i < n_; i++) {
        for (size_t j = 0; j < m_; j++) {
            a[get_index(i, j)] = make_complex(mass_[get_index(i, j)].real(), mass_[get_index(i, j)].imag());
        }
    }

    return a;
}

}  // namespace QComputations
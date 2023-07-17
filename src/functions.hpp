#pragma once
#include "additional_operators.hpp"
#include "matrix.hpp"
#include <vector>
#include <complex>
#include <set>

namespace {
    using E_LEVEL = int;
    using COMPLEX = std::complex<double>;
    using vec_complex = std::vector<COMPLEX>;
    using vec_levels = std::vector<E_LEVEL>;
}

std::pair<double, double> givens(double a, double b);
void tridiagonal_QR(Matrix<double>& T);
size_t get_index_from_state(vec_levels state);
Matrix<double> MGS(const Matrix<COMPLEX>& A);
std::pair<std::vector<double>, Matrix<COMPLEX>> Hermit_Lanczos(const Matrix<COMPLEX>& A);
//std::pair<std::vector<double>, Matrix<double>> Hermit_Lanczos(const Matrix<COMPLEX>& A);
//std::vector<std::vector<double>> Hermit_Lanczos(const matrix& A);

#pragma once
#include "additional_operators.hpp"
#include "matrix.hpp"
#include <vector>
#include <complex>
#include <set>
#include <mkl_lapack.h>
#include <mkl_lapacke.h>

namespace {
    using E_LEVEL = int;
    using COMPLEX = std::complex<double>;
    using vec_complex = std::vector<COMPLEX>;
    using vec_levels = std::vector<E_LEVEL>;
}

std::vector<double> FROM_double_TO_vector(double* A, lapack_int n);
Matrix<COMPLEX> FROM_lapack_complex_double_TO_Matrix(lapack_complex_double* A, lapack_int n, lapack_int m);
std::vector<double> make_timeline(double start, double end, double step);
double off(const Matrix<double>& A);
std::pair<double, double> givens(double a, double b);
void tridiagonal_QR(Matrix<double>& T);
size_t get_index_from_state(vec_levels state);
Matrix<double> MGS(const Matrix<COMPLEX>& A);
std::pair<std::vector<double>, Matrix<COMPLEX>> Hermit_Lanczos(const Matrix<COMPLEX>& A);
double scalar_product(const std::vector<double>& a, const std::vector<double>& b); // <b|a>, (a, b)
COMPLEX scalar_product(const std::vector<COMPLEX>& a, const std::vector<COMPLEX>& b); // <b|a>, (a, b)
double norm(const std::vector<COMPLEX>& v);
//std::pair<std::vector<double>, Matrix<double>> Hermit_Lanczos(const Matrix<COMPLEX>& A);
//std::vector<std::vector<double>> Hermit_Lanczos(const matrix& A);
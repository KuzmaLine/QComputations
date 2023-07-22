#pragma once
#include "additional_operators.hpp"
#include "matrix.hpp"
#include <vector>
#include <complex>
#include <set>
#include <functional>
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
size_t get_index_from_state(vec_levels state);

double off(const Matrix<double>& A);
std::pair<double, double> givens(double a, double b);
void tridiagonal_QR(Matrix<double>& T);
Matrix<double> MGS(const Matrix<COMPLEX>& A);
std::pair<std::vector<double>, Matrix<COMPLEX>> Hermit_Lanczos(const Matrix<COMPLEX>& A);

double scalar_product(const std::vector<double>& a, const std::vector<double>& b); // <b|a>, (a, b)
COMPLEX scalar_product(const std::vector<COMPLEX>& a, const std::vector<COMPLEX>& b); // <b|a>, (a, b)
double norm(const std::vector<COMPLEX>& v);

// ------------------------------- template functions --------------------------------------

template<typename T, typename V>
std::vector<T> Runge_Kutt_4(const std::vector<V>& x, const T& y0, std::function<T(V, T)> f) {
    size_t len = x.size();
    std::vector<T> y(len);
    y[0] = y0;

    for (size_t i = 0; i < len - 1; i++) {
        //std::cout << i << " " << y[i] << " ";
        V h = x[i + 1] - x[i];

        T k1 = f(x[i], y[i]);
        T k2 = f(x[i] + h / 2.0, y[i] + k1 * h / 2.0);
        T k3 = f(x[i] + h / 2.0, y[i] + k2 * h / 2.0);
        T k4 = f(x[i] + h, y[i] + h * k3);
        y[i + 1] = y[i]  + h*(k1 + 2 * k2 + 2 * k3 + k4) / 6.0;
        //std::cout << h << " " << y[i + 1] << " " << 2 * x[i + 1] << std::endl;
    }

    return y;
}
//std::pair<std::vector<double>, Matrix<double>> Hermit_Lanczos(const Matrix<COMPLEX>& A);
//std::vector<std::vector<double>> Hermit_Lanczos(const matrix& A);
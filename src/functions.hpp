#pragma once
#include "additional_operators.hpp"
#include "matrix.hpp"
#include "config.hpp"
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
bool is_zero(double a);

double off(const Matrix<double>& A);
std::pair<double, double> givens(double a, double b);
void tridiagonal_QR(Matrix<double>& T);
Matrix<double> MGS(const Matrix<COMPLEX>& A);
std::pair<std::vector<double>, Matrix<COMPLEX>> Hermit_Lanczos(const Matrix<COMPLEX>& A);

std::function<double(double)> Cubic_Spline_Interpolate(const std::vector<double>& x, const std::vector<double>& y);

double scalar_product(const std::vector<double>& a, const std::vector<double>& b); // <b|a>, (a, b)
COMPLEX scalar_product(const std::vector<COMPLEX>& a, const std::vector<COMPLEX>& b); // <b|a>, (a, b)
double norm(const std::vector<COMPLEX>& v);

// ------------------------------- template functions --------------------------------------

template<typename T>
Matrix<T> E_Matrix(size_t n) {
    Matrix<T> E(n, n, 0);

    for (size_t i = 0; i < n; i++) {
        E[i][i] = T(1);
    }

    return E;
}

template<typename T>
Matrix<T> tensor_multiply(const Matrix<T>& A, const Matrix<T>& B) {
    auto n = A.size() * B.size();
    Matrix<T> C(n, n, 0);

    for (size_t i_a = 0; i_a < A.size(); i_a++) {
        for (size_t j_a = 0; j_a < A.size(); j_a++) {
            for (size_t i_b = 0; i_b < B.size(); i_b++) {
                for (size_t j_b = 0; j_b < B.size(); j_b++) {
                    size_t i = i_a * B.size() + i_b;
                    size_t j = j_a * B.size() + j_b;

                    C[i][j] = A[i_a][j_a] * B[i_b][j_b];
                }
            }
        }
    }

    return C;
}

template<typename T>
std::ostream& operator<<(std::ostream& out, const Matrix<T>& matrix) {
    for (size_t i = 0; i < matrix.n(); i++) {
        for (size_t j = 0; j < matrix.m(); j++) {
            out << std::setw(config::WIDTH) << matrix[i][j] << " ";
        }

        out << std::endl;
    }

    return out;
}

template<typename T, typename V>
std::vector<V> Runge_Kutt_4(const std::vector<T>& x, const V& y0, std::function<V(T, V)> f) {
    size_t len = x.size();
    std::vector<V> y(len);
    y[0] = y0;

    for (size_t i = 0; i < len - 1; i++) {
        //std::cout << i << " " << y[i] << " ";
        T h = x[i + 1] - x[i];

        V k1 = f(x[i], y[i]);
        V k2 = f(x[i] + h / 2.0, y[i] + k1 * (h / 2.0));
        V k3 = f(x[i] + h / 2.0, y[i] + k2 * (h / 2.0));
        V k4 = f(x[i] + h, y[i] + k3 * h);
        y[i + 1] = y[i]  + (k1 + k2 * 2 + k3 * 2 + k4) * (h / 6.0);
        //std::cout << h << " " << y[i + 1] << " " << 2 * x[i + 1] << std::endl;
    }

    return y;
}

template<typename T>
std::vector<T> Pro_Race_Algorithm(const Matrix<T>& B, const std::vector<T>& y) {
    size_t k = B.size();
 
    // Down move
    std::vector<T> x(k, 0);
    std::vector<T> a(k, 0); 
    a[0] = -B[0][1]/ B[0][0];

    std::vector<T> b(k, 0); 
    b[0] = y[0] / B[0][0];
 
    for (size_t i = 1; i < k - 1; i++) {
        a[i] = B[i][i + 1] / (-B[i][i] - B[i][i - 1] * a[i - 1]);
        b[i] = (B[i][i - 1] * b[i - 1] - y[i])/(-B[i][i] - a[i - 1] * B[i][i - 1]);
    }
    a[k - 1] = 0;
    x[k - 1] = (B[k - 1][k - 2] * b[k - 2] - y[k - 1])/( -B[k - 1][k - 1] - B[k - 1][k - 2] * a[k - 2]);
 
    // Up move
 
    for (long long i = k - 2; i > -1; i--) {
        x[i] = a[i] * x[i + 1] + b[i];
    }
 
    return x;
}

//std::pair<std::vector<double>, Matrix<double>> Hermit_Lanczos(const Matrix<COMPLEX>& A);
//std::vector<std::vector<double>> Hermit_Lanczos(const matrix& A);
#pragma once
#include "matrix.hpp"
#include <functional>
#include <random>
#include <complex>
namespace matrix_testing {
    template<typename T>
    Matrix<T> create_rand_matrix(size_t n, size_t m, T a, T b) {
        Matrix<T> A(n, m);
        std::uniform_real_distribution<T> random(a, b);
        std::default_random_engine re;
        for (size_t i = 0; i < n; i++) {
            for(size_t j = 0; j < m; j++) {
                A[i][j] = random(re);
            }
        }

        return A;
    }

    template<typename T>
    Matrix<std::complex<T>> create_rand_matrix(size_t n, size_t m, std::complex<T> a, std::complex<T> b) {
        Matrix<std::complex<T>> A(n, m);

        std::uniform_real_distribution<T> random_real(a.real(), b.real());
        std::uniform_real_distribution<T> random_imag(a.imag(), b.imag());
        std::default_random_engine re;
        for (size_t i = 0; i < n; i++) {
            for(size_t j = 0; j < m; j++) {
                T rand_a = random_real(re);
                T rand_b = random_imag(re);
                A[i][j] = std::complex<T>(rand_a, rand_b);
            }
        }

        return A;
    }
}

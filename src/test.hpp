#pragma once
#include "matrix.hpp"
#include <functional>
#include <random>
#include <complex>
namespace matrix_testing {
    template<typename T>
    Matrix<T> create_rand_matrix(size_t n, size_t m, T a, T b) {
        Matrix<T> A(n, m);

        std::random_device rd;
        std::mt19937 re(rd());
        std::uniform_real_distribution<T> random(a, b);
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

        std::random_device rd;
        std::mt19937 re(rd());
        std::uniform_real_distribution<T> random_real(a.real(), b.real());
        std::uniform_real_distribution<T> random_imag(a.imag(), b.imag());
        for (size_t i = 0; i < n; i++) {
            for(size_t j = 0; j < m; j++) {
                T rand_a = random_real(re);
                T rand_b = random_imag(re);
                A[i][j] = std::complex<T>(rand_a, rand_b);
            }
        }

        return A;
    }

    template<typename T>
    Matrix<std::complex<T>> create_hermit_rand_matrix(size_t n, size_t m, std::complex<T> a, std::complex<T> b) {
        Matrix<std::complex<T>> A(n, m);

        std::random_device rd;
        std::mt19937 re(rd());
        std::uniform_real_distribution<T> random_real(a.real(), b.real());
        std::uniform_real_distribution<T> random_imag(a.imag(), b.imag());
        for (size_t i = 0; i < n; i++) {
            for(size_t j = i; j < m; j++) {
                T rand_a = random_real(re);
                T rand_b = random_imag(re);
                A[i][j] = std::complex<T>(rand_a, rand_b);
                A[j][i] = std::conj(A[i][j]);

                if (i == j) A[i][i] = rand_a;
            }
        }

        return A;
    }
}

namespace functions_testing {
    
    template<typename T>
    void check_eigenvectors(std::vector<double> eigenvalues, const Matrix<T>& eigenvectors, const Matrix<T>& matrix) {
        // Получаем размерность матрицы
        int n = matrix.size();

        // Проходим по всем собственным значениям
        for (int i = 0; i < eigenvalues.size(); i++) {
            // Получаем собственное значение и соответствующий ему собственный вектор
            double lambda = eigenvalues[i];
            //std::complex<double> norm = 0.0;
            // Проверяем условие Av = lambda v для данного собственного вектора
            std::vector<std::complex<double>> Av(n);
            for (int j = 0; j < n; j++) {
                std::complex<double> sum = 0;
                Av = matrix * eigenvectors.col(i);
                
                /*for (int k = 0; k < n; k++) {
                    sum += matrix[j][k] * eigenvectors[k][i];
                }
                Av[j] = sum;
                */
                //norm += Av[j] * Av[j];
            }


            //for (int j = 0; j < n; j++) {
            //    Av[j] /= std::sqrt(norm);
            //}

            std::cout << "Собственное значение " << lambda << " и соответствующий ему собственный вектор: [ ";
            for (int j = 0; j < n; j++) {
                std::cout << eigenvectors[j][i] << " ";
            }
            std::cout << "]" << std::endl;
            std::cout << std::endl;
            for (int j = 0; j < n; j++) {
                std::cout << Av[j] << " ";
            }
            std::cout << std::endl;
            for (int j = 0; j < n; j++) {
                std::cout << eigenvectors[j][i] * lambda << " ";
            }
            std::cout << std::endl;
        }
    }
}
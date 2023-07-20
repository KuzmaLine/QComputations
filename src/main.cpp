#define _USE_MATH_DEFINES
#include <iostream>
#include "functions.hpp"
#include "additional_operators.hpp"
#include "matrix.hpp"
#include "hamiltonian.hpp"
#include "basis.hpp"
#include "matplotlibcpp.h"
#include "test.hpp"
//#include "config.hpp"
//#include "graph.hpp"
//#include "dynamic.hpp"

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
            for (int k = 0; k < n; k++) {
                sum += matrix[j][k] * eigenvectors[k][i];
            }
            Av[j] = sum;
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

namespace plt = matplotlibcpp;

int main(void) {
    int n = 1;
    int m = 2;

    State state("|1>|00>");
    H_TC H(n, m, state);
    std::cout << state.to_string() << " n = " << n << " m = " << m <<" h = " << config::h << " w = " << config::w << " g = " << config::g << " LOSS_PHOTONS = " << config::LOSS_PHOTONS << std::endl;

    //State_Graph graph(state);
    //graph.show();

    //std::cout << std::endl;
    auto bases = H.get_basis();

    for (const auto& b: bases) {
        std::cout << std::setw(config::WIDTH) << b.to_string() << " ";
    }
    std::cout << std::endl;

    H.show(config::WIDTH);

    n = 5;
    Matrix<COMPLEX> A(n, n, 0);
    A[0][1] = COMPLEX(2);
    A[0][2] = COMPLEX(3);
    A[0][3] = COMPLEX(4, -4);
    A[0][4] = COMPLEX(2);
    A[1][2] = COMPLEX(5);
    A[1][3] = COMPLEX(7);
    A[1][4] = COMPLEX(2);
    A[2][3] = COMPLEX(4, 2);
    A[2][4] = COMPLEX(2);
    A[3][4] = COMPLEX(3, 10);

    for (size_t i = 1; i < n; i++) {
        for (size_t j = 0; j < i; j++) {
            A[i][j] = std::conj(A[j][i]);
        }
    }

    A.show();

    //show_vector(A.row(0));

    auto Hessen = MGS(A);

    Hessen.show();

    auto p = Hermit_Lanczos(A);

    p.second.show();

    for(const auto& lambda: p.first) {
        std::cout << lambda << " ";
    }

    std::cout << std::endl;

    check_eigenvectors(p.first, p.second, A);

    //Matrix<COMPLEX> a = matrix_testing::create_rand_matrix(32, 32, COMPLEX(0, 0), COMPLEX(10, 10));
    /*
    std::cout << std::endl;
    for (const auto& v : V) {
        auto tmp = multiply_matrix_on_vector(A, v);
        show_vector(tmp);
    }

    for (const auto& v1: V) {
        for (const auto& v2: V) {
            std::cout << scalar_product(v1, v2) << std::endl;
        }
    }
    */
    /*
    std::vector<double> time_vec = {10, 20, 30, 40, 50, 60, 70 ,80 ,90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200};
    auto probs = Evolution::evolution(state, &H, time_vec);
    size_t index = 0;
    for (const auto& b: bases) {
        std::cout << std::setw(config::WIDTH) << b.to_string() << " : ";
        for (size_t i = 0; i < time_vec.size(); i++) {
            std::cout << std::setw(config::WIDTH) << probs[index][i] << " ";
        }
        std::cout << std::endl;
        index++;
    }
    */
    //Dynamic dynamic(state, std::move(H), bases);
    return 0;
}

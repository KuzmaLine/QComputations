#pragma once
#include <complex>
#include <functional>
#include <random>
#include <set>

#include "additional_operators.hpp"
#include "config.hpp"
#include "functions.hpp"
#include "matrix.hpp"
#include "state.hpp"

#ifdef ENABLE_MPI
#ifdef ENABLE_CLUSTER

#include "blocked_matrix.hpp"
#include "mpi_functions.hpp"

#endif
#endif

namespace QComputations {

    namespace matrix {
        template <typename T>
        Matrix<T> create_rand_matrix(size_t n, size_t m, T a, T b) {
            Matrix<T> A(C_STYLE, n, m);

            std::random_device rd;
            std::mt19937 re(rd());
            std::uniform_real_distribution<T> random(a, b);
            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < m; j++) {
                    A[i][j] = random(re);
                }
            }

            return A;
        }

        template <typename T>
        Matrix<T> create_rand_complex_matrix(size_t n, size_t m, T a, T b) {
            Matrix<T> A(C_STYLE, n, m);

            std::random_device rd;
            std::mt19937 re(rd());
            std::uniform_real_distribution<double> random_real(a.real(), b.real());
            std::uniform_real_distribution<double> random_imag(a.imag(), b.imag());
            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < m; j++) {
                    double rand_a = random_real(re);
                    double rand_b = random_imag(re);
                    A[i][j] = T(rand_a, rand_b);
                }
            }

            return A;
        }

        template <typename T>
        Matrix<std::complex<T>> create_hermit_rand_matrix(size_t n, size_t m, std::complex<T> a, std::complex<T> b) {
            Matrix<std::complex<T>> A(C_STYLE, n, m);

            std::random_device rd;
            std::mt19937 re(rd());
            std::uniform_real_distribution<T> random_real(a.real(), b.real());
            std::uniform_real_distribution<T> random_imag(a.imag(), b.imag());
            for (size_t i = 0; i < n; i++) {
                for (size_t j = i; j < m; j++) {
                    T rand_a = random_real(re);
                    T rand_b = random_imag(re);
                    A[i][j] = std::complex<T>(rand_a, rand_b);
                    A[j][i] = std::conj(A[i][j]);

                    if (i == j) A[i][i] = rand_a;
                }
            }

            return A;
        }
    }  // namespace matrix

    namespace printing {
        template <typename T>
        void binary_print(T num) {
            T mask = T(0);
            size_t bits_count = 32;
            if (typeid(T) == typeid(long)) bits_count = 64;

            mask = T(1) << (bits_count - 1);

            //std::cout << "Printing: " << mask << std::endl;
            for (size_t i = 0; i < bits_count; i++) {
                //std::cout << mask << " " << (num & mask) << std::endl;
                std::cout << (((num & mask) == T(0)) ? 0 : 1);
                mask >>= 1;
            }

            std::cout << std::endl;
        }

        void probs_print(const Matrix<double>& probs,
                         const std::set<Basis_State>& basis,
                         const std::vector<double>& time_vec) {
            size_t index = 0;
            for (const auto& b : basis) {
                std::cout << std::setw(QConfig::instance().width()) << b.to_string() << " : ";
                for (size_t i = 0; i < time_vec.size(); i++) {
                    std::cout << std::setw(QConfig::instance().width()) << probs[index][i] << " ";
                }
                std::cout << std::endl;
                index++;
            }
        }

#ifdef ENABLE_MPI
#ifdef ENABLE_CLUSTER
        void probs_print(const BLOCKED_Matrix<double>& probs,
                         const std::set<Basis_State>& basis,
                         const std::vector<double>& time_vec) {
            ILP_TYPE rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            size_t index = 0;
            for (const auto& b : basis) {
                if (rank == mpi::ROOT_ID) std::cout << std::setw(QConfig::instance().width()) << b.to_string() << " : ";
                for (size_t i = 0; i < time_vec.size(); i++) {
                    auto elem = probs.get(index, i);
                    if (rank == mpi::ROOT_ID)
                        std::cout << std::setw(QConfig::instance().width()) << probs.get(index, i) << " ";
                }
                std::cout << std::endl;
                index++;
            }
        }
#endif
#endif

    }  // namespace printing

    namespace probs_testing {
        void check_probs(const Matrix<double>& probs,
                         const std::set<TCH_State>& basis,
                         const std::vector<double>& time_vec,
                         double eps = QConfig::instance().eps()) {
            for (size_t i = 0; i < time_vec.size(); i++) {
                size_t index = 0;
                double sum = 0;
                for (const auto& b : basis) {
                    //std::cout << std::setw(QConfig::instance().width()) << b.to_string() << " : ";
                    sum += probs[index++][i];
                    //std::cout << std::setw(QConfig::instance().width()) << probs[index][i] << " ";
                }

                if (std::abs(double(1) - sum) >= eps) {
                    std::cout << std::setw(QConfig::instance().width()) << time_vec[i]
                              << " INCORRECT PROBS! SUM = " << sum << std::endl;
                }
            }

            std::cout << std::endl;
        }

#ifdef ENABLE_MPI
#ifdef ENABLE_CLUSTER

        void check_probs(const BLOCKED_Matrix<double>& probs,
                         const std::set<TCH_State>& basis,
                         const std::vector<double>& time_vec,
                         double eps = QConfig::instance().eps()) {
            ILP_TYPE rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            for (size_t i = 0; i < time_vec.size(); i++) {
                size_t index = 0;
                double sum = 0;
                for (const auto& b : basis) {
                    //std::cout << std::setw(QConfig::instance().width()) << b.to_string() << " : ";
                    auto elem = probs.get(index++, i);
                    sum += elem;
                    //std::cout << std::setw(QConfig::instance().width()) << probs[index][i] << " "
                }

                if (rank == mpi::ROOT_ID) {
                    if (std::abs(double(1) - sum) >= eps) {
                        std::cout << std::setw(QConfig::instance().width()) << time_vec[i]
                                  << " INCORRECT PROBS! SUM = " << sum << std::endl;
                    }
                }
            }

            if (rank == mpi::ROOT_ID) std::cout << std::endl;
        }

#endif
#endif

    }  // namespace probs_testing
    namespace functions_testing {
        template <typename T, typename V>
        void check_runge_kutt(const std::vector<V>& X,
                              const T& y0,
                              std::function<T(V, T)> f,
                              std::function<T(V)> f_correct) {
            auto y = Runge_Kutt_4(X, y0, f);

            size_t index = 0;
            for (const auto& x : X) {
                std::cout << std::setw(QConfig::instance().width()) << x << " -> " << y[index++] << " | "
                          << f_correct(x) << std::endl;
            }
        }

        template <typename T>
        void check_pro_race(const Matrix<T>& A, const std::vector<T>& x, const std::vector<T>& b) {
            show_vector(A * x);
            show_vector(b);
        }

        template <typename T>
        void check_eigenvectors(std::vector<double> eigenvalues,
                                const Matrix<T>& eigenvectors,
                                const Matrix<T>& matrix) {
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
    }  // namespace functions_testing

}  // namespace QComputations
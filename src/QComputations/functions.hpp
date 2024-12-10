#pragma once
#include <mkl_cblas.h>
#include <mkl_lapack.h>
#include <mkl_lapacke.h>

#include <chrono>
#include <complex>
#include <functional>
#include <set>
#include <vector>

#include "additional_operators.hpp"
#include "config.hpp"
#include "matrix.hpp"
#include "state.hpp"

#ifdef ENABLE_MPI
#include "mpi.h"
#endif

namespace QComputations {

    namespace {
#ifdef MKL_ILP64
        using ILP_TYPE = long long;
#else
        using ILP_TYPE = int;
#endif

        using E_LEVEL = int;
        using COMPLEX = std::complex<double>;
        using vec_complex = std::vector<COMPLEX>;
        using vec_levels = std::vector<E_LEVEL>;
    }  // namespace

    std::string to_string_complex_with_precision(const COMPLEX a_value, const int n, int max_number_size);

    std::string to_string_double_with_precision(const double a_value, const int n, int max_number_size);

    std::string vector_to_string(const std::vector<std::string>& inp);

    void print_state_biguint(const TCH_State& state);

    std::vector<double> FROM_double_TO_vector(double* A, lapack_int n);
    Matrix<COMPLEX> FROM_lapack_complex_double_TO_Matrix(lapack_complex_double* A, lapack_int n, lapack_int m);

    template <typename T>
    std::vector<double> linspace(T start_in, T end_in, int num_in) {
        std::vector<double> linspaced;
        double start = static_cast<double>(start_in);
        double end = static_cast<double>(end_in);
        double num = static_cast<double>(num_in);
        if (num == 0) {
            return linspaced;
        }
        if (num == 1) {
            linspaced.push_back(start);
            return linspaced;
        }
        double delta = (end - start) / (num - 1);
        for (int i = 0; i < num - 1; ++i) {
            linspaced.push_back(start + delta * i);
        }
        linspaced.push_back(end);  // I want to ensure that start and end
                                   // are exactly the same as the input
        return linspaced;
    }

    // Convert state to 10 numerical system
    size_t get_index_from_state(vec_levels state);

    bool is_zero(double a, double eps = QConfig::instance().eps());
    bool is_zero(COMPLEX a, double eps = QConfig::instance().eps());
    bool is_digit(char c);
    size_t Ck_n(size_t k, size_t n);

    // Eigen Problem
    double off(const Matrix<double>& A);
    std::pair<double, double> givens(double a, double b, double eps = QConfig::instance().eps());
    void tridiagonal_QR(Matrix<double>& T);
    Matrix<double> MGS(const Matrix<COMPLEX>& A);

    // Using for Hamiltonians
    std::pair<std::vector<double>, Matrix<COMPLEX>> Hermit_Lanczos(const Matrix<COMPLEX>& A);

    std::function<double(double)> Cubic_Spline_Interpolate(const std::vector<double>& x, const std::vector<double>& y);

    // Solve f(x) = target
    double fsolve(std::function<double(double)> f,
                  double a,
                  double b,
                  double target = 0,
                  double eps = QConfig::instance().eps());

    // f must be unimodal on [a, b]
    double fmin(std::function<double(double)> f, double a, double b, double eps = QConfig::instance().eps());

    template <typename StateType>
    void show_basis(const BasisType<StateType>& basis) {
        for (const auto& state : basis) {
            std::cout << std::setw(QConfig::instance().width()) << state->to_string() << " ";
        }

        std::cout << std::endl;
    }

    // <a|b>, (b, a)
    double scalar_product(const std::vector<double>& a, const std::vector<double>& b);
    COMPLEX scalar_product(const std::vector<COMPLEX>& a, const std::vector<COMPLEX>& b);

    // Euclidean
    double norm(const std::vector<COMPLEX>& v);

    void make_rank_map(size_t size, int rank, int world_size, size_t& start_col, size_t& count);

    // ------------------------------- template functions --------------------------------------

    template <typename T>
    std::set<T> set_bool_check(const std::set<T>& v, const std::function<bool(T)>& func) {
        std::set<T> res;

        for (const auto& x : v) {
            if (func(x)) {
                res.insert(x);
            }
        }

        return res;
    }

    template <typename T>
    T read_number(const std::string& str, size_t& start_index) {
        T n = T(0);
        size_t index = start_index;

        while (is_digit(str[index])) {
            n *= 10;
            n += str.at(index) - '0';
            index++;
        }

        start_index = index;
        return n;
    }

    template <typename T>
    Matrix<T> tensor_multiply(const Matrix<T>& A, const Matrix<T>& B) {
        auto n = A.size() * B.size();
        Matrix<T> C(C_STYLE, n, n, 0);

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

    template <typename T>
    std::ostream& operator<<(std::ostream& out, const Matrix<T>& matrix) {
        for (size_t i = 0; i < matrix.n(); i++) {
            for (size_t j = 0; j < matrix.m(); j++) {
                out << std::setw(QConfig::instance().width()) << matrix[i][j] << " ";
            }

            out << std::endl;
        }

        return out;
    }

    // Get elem from set by index
    template <typename T>
    T get_elem_from_set(const std::set<T>& st, size_t index) {
        auto it = st.begin();
        std::advance(it, index);
        return *it;
    }

    template <typename T>
    std::shared_ptr<T> get_state_from_basis(const BasisType<T>& st, size_t index) {
        auto it = st.begin();
        std::advance(it, index);
        return *it;
    }

    template <typename T, typename V>
    std::vector<V> Runge_Kutt_4(const std::vector<T>& x, const V& y0, std::function<V(T, V)> f) {
        size_t len = x.size();
        std::vector<V> y(len);
        y[0] = y0;

        for (size_t i = 0; i < len - 1; i++) {
            T h = x[i + 1] - x[i];

            V k1 = f(x[i], y[i]);
            V k2 = f(x[i] + h / 2.0, y[i] + k1 * (h / 2.0));
            V k3 = f(x[i] + h / 2.0, y[i] + k2 * (h / 2.0));
            V k4 = f(x[i] + h, y[i] + k3 * h);
            y[i + 1] = y[i] + (k1 + (k2 + k3) * 2 + k4) * (h / 6.0);
        }

        return y;
    }

    template <typename T, typename V>
    std::vector<V> Runge_Kutt_2(const std::vector<T>& x, const V& y0, std::function<V(T, V)> f) {
        size_t len = x.size();
        std::vector<V> y(len);
        y[0] = y0;

        for (size_t i = 0; i < len - 1; i++) {
            T h = x[i + 1] - x[i];

            V k1 = f(x[i], y[i]);
            V k2 = f(x[i] + h, y[i] + k1 * h);
            y[i + 1] = y[i] + (k1 + k2) * (h / 2.0);
        }

        return y;
    }

    template <typename T>
    bool is_in_vector(const std::vector<T> v, const T& elem) {
        return std::find(v.begin(), v.end(), elem) != v.end();
    }

    // Tridiagonal linear system solving
    template <typename T>
    std::vector<T> Thomas_Algorithm(const Matrix<T>& B, const std::vector<T>& y) {
        size_t k = B.size();

        // Down move
        std::vector<T> x(k, 0);
        std::vector<T> a(k, 0);
        a[0] = -B[0][1] / B[0][0];

        std::vector<T> b(k, 0);
        b[0] = y[0] / B[0][0];

        for (size_t i = 1; i < k - 1; i++) {
            a[i] = B[i][i + 1] / (-B[i][i] - B[i][i - 1] * a[i - 1]);
            b[i] = (B[i][i - 1] * b[i - 1] - y[i]) / (-B[i][i] - a[i - 1] * B[i][i - 1]);
        }
        a[k - 1] = 0;
        x[k - 1] = (B[k - 1][k - 2] * b[k - 2] - y[k - 1]) / (-B[k - 1][k - 1] - B[k - 1][k - 2] * a[k - 2]);

        // Up move

        for (long long i = k - 2; i > -1; i--) {
            x[i] = a[i] * x[i + 1] + b[i];
        }

        return x;
    }

    // Testing function
    template <typename T>
    void show_vector(const std::vector<T>& v) {
        for (const auto& num : v) {
            std::cout << std::setw(15) << num << " ";
        }

        std::cout << std::endl;
    }

    template <typename StateType>
    BasisType<Basis_State> convert_to(const BasisType<StateType>& states) {
        BasisType<Basis_State> res;
        for (auto st : states) {
            res.insert(st);
        }

        return res;
    }

    // from {x1, x2, x3 ...} to {f(x1), f(x2), ...}
    template <typename T>
    std::vector<T> f_vector(std::function<T(T)> f, const std::vector<T>& x) {
        size_t n = x.size();
        std::vector<T> res(n);

        for (size_t i = 0; i < n; i++) {
            res[i] = f(x[i]);
        }

        return res;
    }

    // Не трогать!!!!!
    void cblas_MM_double_complex(COMPLEX* A, COMPLEX* B, COMPLEX* C, int n, int k, int m, double alpha, double betta);

    void cblas_MM_double(double* A, double* B, double* C, int n, int k, int m, double alpha, double betta);

    void cblas_MM_int(int* A, int* B, int* C, int n, int k, int m, double alpha, double betta);

#ifdef ENABLE_MPI

#endif

}  // namespace QComputations
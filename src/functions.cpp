#include "functions.hpp"
#include <algorithm>
#include <numeric>
#include <iostream>
#include "config.hpp"
#include <cassert>

namespace {
    constexpr double eps = 1e-12;
    const double EPS = eps;

    void BP_Mult(Matrix<double>& B, size_t p, size_t q, double c, double s) {
        size_t n = B.size();
        for (size_t i = 0; i < n; i++) {
            auto B_p = B[i][p] * c - B[i][q] * s;
            auto B_q = B[i][p] * s + B[i][q] * c;

            B[i][p] = B_p;
            B[i][q] = B_q;
        }
    }

    void PtransB_Mult(Matrix<double>& B, size_t p, size_t q, double c, double s) {
        size_t n = B.size();
        for (size_t i = 0; i < n; i++) {
            auto B_p = B[p][i] * c - B[q][i] * s;
            auto B_q = B[p][i] * s + B[q][i] * c;

            B[p][i] = B_p;
            B[q][i] = B_q;
        }
    }
}

std::vector<double> make_timeline(double start, double end, double step) {
    size_t n = (end - start) / step;
    std::vector<double> timeline(n + 1);

    auto cur_time = start;
    for (size_t i = 0; i <= n; i++) {
        timeline[i] = start + step * i;
    }

    return timeline;
}

double scalar_product(const std::vector<double>& a, const std::vector<double>& b) {
    double res = 0;

    for (size_t i = 0; i < a.size(); i++) {
        res += a[i] * b[i];
    }

    return res;
}

COMPLEX scalar_product(const std::vector<COMPLEX>& a, const std::vector<COMPLEX>& b) {
    COMPLEX res = 0;

    for (size_t i = 0; i < a.size(); i++) {
        COMPLEX tmp(b[i].real(), -(b[i].imag()));

        res += a[i] * tmp;
    }

    return res;
}


double norm(const std::vector<COMPLEX>& v) {
    double res = 0;

    for (const auto& num: v) {
        auto tmp = std::abs(num);
        res += tmp * tmp;
    }

    return std::sqrt(res);
}

double off(const Matrix<double>& A) {
    double res = 0;

    size_t n = A.n();
    assert(A.n() == A.m());

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            if (i != j) {
                res += A[i][j] * A[i][j];
            }
        }
    }

    res = std::sqrt(res);
    return res;
}

size_t get_index_from_state(vec_levels state) {
    size_t index = 0;
    for (const auto qubit: state) {
        index <<= 1;
        index += qubit;
    }

    return index;
}

/*
matrix transpose(const matrix& A) {
    matrix R(A.size(), std::vector<COMPLEX>(A.size()));

    for (size_t i = 0; i < A.size(); i++) {
        for (size_t j = 0; j < A.size(); j++) {
            R[i][j] = A[j][i];
        }
    }

    return R;
}

matrix hermit(const matrix& A) {
    size_t n = A.size(), m = A[0].size();
    matrix B(m, std::vector<std::complex<double>>(n));

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            B[j][i] = std::conj(A[i][j]);
        }
    }
    return B;
}

*/

// функция, проверяющая равенство двух комплексных чисел
bool equals(std::complex<double> a, std::complex<double> b) {
    return abs(a - b) < EPS;
}
/*
matrix multiply(const matrix& A, const matrix& B) {
    int n1 = A.size();
    int m1 = A[0].size();
    int n2 = B.size();
    int m2 = B[0].size();
    if (m1 != n2) {
        throw std::runtime_error("Multiplication error: matrices have incorrect dimensions.");
    }
    matrix C(n1, std::vector<std::complex<double>>(m2, 0.0));
    for (size_t i = 0; i < n1; ++i) {
        for (size_t j = 0; j < m2; ++j) {
            for (size_t k = 0; k < m1; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}
*/
std::pair<double, double> givens(double a, double b) {
    if (std::abs(b) < eps) return std::make_pair(1, 0);

    double c, s;

    if (std::abs(b) > std::abs(a)) {
        double t = -a / b;
        s = 1 / std::sqrt(1 + t * t);
        c = s * t;
    } else {
        double t = -b / a;
        c = 1 / std::sqrt(1 + t * t);
        s = c * t;
    }

    return std::make_pair(c, s);
}

/*
std::vector<double> divide_and_conquer(const std::vector<std::vector<double>>& A) {
    size_t n = A.size();
    size_t m = n / 2;
    auto T = A;


}
*/

void tridiagonal_QR(Matrix<double>& T) {
    size_t n = T.size();
    for (size_t k = 0; k < n - 1; k++) {
        auto p = givens(T[k][k], T[k + 1][k]);
        double c = p.first;
        double s = p.second;

        double m = std::min(k + 2, n - 1);
        Matrix<double> tmp(2, 3, 0);
        for (size_t i = k; i <= m; i++) {
            tmp[0][i - k] = T[k][i] * c - s * T[k + 1][i];
            tmp[1][i - k] = T[k][i] * s + c * T[k + 1][i];
        }

        for (size_t i = k; i <= m; i++) {
            T[k][i] = tmp[0][i - k];
            T[k + 1][i] = tmp[1][i - k];
        }
        /*
        auto t11 = T[k][k] * c - s * T[k + 1][k];
        auto t12 = T[k][k + 1] * c - s * T[k + 1][k + 1];
        auto t21 = T[k][k] * s + c * T[k + 1][k];
        auto t22 = T[k][k + 1] * s + c * T[k + 1][k + 1];

        T[k][k] = t11;
        T[k][k + 1] = t12;
        T[k + 1][k] = t21;
        T[k + 1][k + 1] = t22;
        */
    }
}

Matrix<double> MGS (const Matrix<COMPLEX>& A) {
    auto m = A.size();
    //std::cout << "HERE 1\n";
    Matrix<COMPLEX> v(m, m, 0);
    v[0][0] = COMPLEX(1);

    Matrix<double> H(m, m, 0);

    //std::cout << "HERE 2\n";
    for (size_t j = 0; j < m; j++) {
        //std::cout << "J = " << j << std::endl;
        //auto w = multiply_matrix_on_vector(A, v[j]);
        //show_vector(v.row(j));
        std::vector<COMPLEX> w = A * v.row(j);

        for (int i = 0; i <= j; i++) {
            //std::cout << "I = " << i << std::endl;
            //show_vector(w);
            //show_vector(v[i]);
            if (i >= j - 1) {
                H[i][j] = scalar_product(w, v.row(i)).real();
            }
            //std::cout << std::setw(15) << H[i][j] << std::endl;
            //show_vector(multiply_vector_on_number(v[i], H[i][j]));
            w = w - (v.row(i) * std::complex<double>(H[i][j]));
            //show_vector(w);
        }

        if (norm(w) < EPS)  {
            //std::cout << "EPS\n";
            return H;
        }

        if (j != m - 1) {
            H[j + 1][j] = norm(w);
            //std::cout << std::setw(15) << H[j + 1][j] << std::endl;
            v.modify_row(j + 1, w / COMPLEX(H[j + 1][j]));
            //show_vector(v[j + 1]);
        }
    }

    return H;
}

// ONLY FOR REAL MATRIX. COMPLEX FOR Hermit_Lanczos
std::pair<std::vector<double>, Matrix<double>> jacobi(const Matrix<double>& A) {
    using namespace std;
    int n = A.size();
    Matrix<double> eigenvectors(n, n);
    vector<double> eigenvalues(n);

    auto B = A;
    // начальное приближение для собственных векторов
    for (int i = 0; i < n; i++) {
        eigenvectors[i][i] = 1;
    }

    size_t max_iter = 100000;
    size_t iter = 0;
    // алгоритм Якоби для эрмитовых матриц
    while (true) {
        //std::cout << iter << std::endl;
        //if (iter == max_iter) break;
        iter++;

        if (iter == max_iter) break;
        if (iter % 1000 == 0) {
            if (off(B) < EPS) break;
        }
        // находим максимальный недиагональный элемент
        double max_element = 0;
        int p = 0, q = 0;
        for (int i = 0; i < n-1; i++) {
            for (int j = i + 1; j < n; j++) {
                double element = abs(B[i][j]);
                if (element > max_element) {
                    max_element = element;
                    p = i;
                    q = j;
                }
            }
        }
        if (max_element < EPS) break;

        // вычисляем угол поворота
        //double theta = arg(B[p][q] / (B[q][q] - B[p][p])) / 2.0;
        double theta = atan2(2.0 * B[p][q], B[q][q] - B[p][p]) / 2.0;

        // создаем матрицу поворота
        //Matrix<double> P(n, n);
        //for (int i = 0; i < n; i++) {
        //    P[i][i] = 1;
        //}
        //P[p][p] = P[q][q] = cos(theta);
        //P[q][p] = -sin(theta);
        //P[p][q] = sin(theta);
        double c = cos(theta);
        double s = sin(theta);
        // обновляем матрицу
        //B = multiply(multiply(P.transpose(), B), P);
        //B = P.transpose() * B * P;
        PtransB_Mult(B, p, q, c, s);
        BP_Mult(B, p, q, c, s);
        BP_Mult(eigenvectors, p, q, c, s);
        //eigenvectors = eigenvectors * P;
        //eigenvectors = multiply(eigenvectors, P);
    }

    for (int i = 0; i < n; i++) {
        eigenvalues[i] = B[i][i];
    }

    return make_pair(eigenvalues, eigenvectors);
}

//std::vector<double> make_timeline(double start, double end, double step, double multiplyer) {
    //double
//}
//std::pair<std::vector<double>, Matrix<double>> Hermit_Lanczos(const Matrix<COMPLEX>& A) {
std::pair<std::vector<double>, Matrix<COMPLEX>> Hermit_Lanczos(const Matrix<COMPLEX>& A) {
    auto m = A.size();
    std::vector<double> alpha(m, 0);
    std::vector<double> betta(m, 0);

    Matrix<COMPLEX> v(m, m, 0);
    v[0][0] = COMPLEX(1);

    /*
    for (size_t j = 0; j < m; j++) {
        //std::cout << j << std::endl;
        std::vector<COMPLEX> w = A * v.col(j);
        alpha[j] = scalar_product(w, v.col(j)).real();
        w = w - (v.col(j) * COMPLEX(alpha[j]));
        if (j != 0) {
            //w = w - (v.col(j - 1) * COMPLEX(betta[j]));
            w = w - (v.col(j - 1) * COMPLEX(betta[j - 1]));
        }

        std::vector<COMPLEX> sum_w(m, 0);
        for (size_t i = 0; i < j; i++) {
            COMPLEX product = scalar_product(w, v.col(i));
            auto v_tmp = v.col(i);
            for (size_t k = 0; k < m; k++) {
                sum_w[k] += v_tmp[k] * product;
            }
        }

        w = w - sum_w;

        betta[j] = norm(w);

        if (betta[j] < EPS) {
            std::cout << "EPS!\n";
            break;
        }
        //std::cout << scalar_product(w, v.col(j)) << std::endl;    
        if (j != m - 1) {
            v.modify_col(j + 1, w / COMPLEX(betta[j]));
        }
    }
    */

    std::vector<COMPLEX> w = A * v.col(0);
    alpha[0] = scalar_product(w, v.col(0)).real();
    w = w - (v.col(0) * COMPLEX(alpha[0]));
    for (size_t j = 1; j < m; j++) {
        betta[j] = norm(w);

        if (betta[j] >= EPS) {
            v.modify_col(j, w / COMPLEX(betta[j]));
        } else {
            std::vector<COMPLEX> v_unit(m, 0);
            v_unit[j] = 1;

            std::cout << "HERE\n";
            for (size_t i = 0; i < j; i++) {
                COMPLEX a = scalar_product(v_unit, v.col(i));
                v_unit = v_unit - v.col(i) * a;
                show_vector(v_unit);
            }

            v_unit = v_unit / COMPLEX(norm(v_unit));


            v.modify_col(j, v_unit);
        }
        
        w = A * v.col(j);
        alpha[j] = scalar_product(w, v.col(j)).real();
        w = w - v.col(j) * COMPLEX(alpha[j]) - v.col(j - 1) * COMPLEX(betta[j]);
    }

    /*
    for (const auto& num: alpha) {
        std::cout << num << " ";
    }

    std::cout << std::endl;
    for (const auto& num: betta) {
        std::cout << num << " ";
    }
    */

    Matrix<double> H(m, m, 0);
    H[0][0] = alpha[0];

    for (size_t i = 1; i < m; i++) {
        H[i][i] = alpha[i];
        H[i - 1][i] = betta[i];
        H[i][i - 1] = betta[i];
    }

    //std::cout << std::endl;
    H.show(config::WIDTH);
    //tridiagonal_QR(H);

    //std::cout << std::endl;
    //return jacobi(H);
    auto p = jacobi(H);

    //std::cout << "Here\n";
    std::vector<double> eigenvalues = p.first;
    Matrix<double> T_eigenvectors = p.second;

    Matrix<COMPLEX> eigenvectors = v * T_eigenvectors;

    auto check = v.hermit() * A * v;
    check.show();

    return std::make_pair(eigenvalues, eigenvectors);
}

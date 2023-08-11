#include "functions.hpp"
//#include "test.hpp"
#include <algorithm>
#include <numeric>
#include <iostream>
#include "config.hpp"
#include <cassert>

extern "C"
{
    void zdotc(COMPLEX*, int*, const COMPLEX*, int*, const COMPLEX*, int*);
}

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

bool is_zero(double a) {
    return std::abs(a) < eps;
}

bool is_zero(COMPLEX a) {
    return std::abs(a) < eps;
}

bool is_digit(char c) {
    return '0' <= c and c <= '9';
}

void show_basis(const std::set<Cavity_State>& basis) {
    for (const auto& state: basis) {
        std::cout << std::setw(config::WIDTH) << state.to_string() << " ";
    }

    std::cout << std::endl;
}

void show_basis(const std::set<State>& basis) {
    for (const auto& state: basis) {
        std::cout << std::setw(config::WIDTH) << state.to_string() << " ";
    }

    std::cout << std::endl;
}

std::vector<size_t> make_rank_map(size_t size, int rank, int world_size, size_t& start_col) {
    size_t size_per_proc = size / world_size;
    std::vector<size_t> rank_map(world_size, size_per_proc);

    size_t rest = size - size_per_proc * world_size;
    start_col = 0;
    for (size_t i = 0; i < world_size; i++) {
        if (i < rest) {
            rank_map[i]++;
        }

        if (i < rank) {
            start_col += rank_map[i];
        }
    }

    return rank_map;
}

std::set<State> Cavity_State_to_State(const std::set<Cavity_State>& st) {
    std::set<State> res;

    for (const auto& item: st) {
        res.insert(State(item));
    }

    return res;
}

std::function<double(double)> Cubic_Spline_Interpolate(const std::vector<double>& x, const std::vector<double>& f) {
    auto begin = std::chrono::steady_clock::now();
    size_t n = x.size() - 1;

    std::vector<double> h(n + 1);
    h[0] = 0;
    for (size_t i = 1; i <= n; i++) {
        h[i] = x[i] - x[i - 1];
    }

    Matrix<double> C(n + 1, n + 1);
    C[0][0] = 1;
    C[0][1] = 0;
    C[n][n] = 1;
    C[n][n - 1] = 0;

    for (size_t i = 1; i < n; i++) {
        C[i][i - 1] = h[i] / 6.0;
        C[i][i] = (h[i] + h[i + 1]) / 3.0;
        C[i][i + 1] = h[i + 1] / 6.0;
    }

    std::vector<double> y(n + 1);
    y[0] = y[n] = 0;

    for (size_t i = 1; i < n; i++) {
        y[i] =(f[i + 1] - f[i]) / h[i + 1] - (f[i] - f[i - 1]) / h[i];
    }

    auto c = Pro_Race_Algorithm(C, y);

    std::vector<double> a(n + 1);
    std::vector<double> b(n + 1);
    std::vector<double> d(n + 1);
    std::vector<std::function<double(double)>> S(n + 1);
    a[0] = f[0];
    for (size_t i = 1; i <= n; i++) {
        a[i] = f[i];
        d[i] = (c[i] - c[i - 1]) / h[i];
        b[i] = (a[i] - a[i - 1]) / h[i] + h[i] / 2.0 * c[i] - h[i] * h[i] / 6.0 * d[i];
    }

    /*
    for (size_t i = 1; i <= n; i++) {
        S[i] = std::function<double(double)> {
            [a, b, c, d, x, i](double t) {
                auto diff = t - x[i];
                return a[i] + b[i] * diff + c[i] * diff * diff / 2.0 + d[i] * diff * diff * diff / 6.0; 
            }
        };
    }
    */
    std::function<double(double)> res {
        [f, x, a, b, c, d](double t) {
            if (t + config::eps < x[0] or t - config::eps > x[x.size() - 1]) {
                std::cerr << "Not between x[0] and x[n - 1]" << std::endl;
                return -1.0;
            }

            if (is_zero(t - x[0])) { 
                return f[0];
            }
            if (is_zero(t - x[x.size() - 1])) { 
                return f[x.size() - 1];
            }
            for (size_t i = 1; i < x.size(); i++) {
                if (x[i - 1] <= t and t <= x[i]) {
                    auto diff = t - x[i];
                    return a[i] + b[i] * diff + c[i] * diff * diff / 2.0 + d[i] * diff * diff * diff / 6.0;
                }
            }

            std::cerr << "Not between x[0] and x[n - 1]" << std::endl;
            return -1.0;
        }
    };
    auto end = std::chrono::steady_clock::now();
    return res;
}

size_t Ck_n(size_t k, size_t n) {
    std::vector<size_t> line(n + 1);
    std::vector<size_t> buf(n + 1);
    line[0] = 1;
    for (size_t i = 1; i <= n; i++) {
        for (size_t k = 0; k <= i; k++) {
            if (k == 0 or k == i) {
                buf[k] = 1;
            } else {
                buf[k] = line[k - 1] + line[k];
            }
        }

        line = buf;
    }

    return line[k];
}

double fsolve(std::function<double(double)> f, double a, double b, double target, double eps) {
    double t = (a + b) / 2.0;

    while(std::abs(f(t) - target) >= eps) {
        if (f(t) - target > 0) {
            b = t;
        } else {
            a = t;
        }

        t = (a + b) / 2.0;

        if (std::abs(a - b) < eps) { std::cerr << "f without zero" << std::endl; return b; }
    }

    return t;
}

// https://group112.github.io/doc/sem2/2019/2019_sem2_lesson3.pdf page 1
double fmin(std::function<double(double)> f, double a, double b, double eps) {
    double a_n = a;
    double b_n = b;
    double l_n = b - a;
    double x_n = (a + b) / 2.0;

    while (l_n >= 2 * eps) {
        x_n = (a_n + b_n) / 2.0;
        double c_n = x_n - eps / 2.0;
        double d_n = x_n + eps / 2.0;

        if (f(c_n) < f(d_n)) {
            b_n = d_n;
        } else {
            a_n = c_n;
        }

        l_n = b_n - a_n;
    }

    return x_n;
}

std::vector<double> FROM_double_TO_vector(double* A, lapack_int n) {
    std::vector<double> res(n);

    for (size_t i = 0; i < n; i++) {
        res[i] = A[i];
    }

    delete [] A;
    return res;
}

Matrix<COMPLEX> FROM_lapack_complex_double_TO_Matrix(lapack_complex_double* A, lapack_int n, lapack_int m) {
    Matrix<COMPLEX> res(n, m);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            COMPLEX tmp(A[i * m + j].real, A[i * m + j].imag);
            res[i][j] = tmp;
        }
    }

    delete [] A;
    return res;
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

std::vector<double> linspace(double start, double end, double npoints) {
    return make_timeline(start, end, (end - start) / (npoints - 1));
}

double scalar_product(const std::vector<double>& a, const std::vector<double>& b) {
    return cblas_ddot(a.size(), a.data(), 1, b.data(), 1);
    
    /*
    double res = 0;

    for (size_t i = 0; i < a.size(); i++) {
        res += a[i] * b[i];
    }

    return res;
    */
}

COMPLEX scalar_product(const std::vector<COMPLEX>& a, const std::vector<COMPLEX>& b) {
    int size = a.size();
    int iONE = 1;
    COMPLEX res;
    zdotc(&res, &size, b.data(), &iONE, a.data(), &iONE);

    return res;
    /*
    COMPLEX res = 0;

    for (size_t i = 0; i < a.size(); i++) {
        COMPLEX tmp(b[i].real(), -(b[i].imag()));

        res += a[i] * tmp;
    }

    return res;
    */
}


double norm(const std::vector<COMPLEX>& v) {
    double res = 0;

    for (const auto& num: v) {
        auto tmp = std::abs(num);
        res += tmp * tmp;
    }

    return std::sqrt(res);
}

// For jacobi
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

// working????
void tridiagonal_QR(Matrix<double>& T) {
    size_t n = T.size();
    for (size_t k = 0; k < n - 1; k++) {
        auto p = givens(T[k][k], T[k + 1][k]);
        double c = p.first;
        double s = p.second;

        double m = std::min(k + 2, n - 1);
        Matrix<double> tmp(2, 3, double(0));
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

// Modified Gramm Schmidt
Matrix<double> MGS (const Matrix<COMPLEX>& A) {
    auto m = A.size();
    //std::cout << "HERE 1\n";
    Matrix<COMPLEX> v(m, m, 0);
    v[0][0] = COMPLEX(1);

    Matrix<double> H(m, m, double(0));

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

// ONLY FOR REAL MATRIX. FOR COMPLEX - Hermit_Lanczos
// Not effective - replace
std::pair<std::vector<double>, Matrix<double>> jacobi(const Matrix<double>& A) {
    int n = A.size();
    Matrix<double> eigenvectors(n, n);
    std::vector<double> eigenvalues(n);

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

// ADD FORTRAN SUPPORT
std::pair<std::vector<double>, Matrix<COMPLEX>> Hermit_Lanczos(const Matrix<COMPLEX>& A) {
    auto m = A.size();
    /*
    std::vector<double> alpha(m, 0);
    std::vector<double> betta(m, 0);

    Matrix<COMPLEX> v(m, m, 0);
    v[0][0] = COMPLEX(1);

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

    lapack_complex_double* lapack_A = A.to_upper_lapack();
    lapack_int n = A.size();
    lapack_int res;
    double *d, *e;

    auto B = A;
    auto lapack_B = B.to_lapack();
    d = new double [m];
    e = new double [n - 1];
    // reduce to tridiagonal form -> lapack_A
    res = LAPACKE_zhetrd(LAPACK_ROW_MAJOR, 'U', n, lapack_A, n, d, e, lapack_B);
    if (res != 0) std::cout << "LAPACKE_zhetrd error = " << res << std::endl;

    //find Q matrix for reducing matrix lapack_A -> lapack_A
    res = LAPACKE_zungtr(LAPACK_ROW_MAJOR, 'U', n, lapack_A, n, lapack_B);
    if (res != 0) std::cout << "LAPACKE_zungtr error = " << res << std::endl;

    //find eigenvalues and eigenvectors with Q matrix of matrix lapack_A -> d, lapack_A
    res = LAPACKE_zstedc(LAPACK_ROW_MAJOR, 'V', n, d, e, lapack_A, n); 
    if (res != 0) std::cout << "LAPACKE_zstedc error = " << res << std::endl;
    /*
    Matrix<double> H(m, m, 0);
    H[0][0] = d[0];

    for (size_t i = 1; i < m; i++) {
        H[i][i] = d[i];
        H[i - 1][i] = e[i - 1];
        H[i][i - 1] = e[i - 1];
    }
    //std::cout << std::endl;
    H.show(config::WIDTH);
    */


    //LAPACKE_dstedc(LAPACK_ROW_MAJOR, 'I', n, d, e, NULL, n);   
    /*
    for (const auto& num: alpha) {
        std::cout << num << " ";
    }

    std::cout << std::endl;
    for (const auto& num: betta) {
        std::cout << num << " ";
    }
    */

    /*
    Matrix<double> H(m, m, 0);
    H[0][0] = alpha[0];

    for (size_t i = 1; i < m; i++) {
        H[i][i] = alpha[i];
        H[i - 1][i] = betta[i];
        H[i][i - 1] = betta[i];
    }
    */

    //std::cout << std::endl;
    //return jacobi(H);
    //auto p = jacobi(H);

    //std::cout << "Here\n";
    //std::vector<double> eigenvalues = p.first;
    //Matrix<double> T_eigenvectors = p.second;

    //Matrix<COMPLEX> eigenvectors = v * T_eigenvectors;

    //auto check = v.hermit() * A * v;
    //check.show();

    auto eigenvectors = FROM_lapack_complex_double_TO_Matrix(lapack_A, n, n);
    auto eigenvalues = FROM_double_TO_vector(d, n);

    delete [] e;
    delete [] lapack_B;
    return std::make_pair(eigenvalues, eigenvectors);
}

void cblas_MM_double_complex(COMPLEX* A, COMPLEX* B, COMPLEX* C, int n, int k, int m, double alpha, double betta) {
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, m, k, &alpha, A,
                k, B, m, &betta,
                C, m);
}

void cblas_MM_double(double* A, double* B, double* C, int n, int k, int m, double alpha, double betta) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, m, k, alpha, A,
                k, B, m, betta,
                C, m);
}

void cblas_MM_int(int* A, int* B, int* C, int n, int k, int m, double alpha, double betta) {
}

#ifdef ENABLE_MPI

#endif
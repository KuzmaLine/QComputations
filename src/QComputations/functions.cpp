#include "functions.hpp"
//#include "test.hpp"
#include <algorithm>
#include <numeric>
#include <iostream>
#include "config.hpp"
#include <cassert>
#include <regex>

extern "C"
{
    void zdotc(COMPLEX*, int*, const COMPLEX*, int*, const COMPLEX*, int*);
}

namespace QComputations {

namespace {
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

std::string to_string_complex_with_precision(const COMPLEX a_value,
                                             const int n, int max_number_size) {
  std::ostringstream out;
  out.precision(n);
  out << ((a_value.real() >= 0) ? "+" : "-") << std::setfill('0')
      << std::setw(max_number_size) << std::fixed << std::abs(a_value.real());
  out << ((a_value.imag() >= 0) ? "+" : "-") << std::setfill('0')
      << std::setw(max_number_size) << std::fixed << std::abs(a_value.imag());
  out << "j";
  return std::move(out).str();
}

std::string to_string_double_with_precision(const double a_value,
                                             const int n, int max_number_size) {
  std::ostringstream out;
  out.precision(n);
  out << ((a_value >= 0) ? "+" : "-") << std::setfill('0')
      << std::setw(max_number_size) << std::fixed << std::abs(a_value);
  return std::move(out).str();
}

std::string vector_to_string(const std::vector<std::string>& inp) {
  std::ostringstream out;
  for (auto i : inp) {
    if (i.empty())
      break;
    out << i << '\n';
  }
  return std::move(out).str();
}

bool is_zero(double a, double eps) {
    return std::abs(a) < eps;
}

bool is_zero(COMPLEX a, double eps) {
    return std::abs(a) < eps;
}

bool is_digit(char c) {
    return '0' <= c and c <= '9';
}

std::string make_state_regex_pattern(const std::string& format, bool is_freq_display, bool is_sequence) {
    std::regex format_regex("($[N,W,M])");

    auto regex_begin = std::sregex_iterator(format.begin(), format.end(), format_regex);
    auto regex_end = std::sregex_iterator();

    for (std::sregex_iterator i = regex_begin; i != regex_end; ++i) {
        std::smatch match = *i;
        std::cout << match.str() << std::endl;
    }

    return regex_begin->str();
}

void make_rank_map(size_t size, int rank, int world_size, size_t& start_col, size_t& count) {
    size_t size_per_proc = size / world_size;
    count = size_per_proc;

    int rest = size - size_per_proc * world_size;
    start_col = 0;

    if (rank < rest) {
        count++;
    }

    start_col = rank * size_per_proc + std::min(rank, rest);
}

std::function<double(double)> Cubic_Spline_Interpolate(const std::vector<double>& x, const std::vector<double>& f) {
    auto begin = std::chrono::steady_clock::now();
    size_t n = x.size() - 1;

    std::vector<double> h(n + 1);
    h[0] = 0;
    for (size_t i = 1; i <= n; i++) {
        h[i] = x[i] - x[i - 1];
    }

    Matrix<double> C(C_STYLE, n + 1, n + 1);
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

    auto c = Thomas_Algorithm(C, y);

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

    std::function<double(double)> res {
        [f, x, a, b, c, d](double t) {
            if (t + QConfig::instance().eps() < x[0] or t - QConfig::instance().eps() > x[x.size() - 1]) {
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
    Matrix<COMPLEX> res(C_STYLE, n, m);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            COMPLEX tmp(A[i * m + j].real(), A[i * m + j].imag());
            res[i][j] = tmp;
        }
    }

    delete [] A;
    return res;
}


double scalar_product(const std::vector<double>& a, const std::vector<double>& b) {
    return cblas_ddot(a.size(), a.data(), 1, b.data(), 1);
}

COMPLEX scalar_product(const std::vector<COMPLEX>& a, const std::vector<COMPLEX>& b) {
    int size = a.size();
    int iONE = 1;
    COMPLEX res;
    zdotc(&res, &size, a.data(), &iONE, b.data(), &iONE);

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


std::pair<double, double> givens(double a, double b, double eps) {
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

// working????
void tridiagonal_QR(Matrix<double>& T) {
    size_t n = T.size();
    for (size_t k = 0; k < n - 1; k++) {
        auto p = givens(T[k][k], T[k + 1][k]);
        double c = p.first;
        double s = p.second;

        double m = std::min(k + 2, n - 1);
        Matrix<double> tmp(C_STYLE, 2, 3, double(0));
        for (size_t i = k; i <= m; i++) {
            tmp[0][i - k] = T[k][i] * c - s * T[k + 1][i];
            tmp[1][i - k] = T[k][i] * s + c * T[k + 1][i];
        }

        for (size_t i = k; i <= m; i++) {
            T[k][i] = tmp[0][i - k];
            T[k + 1][i] = tmp[1][i - k];
        }
    }
}

// Modified Gramm Schmidt
Matrix<double> MGS (const Matrix<COMPLEX>& A, double eps = QConfig::instance().eps()) {
    auto m = A.size();
    Matrix<COMPLEX> v(C_STYLE, m, m, COMPLEX(0));
    v[0][0] = COMPLEX(1);

    Matrix<double> H(C_STYLE, m, m, double(0));

    for (size_t j = 0; j < m; j++) {
        std::vector<COMPLEX> w = A * v.row(j);

        for (int i = 0; i <= j; i++) {
            if (i >= j - 1) {
                H[i][j] = scalar_product(w, v.row(i)).real();
            }

            w = w - (v.row(i) * std::complex<double>(H[i][j]));
        }

        if (norm(w) < eps)  {
            return H;
        }

        if (j != m - 1) {
            H[j + 1][j] = norm(w);
            v.modify_row(j + 1, w / COMPLEX(H[j + 1][j]));
        }
    }

    return H;
}

// ONLY FOR REAL MATRIX. FOR COMPLEX - Hermit_Lanczos
// Not effective - replace
std::pair<std::vector<double>, Matrix<double>> jacobi(const Matrix<double>& A, double eps = QConfig::instance().eps()) {
    int n = A.size();
    Matrix<double> eigenvectors(C_STYLE, n, n);
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
        iter++;

        if (iter == max_iter) break;
        if (iter % 1000 == 0) {
            if (off(B) < eps) break;
        }
        // находим максимальный недиагональный элемент
        double max_element = 0;
        int p = 0, q = 0;
        for (int i = 0; i < n-1; i++) {
            for (int j = i + 1; j < n; j++) {
                double element = std::abs(B[i][j]);
                if (element > max_element) {
                    max_element = element;
                    p = i;
                    q = j;
                }
            }
        }
        if (max_element < eps) break;

        // вычисляем угол поворота
        //double theta = arg(B[p][q] / (B[q][q] - B[p][p])) / 2.0;
        double theta = atan2(2.0 * B[p][q], B[q][q] - B[p][p]) / 2.0;

        double c = cos(theta);
        double s = sin(theta);
        PtransB_Mult(B, p, q, c, s);
        BP_Mult(B, p, q, c, s);
        BP_Mult(eigenvectors, p, q, c, s);
    }

    for (int i = 0; i < n; i++) {
        eigenvalues[i] = B[i][i];
    }

    return make_pair(eigenvalues, eigenvectors);
}

// ADD FORTRAN SUPPORT
std::pair<std::vector<double>, Matrix<COMPLEX>> Hermit_Lanczos(const Matrix<COMPLEX>& A) {
    auto m = A.size();

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

} // namespace QComputations

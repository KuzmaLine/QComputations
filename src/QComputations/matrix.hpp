#pragma once
#include <typeinfo>
#include <complex>
#include <vector>
#include <iostream>
#include <iterator>
#include <cassert>
#include <iomanip>
#define MKL_Complex16 std::complex<double>
#include <mkl_lapack.h>
#include <mkl_lapacke.h>
#include <functional>
#include "config.hpp"

namespace QComputations {

namespace {
    using COMPLEX = std::complex<double>;
    template<typename T>
    T conj(T a) {
        if (typeid(T) == typeid(std::complex<int>) or
            typeid(T) == typeid(std::complex<short>) or
            typeid(T) == typeid(std::complex<double>) or
            typeid(T) == typeid(std::complex<long>) or
            typeid(T) == typeid(std::complex<double>) or
            typeid(T) == typeid(std::complex<long double>)) {
                return std::conj(a);
            } else {
                return a;
            }
    }

    double conj(double a) { return a; }
}

enum MATRIX_STYLE { C_STYLE = 120, FORTRAN_STYLE = 121 };


// (!!!) NEED CRS_MATRIX for memory optimization. sparseBLAS before working
// ---------------------------------- class Matrix ----------------------------
template<typename T> class Matrix {
    public:
        Matrix() = default;
        explicit Matrix(MATRIX_STYLE matrix_style, size_t n, size_t m) : matrix_style_(matrix_style), n_(n), m_(m), mass_(n_ * m_) {}
        explicit Matrix(MATRIX_STYLE matrix_style, size_t n, size_t m, const T& init_val) : matrix_style_(matrix_style), n_(n), m_(m), mass_(n_ * m_, init_val) {}
        explicit Matrix(const Matrix<T>& A): n_(A.n_), m_(A.m_), mass_(A.mass_), matrix_style_(A.matrix_style_) {}
        explicit Matrix(const std::vector<T>& mass, size_t n, size_t m, MATRIX_STYLE matrix_style): n_(n), m_(m), mass_(mass), matrix_style_(matrix_style) {}
        explicit Matrix(MATRIX_STYLE matrix_style, size_t n, size_t m, std::function<COMPLEX(size_t, size_t)> func);

        // Conversation to another type
        template<typename V>
        Matrix(const Matrix<V>& A): n_(A.n()), m_(A.m()) {
            for (size_t i = 0; i < n_; i++) {
                for (size_t j = 0; j < m_; j++) {
                    mass_.emplace_back(static_cast<V>(A[i][j]));
                }
            }

            matrix_style_ = A.get_matrix_style();
        }

        explicit Matrix(const std::vector<std::vector<T>>& A, MATRIX_STYLE matrix_style);
        //explicit Matrix(const T* A);

        Matrix<T>& operator=(const Matrix<T>& A);

        void modify_row(size_t index, const std::vector<T>& v);
        void modify_col(size_t index, const std::vector<T>& v);

        std::vector<T> row (size_t index) const;
        std::vector<T> col (size_t index) const;
        size_t n() const { return n_; }
        size_t size() const { return n_; }
        size_t m() const { return m_; }

        void add_rows(size_t n);
        void add_cols(size_t m);
        void remove_rows(size_t n);
        void remove_cols(size_t m);
        void expand(size_t n); // add_rows + add_cols
        void reduce(size_t n); // remove_rows + remove_cols

        // DON'T ADD GENERAL TEMPLATE VERSION
        Matrix<T> operator* (const Matrix<T>& A) const;

        Matrix<T> operator+ (const Matrix<T>& A) const;
        Matrix<T> operator- (const Matrix<T>& A) const;

        std::vector<T> operator* (const std::vector<T>& v) const;

        Matrix<T> operator* (const T& num) const;
        Matrix<T>& operator*= (T num);
        Matrix<T> operator+ (const T& num) const;
        Matrix<T> operator- (const T& num) const;
        Matrix<T> operator/ (const T& num) const;
        Matrix<T>& operator/= (T num);

        Matrix<T>& operator+=(const Matrix<T>& A);
        Matrix<T>& operator-=(const Matrix<T>& A);
        Matrix<T>& operator*=(const Matrix<T>& A);

        bool operator==(const Matrix<T>& A) const;

        std::vector<T>& get_mass() { return mass_; }
        std::vector<T> get_mass() const { return mass_; }
        T* data() { return mass_.data(); }
        const T* data() const { return mass_.data(); }
        Matrix<T> transpose() const;
        Matrix<T> hermit() const;
        double determinant() const; // not ready
        void show(size_t width = QConfig::instance().width()) const;
    
        // (!!!) work only with C style
        T* operator[](size_t index_row) { return mass_.data() + index_row * m_; }; 
        const T* operator[](size_t index_row) const { return mass_.data() + index_row * m_; };
    
        // (!!!) work only with FORTRAN style
        T& operator()(size_t index_row, size_t index_col) { return mass_.data()[index_col * n_ + index_row]; }
        const T operator()(size_t index_row, size_t index_col) const { return mass_.data()[index_col * n_ + index_row]; }
    
        size_t LD() const { return (matrix_style_ == C_STYLE ? m_ : n_); }
        bool is_c_style() const { return matrix_style_ == C_STYLE; }
        MATRIX_STYLE get_matrix_style() const { return matrix_style_; }
        MATRIX_STYLE matrix_style() const { return matrix_style_; }
        void to_fortran_style();
        void to_c_style();

        lapack_complex_double* to_upper_lapack() const;
        lapack_complex_double* to_lapack() const;

        //void set_multiply_mode(int multiply_mode) { MULTIPLY_MODE = multiply_mode; }
        Matrix<T> submatrix(size_t n, size_t m, size_t row_index, size_t col_index) const;
        

        size_t index(size_t i, size_t j) const { if (matrix_style_ == C_STYLE) return i * m_ + j;
                                                 else return j * n_ + i; }

        T& elem(size_t i, size_t j) { return mass_.data()[this->index(i, j)]; }
        const T elem(size_t i, size_t j) const { return mass_.data()[this->index(i, j)]; }

        void write_to_csv_file(const std::string& filename) const;
    private:
        size_t get_index(size_t i, size_t j) const { if (matrix_style_ == C_STYLE) return i * m_ + j;
                                                     else return j * n_ + i; }
        size_t n_;
        size_t m_;
        std::vector<T> mass_;
        MATRIX_STYLE matrix_style_;
};

// -------------------------------- Matrix Methods ----------------------------------

template<typename T>
Matrix<T>::Matrix(MATRIX_STYLE matrix_style, size_t n, size_t m, std::function<COMPLEX(size_t, size_t)> func): matrix_style_(matrix_style), n_(n), m_(m), mass_(n_ * m_) {
    for (size_t i = 0; i < n_; i++) {
        for (size_t j = 0; j < m_; j++) {
            mass_[get_index(i,j)] = func(i, j);
        }
    }
}

template<typename T>
void Matrix<T>::to_fortran_style() {
    assert(this->matrix_style_ == C_STYLE);
    auto c_mass = mass_;
    size_t index = 0;
    for (size_t j = 0; j < m_; j++) {
        for (size_t i = 0; i < n_; i++) {
            mass_[index++] = c_mass[get_index(i, j)];
        }
    }

    this->matrix_style_ = FORTRAN_STYLE;
}

template<typename T>
void Matrix<T>::to_c_style() {
    assert(this->matrix_style_ == FORTRAN_STYLE);
    auto fort_mass = mass_;
    size_t index = 0;
    for (size_t i = 0; i < n_; i++) {
        for (size_t j = 0; j < m_; j++) {
            mass_[index++] = fort_mass[j * n_ + i];
        }
    }

    this->matrix_style_ = C_STYLE;
}

template<typename T>
Matrix<T> Matrix<T>::submatrix(size_t n, size_t m, size_t row_index, size_t col_index) const {
    Matrix<T> res(this->get_matrix_style(), n, m);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            res.mass_[res.get_index(i, j)] = this->mass_[get_index(row_index + i, col_index + j)];
        }
    }

    return res;
}

template<typename T>
void Matrix<T>::add_rows(size_t n) {
    if (matrix_style_) {
        n_ += n;
        mass_.resize(n_ * m_, T(0));
    } else {
        for (size_t j = 0; j < m_; j++) {
            mass_.insert(std::next(mass_.begin(), (j + 1) * m_ + j * n), n, T(0));
        }

        n_ += n;
    }
}

template<typename T>
void Matrix<T>::add_cols(size_t m) {
    if (matrix_style_) {
        for (size_t i = 0; i < n_; i++) {
            mass_.insert(std::next(mass_.begin(), (i + 1) * m_ + i * m), m, T(0));
        }

        m_ += m;
    } else {
        m_ += m;
        mass_.resize(n_ * m_, T(0));
    }
}

template<typename T>
void Matrix<T>::expand(size_t n) {
    this->add_rows(n);
    this->add_cols(n);
}

// MAKE FOR FORTRAN
template<typename T>
void Matrix<T>::remove_rows(size_t n) {
    n_ -= n;
    mass_.resize(n_ * m_);
}

// MAKE FOR FORTRAN
template<typename T>
void Matrix<T>::remove_cols(size_t m) {
    m_ -= m;
    for (size_t i = 0; i < n_; i++) {
        mass_.erase(std::next(mass_.begin(), (i + 1) * m_), std::next(mass_.begin(), (i + 1) * m_ + m));
    }
}

template<typename T>
void Matrix<T>::reduce(size_t n) {
    this->remove_rows(n);
    this->remove_cols(n);
}

template<typename T>
Matrix<T>::Matrix(const std::vector<std::vector<T>>& A, MATRIX_STYLE matrix_style) {
    matrix_style_ = matrix_style;
    n_ = A.size();
    m_ = A[0].size();

    mass_.resize(n_ * m_);
    for (size_t i = 0; i < n_; i++) {
        for (size_t j = 0; j < m_; j++) {
            mass_[i * m_ + j] = A[i][j];
        }
    }
}

template<typename T>
void Matrix<T>::modify_row (size_t index, const std::vector<T>& v) {
    for(size_t j = 0; j < m_; j++) {
        mass_[this->get_index(index, j)] = v[j];
    }
}

template<typename T>
void Matrix<T>::modify_col (size_t index, const std::vector<T>& v) {
    for(size_t i = 0; i < n_; i++) {
        mass_[this->get_index(i, index)] = v[i];
    }
}


template<typename T>
Matrix<T> Matrix<T>::operator+ (const Matrix<T>& A) const {
    assert(m_ == A.m_);
    assert(n_ == A.n_);
    assert(matrix_style_ == A.matrix_style_);

    Matrix<T> res(matrix_style_, n_, A.m_);

    for (size_t i = 0; i < n_; i++) {
        for (size_t j = 0; j < A.m_; j++) {
            res.mass_[res.get_index(i, j)] = mass_[this->get_index(i, j)] + A.mass_[A.get_index(i, j)];
        }
    }

    return res;
}

template<typename T>
Matrix<T> Matrix<T>::operator- (const Matrix<T>& A) const {
    assert(m_ == A.m_);
    assert(n_ == A.n_);
    assert(matrix_style_ == A.matrix_style_);

    Matrix<T> res(matrix_style_, n_, A.m_);

    for (size_t i = 0; i < n_; i++) {
        for (size_t j = 0; j < A.m_; j++) {
            res.mass_[res.get_index(i, j)] = mass_[this->get_index(i, j)] - A.mass_[A.get_index(i, j)];
        }
    }

    return res;
}

template<typename T>
Matrix<T> Matrix<T>::operator- (const T& num) const {
    Matrix<T> res(matrix_style_, n_, m_);

    for (size_t i = 0; i < n_; i++) {
        for (size_t j = 0; j < m_; j++) {
            res.mass_[res.get_index(i, j)] = mass_[this->get_index(i, j)] - num;
        }
    }

    return res;
}

template<typename T>
Matrix<T> Matrix<T>::operator/ (const T& num) const {
    Matrix<T> res(matrix_style_, n_, m_);

    for (size_t i = 0; i < n_; i++) {
        for (size_t j = 0; j < m_; j++) {
            res.mass_[res.get_index(i, j)] = mass_[this->get_index(i, j)] / num;
        }
    }

    return res;
}

template<typename T>
Matrix<T> Matrix<T>::operator* (const T& num) const {
    Matrix<T> res(matrix_style_, n_, m_);

    for (size_t i = 0; i < n_; i++) {
        for (size_t j = 0; j < m_; j++) {
            res.mass_[res.get_index(i, j)] = mass_[this->get_index(i, j)] * num;
        }
    }

    return res;
}

template<typename T>
Matrix<T> Matrix<T>::operator+ (const T& num) const {
    Matrix<T> res(matrix_style_, n_, m_);

    for (size_t i = 0; i < n_; i++) {
        for (size_t j = 0; j < m_; j++) {
            res.mass_[res.get_index(i, j)] = mass_[this->get_index(i, j)] + num;
        }
    }

    return res;
}

template<typename T>
std::vector<T> Matrix<T>::operator*(const std::vector<T>& v) const {
    assert(m_ == v.size());
    std::vector<T> res(v.size(), T(0));

    size_t n = v.size();

    for (size_t i = 0; i < n; i++) {
        for (size_t k = 0; k < n; k++) {
            res[i] += mass_[this->get_index(i, k)] * v[k];
        }
    }

    return res;
}

template<typename T>
std::vector<T> operator*(const std::vector<T>& v, const Matrix<T>& A) {
    assert(A.n_ == v.size());
    std::vector<T> res(v.size(), T(0));

    size_t n = v.size();

    for (size_t i = 0; i < n; i++) {
        for (size_t k = 0; k < n; k++) {
            res[i] += v[k] * A[A.index(k, i)];
        }
    }

    return res;
}

template<typename T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& A) {
    assert(m_ == A.m_);
    assert(n_ == A.n_);
    assert(matrix_style_ == A.matrix_style_);

    for (size_t i = 0; i < n_; i++) {
        for (size_t j = 0; j < m_; j++) {
            this->mass_[this->get_index(i, j)] += A.mass_[A.get_index(i, j)];
        }
    }

    return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::operator-=(const Matrix<T>& A) {
    assert(m_ == A.m_);
    assert(n_ == A.n_);
    assert(matrix_style_ == A.matrix_style_);

    for (size_t i = 0; i < n_; i++) {
        for (size_t j = 0; j < m_; j++) {
            this->mass_[this->get_index(i, j)] -= A.mass_[A.get_index(i, j)];
        }
    }

    return *this;
}

template<typename T>
bool Matrix<T>::operator==(const Matrix<T>& A) const {
    if (n_ == A.n_ and m_ == A.m_ and mass_ == A.mass_ and matrix_style_ == A.matrix_style_) {
        return true;
    }

    return false;
}

template<typename T>
std::vector<T> Matrix<T>::row(size_t index) const {
    std::vector<T> res(this->m_);
    for(size_t j = 0; j < m_; j++) {
        res[j] = mass_[get_index(index, j)];
    }

    return res;
}

template<typename T>
std::vector<T> Matrix<T>::col(size_t index) const {
    std::vector<T> res(this->n_);
    for(size_t i = 0; i < n_; i++) {
        res[i] = mass_[get_index(i, index)];
    }

    return res;
}

template<typename T>
Matrix<T> Matrix<T>::transpose() const {
    Matrix<T> res(matrix_style_, m_, n_);

    for (size_t i = 0; i < m_; i++) {
        for (size_t j = 0; j < n_; j++) {
            res[i][j] = mass_[this->get_index(j, i)];
        }
    }

    return res;
}

template<typename T>
Matrix<T> Matrix<T>::hermit() const {
    assert(typeid(T) == typeid(std::complex<double>) or
           typeid(T) == typeid(std::complex<int>) or
           typeid(T) == typeid(std::complex<short>) or
           typeid(T) == typeid(std::complex<long double>) or
           typeid(T) == typeid(std::complex<long>));
    Matrix<T> res(matrix_style_, m_, n_);

    for (size_t i = 0; i < m_; i++) {
        for (size_t j = 0; j < n_; j++) {
            res[i][j] = std::conj(mass_[this->get_index(j, i)]);
        }
    }

    return res;
}

template<typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& A) {
    n_ = A.n_;
    m_ = A.m_;
    mass_ = A.mass_;
    matrix_style_ = A.matrix_style_;
    return *this;
}

template<typename T>
void Matrix<T>::show(size_t width) const {
    for (size_t i = 0; i < n_; i++) {
        for (size_t j = 0; j < m_; j++) {
            std::cout << std::setw(width) << mass_[get_index(i, j)] << " ";
        }

        std::cout << std::endl;
    }

    std::cout << std::endl;
}

template<>
Matrix<COMPLEX>& Matrix<COMPLEX>::operator*=(const Matrix<COMPLEX>& A);

template<>
Matrix<double>& Matrix<double>::operator*=(const Matrix<double>& A);

/*-------------------------- FUNCTIONS -------------------- */

template<typename T>
Matrix<T> E_Matrix(size_t n) {
    Matrix<T> E(C_STYLE, n, n, COMPLEX(0));

    for (size_t i = 0; i < n; i++) {
        E[i][i] = T(1);
    }

    return E;
}

template<typename T>
Matrix<T> exp(const Matrix<T>& A, double t, COMPLEX arg = COMPLEX(1, 0), int EXP_ACCURACY = QConfig::instance().exp_accuracy()) {
    Matrix<T> RES = E_Matrix<COMPLEX>(A.n());
    Matrix<T> B = RES;
    int coef = 1;
    COMPLEX x = 1;

    for (size_t i = 1; i <= EXP_ACCURACY; i++) {
        coef *= i;
        B *= A;
        x *= t * arg;
        RES += B * x / coef;
    }

    return RES;
}

template<typename T>
void optimized_multiply(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C,
                        T alpha, T betta, char trans_A = 'N', char trans_B = 'N');

template<>
void optimized_multiply(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C,
                         double alpha, double betta, char trans_A, char trans_B);

template<>
void optimized_multiply(const Matrix<COMPLEX>& A, const Matrix<COMPLEX>& B, Matrix<COMPLEX>& C,
                        COMPLEX alpha, COMPLEX betta, char trans_A, char trans_B);

std::vector<Matrix<COMPLEX>> OPT_Runge_Kutt_4(const std::vector<double>& x,
                                        const Matrix<COMPLEX>& y0,
                                        std::function<void(double, const Matrix<COMPLEX>&, Matrix<COMPLEX>&)> f);

std::vector<Matrix<COMPLEX>> OPT_Runge_Kutt_2(const std::vector<double>& x,
                                        const Matrix<COMPLEX>& y0,
                                        std::function<void(double, const Matrix<COMPLEX>&, Matrix<COMPLEX>&)> f);


/* ------------------------- DON'T TOUCH -------------------- */

template<>
lapack_complex_double* Matrix<COMPLEX>::to_upper_lapack() const;

template<>
lapack_complex_double* Matrix<COMPLEX>::to_lapack() const;


} // namespace QComputations
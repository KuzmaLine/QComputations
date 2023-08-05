#pragma once
#include <typeinfo>
#include <complex>
#include <vector>
#include <iostream>
#include <iterator>
#include <cassert>
#include <iomanip>
#include <mkl_lapack.h>
#include <mkl_lapacke.h>
#include "config.hpp"

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

    double conj(double a) { return a;}
}

// (!!!) NEED CRS_MATRIX for memory optimization
// ---------------------------------- class Matrix ----------------------------
template<typename T> class Matrix {
    public:
        Matrix() = default;
        Matrix(size_t n, size_t m): n_(n), m_(m), mass_(n_ * m_) {}
        Matrix(size_t n, size_t m, const T& init_val): n_(n), m_(m), mass_(n_ * m_, init_val) {}
        Matrix(const Matrix<T>& A): n_(A.n_), m_(A.m_), mass_(A.mass_) {}
        Matrix(const std::vector<T>& mass, size_t n, size_t m): n_(n), m_(m), mass_(mass) {}

        template<typename V>
        Matrix(const Matrix<V>& A): n_(A.n()), m_(A.m()) {
            for (size_t i = 0; i < n_; i++) {
                for (size_t j = 0; j < m_; j++) {
                    mass_.emplace_back(static_cast<V>(A[i][j]));
               }
            }
        }
        explicit Matrix(const std::vector<std::vector<T>>& A);
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
        void expand(size_t n);
        void reduce(size_t n);

        // DON'T ADD TEMPLATE VERSION (C++11)
        Matrix<T> operator* (const Matrix<T>& A) const;
        Matrix<T> operator+ (const Matrix<T>& A) const;
        Matrix<T> operator- (const Matrix<T>& A) const;

        // DON'T ADD TEMPLATE VERSION (C++11)
        //template<typename V>
        //Matrix<T> operator* (const Matrix<V>& A) const;

        std::vector<T> operator* (const std::vector<T>& v) const;

        Matrix<T> operator* (const T& num) const;
        Matrix<T> operator+ (const T& num) const;
        Matrix<T> operator- (const T& num) const;
        Matrix<T> operator/ (const T& num) const;

        Matrix<T>& operator+=(const Matrix<T>& A);

        bool operator==(const Matrix<T>& A) const;

        std::vector<T> get_mass() const { return mass_; }
        T* mass_data() { return mass_.data(); }
        const T* mass_data() const { return mass_.data(); }
        Matrix<T> transpose() const;
        Matrix<T> hermit() const;
        double determinant() const; // not ready
        void show(size_t width = 10) const;
        T* operator[](size_t index_row) { return mass_.data() + index_row * m_; };
        const T* operator[](size_t index_row) const { return mass_.data() + index_row * m_; };

        T& operator()(size_t i, size_t j) { return mass_[get_index(i, j)];}
        lapack_complex_double* to_upper_lapack() const;
        lapack_complex_double* to_lapack() const;

    private:
        size_t get_index(size_t i, size_t j) const { return i * m_ + j; }
        size_t n_;
        size_t m_;
        std::vector<T> mass_;

        int MULTIPLY_MODE = config::MULTIPLY_MODE;
};

// -------------------------------- Matrix Methods ----------------------------------

template<typename T>
void Matrix<T>::add_rows(size_t n) {
    n_ += n;
    mass_.resize(n_ * m_, T(0));
}

template<typename T>
void Matrix<T>::add_cols(size_t m) {
    for (size_t i = 0; i < n_; i++) {
        mass_.insert(std::next(mass_.begin(), (i + 1) * m_ + i * m), m, T(0));
    }

    m_ += m;
}

template<typename T>
void Matrix<T>::expand(size_t n) {
    this->add_rows(n);
    this->add_cols(n);
}

template<typename T>
void Matrix<T>::remove_rows(size_t n) {
    n_ -= n;
    mass_.resize(n_ * m_);
}

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
Matrix<T>::Matrix(const std::vector<std::vector<T>>& A) {
    n_ = A.size();
    m_ = A[0].size();

    for (size_t i = 0; i < n_; i++) {
        for (size_t j = 0; j < m_; j++) {
            mass_.emplace_back(A[i][j]);
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

    Matrix<T> res(n_, A.m_);

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

    Matrix<T> res(n_, A.m_);

    for (size_t i = 0; i < n_; i++) {
        for (size_t j = 0; j < A.m_; j++) {
            res.mass_[res.get_index(i, j)] = mass_[this->get_index(i, j)] - A.mass_[A.get_index(i, j)];
        }
    }

    return res;
}

template<typename T>
Matrix<T> Matrix<T>::operator- (const T& num) const {
    Matrix<T> res(n_, m_);

    for (size_t i = 0; i < n_; i++) {
        for (size_t j = 0; j < m_; j++) {
            res.mass_[res.get_index(i, j)] = mass_[this->get_index(i, j)] - num;
        }
    }

    return res;
}

template<typename T>
Matrix<T> Matrix<T>::operator/ (const T& num) const {
    Matrix<T> res(n_, m_);

    for (size_t i = 0; i < n_; i++) {
        for (size_t j = 0; j < m_; j++) {
            res.mass_[res.get_index(i, j)] = mass_[this->get_index(i, j)] / num;
        }
    }

    return res;
}

template<typename T>
Matrix<T> Matrix<T>::operator* (const T& num) const {
    Matrix<T> res(n_, m_);

    for (size_t i = 0; i < n_; i++) {
        for (size_t j = 0; j < m_; j++) {
            res.mass_[res.get_index(i, j)] = mass_[this->get_index(i, j)] * num;
        }
    }

    return res;
}

template<typename T>
Matrix<T> Matrix<T>::operator+ (const T& num) const {
    Matrix<T> res(n_, m_);

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
            //res[i] += conj(mass_[this->get_index(i, k)]) * v[k];
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
            //res[i] += conj(v[k]) * A[k][i];
            res[i] += v[k] * A[k][i];
        }
    }

    return res;
}

template<typename T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& A) {
    for (size_t i = 0; i < n_; i++) {
        for (size_t j = 0; j < m_; j++) {
            this->mass_[this->get_index(i, j)] += A[i][j];
        }
    }

    return *this;
}

template<typename T>
bool Matrix<T>::operator==(const Matrix<T>& A) const {
    if (n_ == A.n_ and m_ == A.m_ and mass_ == A.mass_) {
        return true;
    }

    return false;
}

template<typename T>
std::vector<T> Matrix<T>::row(size_t index) const {
    std::vector<T> res;
    std::copy(mass_.begin() + index * m_, mass_.begin() + (index + 1) * m_, std::back_inserter(res));
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
    Matrix<T> res(m_, n_);

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
    Matrix<T> res(m_, n_);

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
    return *this;
}

template<typename T>
void Matrix<T>::show(size_t width) const {
    for (size_t i = 0; i < n_; i++) {
        for (size_t j = 0; j < m_; j++) {
            std::cout << std::setw(width) << mass_[i * m_ + j] << " ";
        }

        std::cout << std::endl;
    }

    std::cout << std::endl;
}

template<>
lapack_complex_double* Matrix<COMPLEX>::to_upper_lapack() const;

template<>
lapack_complex_double* Matrix<COMPLEX>::to_lapack() const;

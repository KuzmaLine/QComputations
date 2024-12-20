#ifdef ENABLE_MPI
#ifdef ENABLE_CLUSTER

#pragma once
#include <complex>

#include "blocked_matrix.hpp"

namespace QComputations {

    namespace {
#ifdef MKL_ILP64
        using ILP_TYPE = long long;
#else
        using ILP_TYPE = int;
#endif
        ILP_TYPE INC = 1;
        using COMPLEX = std::complex<double>;
    }  // namespace

    template <typename T>
    class BLOCKED_Vector : public BLOCKED_Matrix<T> {
       public:
        explicit BLOCKED_Vector() = default;
        explicit BLOCKED_Vector(ILP_TYPE ctxt, const std::vector<T>& x)
            : BLOCKED_Matrix<T>(ctxt, GE, Matrix<T>(x, x.size(), 1)){};
        explicit BLOCKED_Vector(const BLOCKED_Vector<T>& x, const Matrix<T>& local_vector);
        explicit BLOCKED_Vector(ILP_TYPE ctxt, size_t n, std::function<T(size_t, size_t)> func)
            : BLOCKED_Matrix<T>(ctxt, GE, n, 1, func) {}
        explicit BLOCKED_Vector(ILP_TYPE ctxt, size_t n, size_t NB = 0)
            : BLOCKED_Matrix<T>(ctxt, GE, n, 1, size_t(NB), size_t(1)) {}
        explicit BLOCKED_Vector(ILP_TYPE ctxt, size_t n, T value, size_t NB = 0)
            : BLOCKED_Matrix<T>(ctxt, GE, n, 1, T(value), size_t(NB), size_t(1)) {}

        // Сделать по размерностям результата умножения
        explicit BLOCKED_Vector(const BLOCKED_Matrix<T>& A, const BLOCKED_Vector<T>& x)
            : BLOCKED_Vector(x.ctxt(), A.n(), 1, x.NB()) {}

        ILP_TYPE inc() const { return INC; }

        T get(size_t i) const { return this->BLOCKED_Matrix<T>::get(i, 0); }
        void set(size_t i, T num) { this->BLOCKED_Matrix<T>::set(i, 0, num); }

        T& operator[](size_t i) { return BLOCKED_Matrix<T>::local_matrix_(i, 0); }
        const T& operator[](size_t i) const { return BLOCKED_Matrix<T>::local_matrix_(i, 0); }
        T& operator()(size_t i) { return BLOCKED_Matrix<T>::local_matrix_(i, 0); }
        const T& operator()(size_t i) const { return BLOCKED_Matrix<T>::local_matrix_(i, 0); }

        BLOCKED_Vector<T> operator*(const BLOCKED_Matrix<T>& A) const;
        BLOCKED_Vector<T> operator+(const BLOCKED_Vector<T>& x) const;
        void operator+=(const BLOCKED_Vector<T>& x);
        BLOCKED_Vector<T> operator-(const BLOCKED_Vector<T>& x) const;
        void operator-=(const BLOCKED_Vector<T>& x);
        BLOCKED_Vector<T> operator*(const BLOCKED_Vector<T>& x) const;
        BLOCKED_Vector<T> operator/(const BLOCKED_Vector<T>& x) const;
        std::vector<T> get_vector() const;

        BLOCKED_Vector<T> operator+(T x) const;
        BLOCKED_Vector<T> operator-(T x) const;
        BLOCKED_Vector<T> operator*(T x) const;
        BLOCKED_Vector<T> operator/(T x) const;
    };

    template <typename T>
    BLOCKED_Vector<T>::BLOCKED_Vector(const BLOCKED_Vector<T>& A, const Matrix<T>& x)
        : BLOCKED_Matrix<T>::ctxt_(A.ctxt()),
          BLOCKED_Matrix<T>::matrix_type_(A.get_matrix_type()),
          BLOCKED_Matrix<T>::n_(A.n()),
          BLOCKED_Matrix<T>::m_(A.m_),
          BLOCKED_Matrix<T>::NB_(A.NB()),
          BLOCKED_Matrix<T>::MB_(A.MB_),
          BLOCKED_Matrix<T>::local_matrix_(x) {}

    template <typename T>
    BLOCKED_Vector<T> BLOCKED_Vector<T>::operator+(T num) const {
        BLOCKED_Vector<T> res(*this, BLOCKED_Matrix<T>::local_matrix_ + num);

        return res;
    }

    template <typename T>
    std::vector<T> BLOCKED_Vector<T>::get_vector() const {
        std::vector<T> res(this->n());

        for (size_t i = 0; i < this->n(); i++) {
            res[i] = this->get(i);
        }

        return res;
    }

    template <typename T>
    BLOCKED_Vector<T> BLOCKED_Vector<T>::operator-(T num) const {
        BLOCKED_Vector<T> res(*this, BLOCKED_Matrix<T>::local_matrix_ - num);

        return res;
    }

    template <typename T>
    BLOCKED_Vector<T> BLOCKED_Vector<T>::operator/(T num) const {
        BLOCKED_Vector<T> res(*this, BLOCKED_Matrix<T>::local_matrix_ / num);

        return res;
    }

    // ----------------------------------------- FUNCTIONS --------------------------------------

    // a * b
    double scalar_product(const BLOCKED_Vector<double>& a, const BLOCKED_Vector<double>& b);
    // <a|b>
    COMPLEX scalar_product(const BLOCKED_Vector<COMPLEX>& a, const BLOCKED_Vector<COMPLEX>& b);
    BLOCKED_Vector<double> blocked_matrix_get_col(ILP_TYPE ctxt, const BLOCKED_Matrix<double>& A, size_t col);
    BLOCKED_Vector<double> blocked_matrix_get_row(ILP_TYPE ctxt, const BLOCKED_Matrix<double>& A, size_t row);
    BLOCKED_Vector<COMPLEX> blocked_matrix_get_col(ILP_TYPE ctxt, const BLOCKED_Matrix<COMPLEX>& A, size_t col);
    BLOCKED_Vector<COMPLEX> blocked_matrix_get_row(ILP_TYPE ctxt, const BLOCKED_Matrix<COMPLEX>& A, size_t row);
    std::vector<BLOCKED_Vector<double>> blocked_matrix_to_blocked_vectors(ILP_TYPE ctxt,
                                                                          const BLOCKED_Matrix<double>& A);
    std::vector<BLOCKED_Vector<COMPLEX>> blocked_matrix_to_blocked_vectors(ILP_TYPE ctxt,
                                                                           const BLOCKED_Matrix<COMPLEX>& A);

}  // namespace QComputations

#endif
#endif
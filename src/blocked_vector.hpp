#ifdef ENABLE_MPI
#ifdef ENABLE_CLUSTER

#pragma once
#include "blocked_matrix.hpp"
#include <complex>

namespace QComputations {

namespace {
#ifdef MKL_ILP64
    using ILP_TYPE = long long;
#else
    using ILP_TYPE = int;
#endif
    ILP_TYPE INC = 1;
    using COMPLEX = std::complex<double>;
}

namespace mpi {

void init_vector_grid(ILP_TYPE& ctxt, ILP_TYPE proc_rows = 0, ILP_TYPE proc_cols = 0);

}

template<typename T>
class BLOCKED_Vector: public BLOCKED_Matrix<T> {
    public:
        explicit BLOCKED_Vector() = default;
        explicit BLOCKED_Vector(ILP_TYPE ctxt, const std::vector<T>& x): BLOCKED_Matrix<T>(ctxt, GE, Matrix<T>(x, x.size(), 1, FORTRAN_STYLE)) {};
        explicit BLOCKED_Vector(const BLOCKED_Vector<T>& x, const Matrix<T>& local_vector);
        explicit BLOCKED_Vector(ILP_TYPE ctxt, size_t n, std::function<T(size_t, size_t)> func): BLOCKED_Matrix(ctxt, GE, n, 1, func) {}
        explicit BLOCKED_Vector(ILP_TYPE ctxt, size_t n, T value, size_t NB = 0): BLOCKED_Matrix(ctxt, GE, n, 1, value, NB, 1) {}
        explicit BLOCKED_Vector(ILP_TYPE ctxt, size_t n, size_t NB = 0): BLOCKED_Matrix(ctxt, GE, n, 1, NB, 1) {}
        explicit BLOCKED_Vector(ILP_TYPE ctxt, const std::vector<T>& x, ILP_TYPE root_id): BLOCKED_Matrix(ctxt, Matrix<T>(x, x.size(), 1, FORTRAN_STYLE), root_id) {}

        // Сделать по размерностям результата умножения
        explicit BLOCKED_Vector(const BLOCKED_Matrix<T>& A, const BLOCKED_Vector<T>& x): BLOCKED_Vector(x.ctxt(), A.n(), 1, x.NB()) {}

        ILP_TYPE inc() const { return INC; }

        T get(size_t i) const { return res.get(i, 1); }
        void set(size_t i, T num) { res.set(i, 1, num); }

        T& operator[](size_t i) { return local_matrix(i, 1); }
        const T& operator[](size_t i) const { return local_matrix(i, 1); }
        T& operator()(size_t i) { return local_matrix_(i, 1); }
        const T& operator()(size_t i) const { return local_matrix_(i, 1); }

        BLOCKED_Vector<T> operator*(const BLOCKED_Matrix<T>& A) const;
        BLOCKED_Vector<T> operator+(const BLOCKED_Vector<T>& x) const;
        void operator+=(const BLOCKED_Vector<T>& x);
        BLOCKED_Vector<T> operator-(const BLOCKED_Vector<T>& x) const;
        void operator-=(const BLOCKED_Vector<T>& x);
        BLOCKED_Vector<T> operator*(const BLOCKED_Vector<T>& x) const;
        BLOCKED_Vector<T> operator/(const BLOCKED_Vector<T>& x) const;

        BLOCKED_Vector<T> operator+(T x) const;
        BLOCKED_Vector<T> operator-(T x) const;
        BLOCKED_Vector<T> operator*(T x) const;
        BLOCKED_Vector<T> operator/(T x) const;
};

template<typename T>
BLOCKED_Vector<T>::BLOCKED_Vector<T>(const BLOCKED_Vector<T>& A, const Matrix<T>& x): ctxt_(A.ctxt()), matrix_type_(A.get_matrix_type()),
                                                                        n_(A.n()), m_(A.m_), NB_(A.NB()), MB_(A.MB_), local_matrix_(x) {}

template<typename T>
BLOCKED_Vector<T> BLOCKED_Vector<T>::operator+(T num) const {
    BLOCKED_Vector<T> res(*this, local_matrix_ + num);

    return res;
}


template<typename T>
BLOCKED_Vector<T> BLOCKED_Vector<T>::operator-(T num) const {
    BLOCKED_Vector<T> res(*this, local_matrix_ - num);

    return res;
}

template<typename T>
BLOCKED_Vector<T> BLOCKED_Vector<T>::operator/(T num) const {
    BLOCKED_Vector<T> res(*this, local_matrix_ / num);

    return res;
}

// ----------------------------------------- FUNCTIONS --------------------------------------

double scalar_product(const BLOCKED_Vector<double>& a, const BLOCKED_Vector<double>& b);
COMPLEX scalar_product(const BLOCKED_Vector<COMPLEX>& a, const BLOCKED_Vector<COMPLEX>& b);
BLOCKED_Vector<double> blocked_matrix_get_col(ILP_TYPE ctxt, const BLOCKED_Matrix<double>& A, size_t col);
BLOCKED_Vector<COMPLEX> blocked_matrix_get_col(ILP_TYPE ctxt, const BLOCKED_Matrix<COMPLEX>& A, size_t col);
std::vector<BLOCKED_Vector<double>> blocked_matrix_to_blocked_vectors(ILP_TYPE ctxt, const BLOCKED_Matrix<double>& A);
std::vector<BLOCKED_Vector<COMPLEX>> blocked_matrix_to_blocked_vectors(ILP_TYPE ctxt, const BLOCKED_Matrix<COMPLEX>& A);

} // namespace QComputations

#endif
#endif
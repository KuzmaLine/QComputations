#ifdef ENABLE_MPI
#ifdef ENABLE_CLUSTER

#pragma once
#include <iostream>
#include <functional>
#include "matrix.hpp"
#include "functions.hpp"
#include "mpi_functions.hpp"
#include <vector>
#include <string>
#include <complex>

namespace {
#ifdef MKL_ILP64
    using ILP_TYPE = long long;
#else
    using ILP_TYPE = int;
#endif

    using COMPLEX = std::complex<double>;
}

namespace QComputations {

enum MATRIX_TYPE { GE = 120, SY = 121, HE = 122};

template <typename T>
class BLOCKED_Matrix {
    public:
        explicit BLOCKED_Matrix() = default;
        explicit BLOCKED_Matrix(ILP_TYPE ctxt, MATRIX_TYPE type, size_t n, size_t m, std::function<T(size_t, size_t)> func);
        explicit BLOCKED_Matrix(ILP_TYPE ctxt, MATRIX_TYPE type, const Matrix<T>& A, ILP_TYPE root_id);

        // Make dims for multiply matrix
        explicit BLOCKED_Matrix(const BLOCKED_Matrix<T>& A, const BLOCKED_Matrix<T>& B);

        T get(size_t i, size_t j) const;
        void set(T num, size_t i, size_t j);

        Matrix<T> Gather(ILP_TYPE root_row, ILP_TYPE root_col) const;

        std::vector<ILP_TYPE> desc() const;

        BLOCKED_Matrix<T> operator*(const BLOCKED_Matrix<T>& B) const;
        BLOCKED_Matrix<T> operator*(T num) const;
        void operator*=(const BLOCKED_Matrix<T>& B);
        BLOCKED_Matrix<T> operator+(const BLOCKED_Matrix<T>& B) const;
        BLOCKED_Matrix<T> operator+(T num) const;
        void operator+=(const BLOCKED_Matrix<T>& B);
        BLOCKED_Matrix<T> operator-(const BLOCKED_Matrix<T>& B) const;
        BLOCKED_Matrix<T> operator-(T num) const;
        void operator-=(const BLOCKED_Matrix<T>& B);
        BLOCKED_Matrix<T> operator/(T num) const;

        size_t local_n() const { return local_matrix_.n(); }
        size_t local_m() const { return local_matrix_.m(); }
        size_t n() const { return n_; }
        size_t m() const { return m_; }
        size_t NB() const { return NB_; }
        size_t MB() const { return MB_; }
        MATRIX_TYPE matrix_type() const { return matrix_type_; }
        Matrix<T>& get_local_matrix() { return local_matrix_;}
        const Matrix<T>& get_local_matrix() const { return local_matrix_;}

        T* data() { return local_matrix_.data(); }
        const T* data() const { return local_matrix_.data(); }

        void print_distributed(const std::string& name) const { mpi::print_distributed_matrix<T>(local_matrix_, name, ctxt_); }
        void show(ILP_TYPE root_id, size_t width = 10) const;
    private:
        size_t get_global_index(size_t i, size_t j) { return j * n_ + i; }
        size_t get_local_index(size_t i, size_t j) { return j * local_matrix_.n() + i; }

        MATRIX_TYPE matrix_type_;
        ILP_TYPE ctxt_;
        size_t n_;
        size_t m_;
        size_t NB_;
        size_t MB_;
        Matrix<T> local_matrix_;
};

template<typename T>
BLOCKED_Matrix<T>::BLOCKED_Matrix<T>(const BLOCKED_Matrix<T>& A,
                                     const BLOCKED_Matrix<T>& B): ctxt_(A.ctxt_), n_(A.n_),
                                                                  m_(B.m_), NB_(A.NB_), MB_(B.MB_) {
    ILP_TYPE iZERO = 0;
    matrix_type_ = GE;
    ILP_TYPE proc_rows, proc_cols, myrow, mycol;
    mpi::blacs_gridinfo(ctxt_, proc_rows, proc_cols, myrow, mycol);
    ILP_TYPE nrows = mpi::numroc(n_, NB_, myrow, iZERO, proc_rows);
    ILP_TYPE ncols = mpi::numroc(m_, MB_, mycol, iZERO, proc_cols);
    local_matrix_ = Matrix<T>(FORTRAN_STYLE, nrows, ncols);
}

template<typename T>
void BLOCKED_Matrix<T>::operator+=(const BLOCKED_Matrix<T>& B) {
    *this = *this + B;
}

template<typename T>
void BLOCKED_Matrix<T>::operator*=(const BLOCKED_Matrix<T>& B) {
    *this = *this * B;
}

template<typename T>
BLOCKED_Matrix<T>::BLOCKED_Matrix<T>(ILP_TYPE ctxt, MATRIX_TYPE type,
                                     size_t n, size_t m,
                                     std::function<T(size_t, size_t)> func): ctxt_(ctxt), matrix_type_(type), n_(n), m_(m) {
    ILP_TYPE iZERO = 0;
    ILP_TYPE proc_rows, proc_cols, myrow, mycol;
    mpi::blacs_gridinfo(ctxt_, proc_rows, proc_cols, myrow, mycol);
    NB_ = n / proc_rows;
    MB_ = m / proc_cols;

    ILP_TYPE nrows = mpi::numroc(n, NB_, myrow, iZERO, proc_rows);
    ILP_TYPE ncols = mpi::numroc(m, MB_, mycol, iZERO, proc_cols);

    if (matrix_type_ == GE) {
        local_matrix_ = Matrix<T>(FORTRAN_STYLE, nrows, ncols);

        for (size_t i = 0; i < nrows; i++) {
            for (size_t j = 0; j < ncols; j++) {
                auto index_row = mpi::indxl2g(i, NB_, myrow, iZERO, proc_rows);
                auto index_col = mpi::indxl2g(j, MB_, mycol, iZERO, proc_cols);

                local_matrix_(i, j) = func(index_row, index_col);
            }
        }
    } else if (matrix_type_ == HE) {
        local_matrix_ = Matrix<T>(FORTRAN_STYLE, nrows, ncols);

        for (size_t i = 0; i < nrows; i++) {
            for (size_t j = 0; j < ncols; j++) {
                auto index_row = mpi::indxl2g(i, NB_, myrow, iZERO, proc_rows);
                auto index_col = mpi::indxl2g(j, MB_, mycol, iZERO, proc_cols);

                if (index_col >= index_row) local_matrix_(i, j) = func(index_row, index_col);
            }
        }
    } else if (matrix_type_ == SY) {

    } else {
        std::cerr << "Incorrect matrix type!" << std::endl;
    }
}

template<typename T>
std::vector<ILP_TYPE> BLOCKED_Matrix<T>::desc() const {
    ILP_TYPE iZERO = 0;
    ILP_TYPE info;
    return mpi::descinit(n_, m_, NB_, MB_, iZERO, iZERO, ctxt_, local_matrix_.n(), info);
}

template<typename T>
void BLOCKED_Matrix<T>::show(ILP_TYPE root_id, size_t width) const {
    ILP_TYPE rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (matrix_type_ == GE) {
        for (size_t i = 0; i < n_; i++) {
            for (size_t j = 0; j < m_; j++) {
                auto elem = this->get(i, j);

                if (rank == root_id) {
                    std::cout << std::setw(width) << elem << " ";
                }
            }

            if (rank == root_id) std::cout << std::endl;
        }
    } else if (matrix_type_ == HE) {
        for (size_t i = 0; i < n_; i++) {
            for (size_t j = 0; j < m_; j++) {
                COMPLEX elem;
                if (j >= i) {
                    elem = this->get(i, j);
                } else {
                    elem = std::conj(this->get(j, i));
                }

                if (rank == root_id) {
                    std::cout << std::setw(width) << elem << " ";
                }
            }

            if (rank == root_id) std::cout << std::endl;
        }
    }

    if (rank == root_id) std::cout << std::endl;
}

} // namespace QComputations

#endif
#endif
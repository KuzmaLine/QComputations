#ifdef ENABLE_MPI
#ifdef ENABLE_CLUSTER

#pragma once
#include <complex>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "config.hpp"
#include "functions.hpp"
#include "matrix.hpp"
#include "mpi_functions.hpp"

namespace {
#ifdef MKL_ILP64
using ILP_TYPE = long long;
#else
using ILP_TYPE = int;
#endif

using COMPLEX = std::complex<double>;
}  // namespace

namespace QComputations {

enum MATRIX_TYPE { GE = 120, SY = 121, HE = 122 };

template <typename T>
class BLOCKED_Matrix {
   public:
    explicit BLOCKED_Matrix() = default;
    explicit BLOCKED_Matrix(const BLOCKED_Matrix<T>& A, const Matrix<T>& local_matrix);
    explicit BLOCKED_Matrix(ILP_TYPE ctxt,
                            MATRIX_TYPE type,
                            size_t n,
                            size_t m,
                            std::function<T(size_t, size_t)> func,
                            size_t NB = 0,
                            size_t MB = 0);
    explicit BLOCKED_Matrix(ILP_TYPE ctxt, MATRIX_TYPE type, size_t n, size_t m, size_t NB = 0, size_t MB = 0);
    explicit BLOCKED_Matrix(ILP_TYPE ctxt, MATRIX_TYPE type, size_t n, size_t m, T value, size_t NB = 0, size_t MB = 0);
    explicit BLOCKED_Matrix(ILP_TYPE ctxt, MATRIX_TYPE type, const Matrix<T>& A, size_t NB = 0, size_t MB = 0);

    // Make dims for multiply matrix
    explicit BLOCKED_Matrix(const BLOCKED_Matrix<T>& A, const BLOCKED_Matrix<T>& B);

    T get(size_t i, size_t j) const;
    void set(size_t i, size_t j, T num);

    Matrix<T> Gather(ILP_TYPE root_row, ILP_TYPE root_col) const;

    std::vector<ILP_TYPE> desc() const;

    BLOCKED_Matrix<T> operator*(const BLOCKED_Matrix<T>& B) const;
    void operator*=(const BLOCKED_Matrix<T>& B);
    BLOCKED_Matrix<T> operator*(T num) const;
    void operator*=(T num);
    BLOCKED_Matrix<T> operator+(const BLOCKED_Matrix<T>& B) const;
    void operator+=(const BLOCKED_Matrix<T>& B);
    BLOCKED_Matrix<T> operator+(T num) const;
    void operator+=(T num);
    BLOCKED_Matrix<T> operator-(const BLOCKED_Matrix<T>& B) const;
    void operator-=(const BLOCKED_Matrix<T>& B);
    BLOCKED_Matrix<T> operator-(T num) const;
    void operator-=(T num);
    BLOCKED_Matrix<T> operator/(T num) const;

    size_t local_n() const { return local_matrix_.n(); }
    size_t local_m() const { return local_matrix_.m(); }
    size_t n() const { return n_; }
    size_t m() const { return m_; }
    size_t NB() const { return NB_; }
    size_t MB() const { return MB_; }
    ILP_TYPE ctxt() const { return ctxt_; }
    MATRIX_TYPE matrix_type() const { return matrix_type_; }
    Matrix<T>& get_local_matrix() { return local_matrix_; }
    const Matrix<T>& get_local_matrix() const { return local_matrix_; }

    T* data(size_t i = 0, size_t j = 0) { return local_matrix_.data() + get_local_index(i, j); }
    const T* data(size_t i = 0, size_t j = 0) const { return local_matrix_.data() + get_local_index(i, j); }

    T& operator()(size_t i, size_t j) { return local_matrix_(i, j); }
    const T operator()(size_t i, size_t j) const { return local_matrix_(i, j); }

    void print_distributed(const std::string& name) const;
    void show(size_t width = QConfig::instance().width(), ILP_TYPE root_id = mpi::ROOT_ID) const;

    void write_to_csv_file(const std::string& filename) const;

    ILP_TYPE get_global_row(size_t i) const;
    ILP_TYPE get_global_col(size_t j) const;

    ILP_TYPE get_local_row(size_t i) const;
    ILP_TYPE get_local_col(size_t j) const;

    ILP_TYPE get_row_proc(size_t i) const;
    ILP_TYPE get_col_proc(size_t i) const;

    bool is_my_elem_row(size_t i) const;
    bool is_my_elem_col(size_t j) const;

    BLOCKED_Matrix<T> hermit() const;

    std::vector<T> col(size_t i) const;
    std::vector<T> row(size_t j) const;

   protected:
    size_t get_global_index(size_t i, size_t j) const { return j * n_ + i; }
    size_t get_local_index(size_t i, size_t j) const { return j * local_matrix_.n() + i; }

    MATRIX_TYPE matrix_type_;
    ILP_TYPE ctxt_;
    size_t n_;
    size_t m_;
    size_t NB_;
    size_t MB_;
    Matrix<T> local_matrix_;
};

template <typename T>
BLOCKED_Matrix<T> BLOCKED_Matrix<T>::operator/(T num) const {
    BLOCKED_Matrix<T> A(*this);
    A.local_matrix_ /= num;
    return A;
}

template <typename T>
BLOCKED_Matrix<T>::BLOCKED_Matrix(ILP_TYPE ctxt, MATRIX_TYPE type, const Matrix<T>& A, size_t NB, size_t MB)
    : ctxt_(ctxt), matrix_type_(type), n_(A.n()), m_(A.m()) {
    ILP_TYPE iZERO = 0;
    ILP_TYPE proc_rows, proc_cols, myrow, mycol;
    mpi::blacs_gridinfo(ctxt_, proc_rows, proc_cols, myrow, mycol);

    if (NB == 0) {
        NB_ = n_ / proc_rows;
    }

    if (MB == 0) {
        MB_ = m_ / proc_cols;
    }

    if (NB_ == 0) {
        NB_ = 1;
    }

    if (MB_ == 0) {
        MB_ = 1;
    }

    ILP_TYPE nrows = mpi::numroc(n_, NB_, myrow, iZERO, proc_rows);
    ILP_TYPE ncols = mpi::numroc(m_, MB_, mycol, iZERO, proc_cols);

    if (matrix_type_ == GE) {
        local_matrix_ = Matrix<T>(FORTRAN_STYLE, nrows, ncols);

        for (size_t i = 0; i < nrows; i++) {
            for (size_t j = 0; j < ncols; j++) {
                auto index_row = mpi::indxl2g(i, NB_, myrow, iZERO, proc_rows);
                auto index_col = mpi::indxl2g(j, MB_, mycol, iZERO, proc_cols);

                local_matrix_(i, j) = A.elem(index_row, index_col);
            }
        }
    } else if (matrix_type_ == HE) {
        local_matrix_ = Matrix<T>(FORTRAN_STYLE, nrows, ncols);

        for (size_t i = 0; i < nrows; i++) {
            for (size_t j = 0; j < ncols; j++) {
                auto index_row = mpi::indxl2g(i, NB_, myrow, iZERO, proc_rows);
                auto index_col = mpi::indxl2g(j, MB_, mycol, iZERO, proc_cols);
                if (index_col >= index_row) local_matrix_(i, j) = A.elem(index_row, index_col);
            }
        }
    } else if (matrix_type_ == SY) {
    } else {
        std::cerr << "Incorrect matrix type!" << std::endl;
    }
}

template <typename T>
void BLOCKED_Matrix<T>::print_distributed(const std::string& name) const {
    ILP_TYPE rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (matrix_type_ == HE) {
        if (rank == mpi::ROOT_ID) std::cout << "Hermit Matrix!" << std::endl;
    }

    mpi::print_distributed_matrix<T>(local_matrix_, name, ctxt_);
}

template <typename T>
BLOCKED_Matrix<T>::BLOCKED_Matrix(ILP_TYPE ctxt, MATRIX_TYPE type, size_t n, size_t m, T value, size_t NB, size_t MB)
    : n_(n), m_(m), ctxt_(ctxt), matrix_type_(type), NB_(NB), MB_(MB) {
    ILP_TYPE proc_rows, proc_cols, myrow, mycol;
    mpi::blacs_gridinfo(ctxt_, proc_rows, proc_cols, myrow, mycol);
    ILP_TYPE iZERO = 0;

    if (NB == 0) {
        NB_ = n / proc_rows;
    }

    if (MB == 0) {
        MB_ = m / proc_cols;
    }

    if (NB_ == 0) {
        NB_ = 1;
    }

    if (MB_ == 0) {
        MB_ = 1;
    }

    ILP_TYPE nrows = mpi::numroc(n, NB_, myrow, iZERO, proc_rows);
    ILP_TYPE ncols = mpi::numroc(m, MB_, mycol, iZERO, proc_cols);

    local_matrix_ = Matrix<T>(FORTRAN_STYLE, nrows, ncols, value);
}

template <typename T>
BLOCKED_Matrix<T>::BLOCKED_Matrix(ILP_TYPE ctxt, MATRIX_TYPE type, size_t n, size_t m, size_t NB, size_t MB)
    : n_(n), m_(m), ctxt_(ctxt), matrix_type_(type), NB_(NB), MB_(MB) {
    ILP_TYPE proc_rows, proc_cols, myrow, mycol;
    mpi::blacs_gridinfo(ctxt_, proc_rows, proc_cols, myrow, mycol);
    ILP_TYPE iZERO = 0;

    if (NB == 0) {
        NB_ = n / proc_rows;
    }

    if (MB == 0) {
        MB_ = m / proc_cols;
    }

    if (NB_ == 0) {
        NB_ = 1;
    }

    if (MB_ == 0) {
        MB_ = 1;
    }

    ILP_TYPE nrows = mpi::numroc(n, NB_, myrow, iZERO, proc_rows);
    ILP_TYPE ncols = mpi::numroc(m, MB_, mycol, iZERO, proc_cols);

    local_matrix_ = Matrix<T>(FORTRAN_STYLE, nrows, ncols);
}

template <typename T>
ILP_TYPE BLOCKED_Matrix<T>::get_global_row(size_t i) const {
    ILP_TYPE iZERO = 0;
    ILP_TYPE proc_rows, proc_cols, myrow, mycol;
    mpi::blacs_gridinfo(ctxt_, proc_rows, proc_cols, myrow, mycol);

    return mpi::indxl2g(i, NB_, myrow, iZERO, proc_rows);
}

template <typename T>
ILP_TYPE BLOCKED_Matrix<T>::get_global_col(size_t j) const {
    ILP_TYPE iZERO = 0;
    ILP_TYPE proc_rows, proc_cols, myrow, mycol;
    mpi::blacs_gridinfo(ctxt_, proc_rows, proc_cols, myrow, mycol);

    return mpi::indxl2g(j, MB_, mycol, iZERO, proc_cols);
}

template <typename T>
ILP_TYPE BLOCKED_Matrix<T>::get_row_proc(size_t i) const {
    ILP_TYPE iZERO = 0;
    ILP_TYPE proc_rows, proc_cols, myrow, mycol;
    mpi::blacs_gridinfo(ctxt_, proc_rows, proc_cols, myrow, mycol);

    return mpi::indxg2p(i, NB_, myrow, iZERO, proc_rows);
}

template <typename T>
ILP_TYPE BLOCKED_Matrix<T>::get_col_proc(size_t j) const {
    ILP_TYPE iZERO = 0;
    ILP_TYPE proc_rows, proc_cols, myrow, mycol;
    mpi::blacs_gridinfo(ctxt_, proc_rows, proc_cols, myrow, mycol);

    return mpi::indxg2p(j, MB_, mycol, iZERO, proc_cols);
}

template <typename T>
ILP_TYPE BLOCKED_Matrix<T>::get_local_row(size_t i) const {
    ILP_TYPE iZERO = 0;
    ILP_TYPE proc_rows, proc_cols, myrow, mycol;
    mpi::blacs_gridinfo(ctxt_, proc_rows, proc_cols, myrow, mycol);

    return mpi::indxg2l(i, NB_, myrow, iZERO, proc_rows);
}

template <typename T>
ILP_TYPE BLOCKED_Matrix<T>::get_local_col(size_t j) const {
    ILP_TYPE iZERO = 0;
    ILP_TYPE proc_rows, proc_cols, myrow, mycol;
    mpi::blacs_gridinfo(ctxt_, proc_rows, proc_cols, myrow, mycol);

    return mpi::indxg2l(j, MB_, mycol, iZERO, proc_cols);
}

template <typename T>
bool BLOCKED_Matrix<T>::is_my_elem_row(size_t i) const {
    ILP_TYPE proc_rows, proc_cols, myrow, mycol;
    mpi::blacs_gridinfo(ctxt_, proc_rows, proc_cols, myrow, mycol);

    return this->get_row_proc(i) == myrow;
}

template <typename T>
bool BLOCKED_Matrix<T>::is_my_elem_col(size_t j) const {
    ILP_TYPE proc_rows, proc_cols, myrow, mycol;
    mpi::blacs_gridinfo(ctxt_, proc_rows, proc_cols, myrow, mycol);

    return this->get_col_proc(j) == myrow;
}

template <typename T>
BLOCKED_Matrix<T>::BLOCKED_Matrix(const BLOCKED_Matrix<T>& A, const Matrix<T>& local_matrix)
    : matrix_type_(A.matrix_type_),
      ctxt_(A.ctxt_),
      n_(A.n_),
      m_(A.m_),
      NB_(A.NB_),
      MB_(A.MB_),
      local_matrix_(local_matrix) {}

template <typename T>
BLOCKED_Matrix<T> BLOCKED_Matrix<T>::operator+(T num) const {
    BLOCKED_Matrix<T> C(*this, local_matrix_ + num);
    return C;
}

template <typename T>
void BLOCKED_Matrix<T>::operator+=(T num) {
    local_matrix_ += num;
}

template <typename T>
BLOCKED_Matrix<T> BLOCKED_Matrix<T>::operator-(T num) const {
    BLOCKED_Matrix<T> C(*this, local_matrix_ - num);

    return C;
}

template <typename T>
void BLOCKED_Matrix<T>::operator-=(T num) {
    local_matrix_ -= num;
}

template <typename T>
BLOCKED_Matrix<T> BLOCKED_Matrix<T>::operator*(T num) const {
    BLOCKED_Matrix<T> C(*this, local_matrix_ * num);

    return C;
}

template <typename T>
void BLOCKED_Matrix<T>::operator*=(T num) {
    local_matrix_ *= num;
}

template <typename T>
BLOCKED_Matrix<T>::BLOCKED_Matrix(const BLOCKED_Matrix<T>& A, const BLOCKED_Matrix<T>& B)
    : ctxt_(A.ctxt_), n_(A.n_), m_(B.m_), NB_(A.NB_), MB_(B.MB_) {
    ILP_TYPE iZERO = 0;
    matrix_type_ = GE;
    ILP_TYPE proc_rows, proc_cols, myrow, mycol;
    mpi::blacs_gridinfo(ctxt_, proc_rows, proc_cols, myrow, mycol);
    ILP_TYPE nrows = mpi::numroc(n_, NB_, myrow, iZERO, proc_rows);
    ILP_TYPE ncols = mpi::numroc(m_, MB_, mycol, iZERO, proc_cols);
    local_matrix_ = Matrix<T>(FORTRAN_STYLE, nrows, ncols);
}

template <typename T>
void BLOCKED_Matrix<T>::operator+=(const BLOCKED_Matrix<T>& B) {
    *this = *this + B;
}

template <typename T>
void BLOCKED_Matrix<T>::operator*=(const BLOCKED_Matrix<T>& B) {
    *this = *this * B;
}

template <typename T>
void BLOCKED_Matrix<T>::operator-=(const BLOCKED_Matrix<T>& B) {
    *this = *this - B;
}

template <typename T>
BLOCKED_Matrix<T>::BLOCKED_Matrix(ILP_TYPE ctxt,
                                  MATRIX_TYPE type,
                                  size_t n,
                                  size_t m,
                                  std::function<T(size_t, size_t)> func,
                                  size_t NB,
                                  size_t MB)
    : ctxt_(ctxt), matrix_type_(type), n_(n), m_(m) {
    ILP_TYPE iZERO = 0;
    ILP_TYPE proc_rows, proc_cols, myrow, mycol;
    mpi::blacs_gridinfo(ctxt_, proc_rows, proc_cols, myrow, mycol);

    if (NB == 0) {
        NB_ = n / proc_rows;
    } else {
        NB_ = NB;
    }

    if (MB == 0) {
        MB_ = m / proc_cols;
    } else {
        MB_ = MB;
    }

    if (NB_ == 0) {
        NB_ = 1;
    }

    if (MB_ == 0) {
        MB_ = 1;
    }

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

template <typename T>
std::vector<ILP_TYPE> BLOCKED_Matrix<T>::desc() const {
    ILP_TYPE iZERO = 0;
    ILP_TYPE info;
    ILP_TYPE LLD = std::max(1, ILP_TYPE(local_matrix_.n()));
    return mpi::descinit(n_, m_, NB_, MB_, iZERO, iZERO, ctxt_, LLD, info);
}

template <typename T>
void BLOCKED_Matrix<T>::show(size_t width, ILP_TYPE root_id) const {
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

// ------------------------------------------ FUNCTIONS ------------------------------------

std::pair<std::vector<double>, BLOCKED_Matrix<COMPLEX>> Hermit_Lanczos(const BLOCKED_Matrix<COMPLEX>& A);

template <typename T>
void optimized_add(const BLOCKED_Matrix<T>& A, BLOCKED_Matrix<T>& C, T alpha, T betta, char trans_A = 'N');

template <>
void optimized_add(const BLOCKED_Matrix<double>& A,
                   BLOCKED_Matrix<double>& C,
                   double alpha,
                   double betta,
                   char trans_A);

template <>
void optimized_add(const BLOCKED_Matrix<COMPLEX>& A,
                   BLOCKED_Matrix<COMPLEX>& C,
                   COMPLEX alpha,
                   COMPLEX betta,
                   char trans_A);
template <typename T>
void optimized_multiply(const BLOCKED_Matrix<T>& A,
                        const BLOCKED_Matrix<T>& B,
                        BLOCKED_Matrix<T>& C,
                        T alpha,
                        T betta,
                        char trans_A = 'N',
                        char trans_B = 'N');

template <>
void optimized_multiply(const BLOCKED_Matrix<double>& A,
                        const BLOCKED_Matrix<double>& B,
                        BLOCKED_Matrix<double>& C,
                        double alpha,
                        double betta,
                        char trans_A,
                        char trans_B);

template <>
void optimized_multiply(const BLOCKED_Matrix<COMPLEX>& A,
                        const BLOCKED_Matrix<COMPLEX>& B,
                        BLOCKED_Matrix<COMPLEX>& C,
                        COMPLEX alpha,
                        COMPLEX betta,
                        char trans_A,
                        char trans_B);

// SPECIAL FOR QME
std::vector<BLOCKED_Matrix<COMPLEX>> MPI_Runge_Kutt_2(
    const std::vector<double>& x,
    const BLOCKED_Matrix<COMPLEX>& y0,
    std::function<void(double, const BLOCKED_Matrix<COMPLEX>&, BLOCKED_Matrix<COMPLEX>&)> f);

std::vector<BLOCKED_Matrix<COMPLEX>> MPI_Runge_Kutt_4(
    const std::vector<double>& x,
    const BLOCKED_Matrix<COMPLEX>& y0,
    std::function<void(double, const BLOCKED_Matrix<COMPLEX>&, BLOCKED_Matrix<COMPLEX>&)> f);

}  // namespace QComputations

#endif
#endif

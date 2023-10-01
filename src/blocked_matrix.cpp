#ifdef ENABLE_MPI
#ifdef ENABLE_CLUSTER

#include "blocked_matrix.hpp"
//#include <mkl_blacs.h>
//#include <mkl_pblas.h>

namespace QComputations {


template<>
double BLOCKED_Matrix<double>::get(size_t i, size_t j) const {
    return mpi::pdelget(local_matrix_, i, j, this->desc());
}

template<>
COMPLEX BLOCKED_Matrix<COMPLEX>::get(size_t i, size_t j) const {
    return mpi::pzelget(local_matrix_, i, j, this->desc());
}

template<>
BLOCKED_Matrix<double> BLOCKED_Matrix<double>::operator+(const BLOCKED_Matrix<double>& B) const {
    assert(this->matrix_type_ == GE and B.matrix_type_ == GE);
    assert(this->n_ == B.n_ and this->m_ == B.m_);

    BLOCKED_Matrix<double> C(B);

    mpi::parallel_dgeadd(this->get_local_matrix(), C.get_local_matrix(), this->desc(), C.desc());

    return C;
}

template<>
BLOCKED_Matrix<COMPLEX> BLOCKED_Matrix<COMPLEX>::operator+(const BLOCKED_Matrix<COMPLEX>& B) const {
    assert(this->matrix_type_ == GE and B.matrix_type_ == GE);
    assert(this->n_ == B.n_ and this->m_ == B.m_);

    BLOCKED_Matrix<COMPLEX> C(B);

    mpi::parallel_zgeadd(this->get_local_matrix(), C.get_local_matrix(), this->desc(), C.desc());

    return C;
}

template<>
BLOCKED_Matrix<double> BLOCKED_Matrix<double>::operator*(const BLOCKED_Matrix<double>& B) const {
    BLOCKED_Matrix<double> C(*this, B);

    if (this->matrix_type_ == GE and B.matrix_type_ == GE) {
        mpi::parallel_dgemm(this->get_local_matrix(), B.get_local_matrix(), C.get_local_matrix(), this->desc(), B.desc(), C.desc());
    } else {
        std::cerr << "Matrix types error!" << std::endl;
    }
    return C;
}

template<>
BLOCKED_Matrix<COMPLEX> BLOCKED_Matrix<COMPLEX>::operator*(const BLOCKED_Matrix<COMPLEX>& B) const {
    BLOCKED_Matrix<COMPLEX> C(*this, B);

    if (this->matrix_type_ == GE and B.matrix_type_ == GE) {
        mpi::parallel_zgemm(this->get_local_matrix(), B.get_local_matrix(), C.get_local_matrix(), this->desc(), B.desc(), C.desc());
    } else if (this->matrix_type_ == HE and B.matrix_type_ == GE) {
        mpi::parallel_zhemm('L', this->get_local_matrix(), B.get_local_matrix(), C.get_local_matrix(), this->desc(), B.desc(), C.desc());
    } else if (this->matrix_type_ == GE and B.matrix_type_ == HE) {
        mpi::parallel_zhemm('R', B.get_local_matrix(), this->get_local_matrix(), C.get_local_matrix(), this->desc(), B.desc(), C.desc());
    } else {
        std::cerr << "Matrix types error!" << std::endl;
    }

    return C;
}
/*
template<>
BLOCKED_Matrix<double>::BLOCKED_Matrix<double>(ILP_TYPE ctxt, size_t n, size_t m, std::functional<double(size_t, size_t)> func): n_(n), m_(m) {
    ILP_TYPE iZERO = 0;
    ILP_TYPE proc_rows, proc_cols, myrow, mycol;
    blacs_gridinfo(&ctxt, &proc_rows, &proc_cols, &myrow, &mycol);
    NB_ = n / proc_rows;
    MB_ = m / proc_cols;

    ILP_TYPE nrows = numroc_(&n, &NB_, &myrow, &iZERO, &proc_rows);
    ILP_TYPE ncols = numroc_(&m, &MB_, &mycol, &iZERO, &proc_cols);

    local_matrix_ = Matrix<double>(FORTRAN_STYLE, nrows, ncols);

    for (size_t i = 0; i < nrows; i++) {
        for (size_t j = 0; j < ncols; j++) {
            auto index_row = indxl2g_(i, NB_, &myrow, &iZERO, &proc_rows);
            auto index_col = indxl2g_(j, MB_, &mycol, &iZERO, &proc_cols);

            local_matrix_(index_row, index_col) = func(index_row, index_col);
        }
    }
}
*/


// --------------------------------------------- FUNCTIONS -------------------------------------------

std::pair<std::vector<double>, BLOCKED_Matrix<COMPLEX>> mpi::Hermit_Lanzcos(BLOCKED_Matrix<COMPLEX>& A) {
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
}

} // namespace QComputations

#endif
#endif
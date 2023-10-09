#ifdef ENABLE_MPI
#ifdef ENABLE_CLUSTER
#define MKL_Complex16 std::complex<double>

#include "blocked_matrix.hpp"
//#include <mkl_blacs.h>
// #include <mkl_pblas.h>

namespace QComputations {

extern "C" {
    void pzheev(char*, char*, ILP_TYPE*, const COMPLEX*, ILP_TYPE*, ILP_TYPE*,
                const ILP_TYPE*, double*, COMPLEX*, ILP_TYPE*, ILP_TYPE*,
                const ILP_TYPE*, COMPLEX*, ILP_TYPE*, double*, ILP_TYPE*, ILP_TYPE*);
    void pztranc(ILP_TYPE*, ILP_TYPE*, const COMPLEX*, const COMPLEX*, ILP_TYPE*, ILP_TYPE*,
                 ILP_TYPE*, COMPLEX*, COMPLEX*, ILP_TYPE*, ILP_TYPE*, ILP_TYPE*);
    // CALL PZHEEV (jobz, uplo, n, a, ia, ja, desc_a, w, z, iz, jz, desc_z, work, lwork, rwork, lrwork, info)
    //void pzheevx(char*, char*, char*, ILP_TYPE*, COMPLEX*, ILP_TYPE*, ILP_TYPE*, ILP_TYPE*,
    //             double*, double*, ILP_TYPE*, ILP_TYPE*, double*, ILP_TYPE*, ILP_TYPE*, double*, double*,
    //             COMPLEX*, ILP_TYPE*, ILP_TYPE*, ILP_TYPE*, COMPLEX*, ILP_TYPE*, double*, ILP_TYPE*, ILP_TYPE*, ILP_TYPE*,
    //             ILP_TYPE*, ILP_TYPE*, double*, ILP_TYPE*);
    //void pzheevx(jobz, range, uplo, n, a, ia, ja, desca,
    //             vl, vu, il, iu, abstol, m, nz, w, orfac,
    //             z, iz, jz, descz, work, lwork, rwork, lrwork, iwork,
    //             liwork, ifail, iclustr, gap, info)
}

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

template<>
BLOCKED_Matrix<COMPLEX> BLOCKED_Matrix<COMPLEX>::hermit() const {
    if (matrix_type_ == GE) {
        BLOCKED_Matrix<COMPLEX> A(ctxt_, GE, m_, n_);
        COMPLEX alpha(1, 0);
        COMPLEX betta(0, 0);
        ILP_TYPE iONE = 1;
        ILP_TYPE m = m_, n = n_;

        pztranc(&m, &n, &alpha, this->data(),
                &iONE, &iONE, (this->desc()).data(),
                &betta, A.data(), &iONE, &iONE, A.desc().data());

        return A;
    }
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



std::pair<std::vector<double>, BLOCKED_Matrix<COMPLEX>> Hermit_Lanzcos(const BLOCKED_Matrix<COMPLEX>& A) {
    char jobz = 'V';
    char range = 'A';
    char uplo = 'U';

    ILP_TYPE n = A.n();
    ILP_TYPE iONE = 1;
    ILP_TYPE iMINUS = -1;
    ILP_TYPE iZERO = 0;
    ILP_TYPE info;
    ILP_TYPE lwork = (A.local_n() + A.local_m() + A.NB()) * A.NB() + A.n() * 3 + A.n() * A.n();
    ILP_TYPE lrwork = 2 * n + 2 *n - 2;

    BLOCKED_Matrix<COMPLEX> Z(A.ctxt(), GE, A.n(), A.m(), A.NB(), A.MB());
    std::vector<double>w(A.n());
    std::vector<COMPLEX>work(lwork);
    std::vector<double>rwork(lrwork);
    pzheev(&jobz, &uplo, &n, A.data(), &iONE, &iONE,
        A.desc().data(), w.data(), Z.data(), &iONE, &iONE,
        Z.desc().data(), work.data(), &lwork, rwork.data(), &lrwork, &info);

    return std::make_pair(w, Z);
}


} // namespace QComputations

#endif
#endif
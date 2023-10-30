#ifdef ENABLE_MPI
#ifdef ENABLE_CLUSTER

#include "blocked_vector.hpp"
#include <mkl_blacs.h>
#include <mkl_scalapack.h>
#include "mpi_functions.hpp"
#include <mpi.h>
#include <mkl_pblas.h>

namespace QComputations {

void mpi::init_vector_grid(ILP_TYPE& ctxt, ILP_TYPE proc_rows, ILP_TYPE proc_cols) {
    ILP_TYPE iZERO = 0;
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ILP_TYPE myid, numproc, myrow, mycol;
    char order = 'R';
    if (proc_rows == 0 or proc_cols == 0) {
        proc_rows = world_size;
        proc_cols = 1;
    }
    //std::cout << rank << " Here1\n";
    blacs_pinfo(&myid, &numproc);
    ILP_TYPE iMINUS = -1;
    blacs_get(&iMINUS, &iZERO, &ctxt);
    //std::cout << rank << " Here3\n";
    blacs_gridinit(&ctxt, &order, &proc_rows, &proc_cols);
}

template<>
BLOCKED_Vector<COMPLEX> BLOCKED_Vector<COMPLEX>::operator*(const BLOCKED_Matrix<COMPLEX>& A) const {
    BLOCKED_Vector<COMPLEX> y(*this, A);

    mpi::parallel_zgemv(A.data(), this->data(), y.data(), A.desc(), this->desc(), y.desc());

    return y;
}

template<>
BLOCKED_Vector<double> BLOCKED_Vector<double>::operator*(const BLOCKED_Matrix<double>& A) const {
    BLOCKED_Vector<double> y(*this, A);

    mpi::parallel_dgemv(A.data(), this->data(), y.data(), A.desc(), this->desc(), y.desc());

    return y;
}

template<>
BLOCKED_Vector<COMPLEX> BLOCKED_Vector<COMPLEX>::operator*(COMPLEX num) const {
    BLOCKED_Vector<COMPLEX> res(*this);

    mpi::parallel_zscal(res.data(), num, res.desc(), res.inc());

    return res;
}

template<>
BLOCKED_Vector<double> BLOCKED_Vector<double>::operator*(double num) const {
    BLOCKED_Vector<double> res(*this);

    mpi::parallel_dscal(res.data(), num, res.desc(), res.inc());

    return res;
}

// ------------------------------ FUNCTIONS -------------------------------------

COMPLEX scalar_produc t(const BLOCKED_Vector<COMPLEX>& a, const BLOCKED_Vector<COMPLEX>& b) const {
    return mpi::parallel_zdotu(a.data(), b.data(), a.desc(), a.inc(), b.desc(), b.inc());
}

double scalar_product (const BLOCKED_Vector<COMPLEX>& a, const BLOCKED_Vector<double>& b) const {
    return mpi::parallel_dot(a.data(), b.data(), a.desc(), a.inc(), b.desc(), b.inc());
}

BLOCKED_Vector<COMPLEX> blocked_matrix_get_col(ILP_TYPE ctxt, const BLOCKED_Matrix<COMPLEX>& A, size_t col) {
    BLOCKED_Vector<COMPLEX> res(ctxt, A.n());

    for (size_t i = 0; i < A.n(); i++) {
        res.set(i, A.get(i, col));
    }

    return res;
}

BLOCKED_Vector<double> blocked_matrix_get_col(ILP_TYPE ctxt, const BLOCKED_Matrix<double>& A, size_t col) {
    BLOCKED_Vector<double> res(ctxt, A.n());

    for (size_t i = 0; i < A.n(); i++) {
        res.set(i, A.get(i, col));
    }

    return res;
}

} // namespace QComputations

#endif
#endif
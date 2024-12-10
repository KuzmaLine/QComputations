#ifdef ENABLE_MPI
#ifdef ENABLE_CLUSTER

#include "blocked_vector.hpp"

#include <mkl_blacs.h>
#include <mkl_pblas.h>
#include <mkl_scalapack.h>
#include <mpi.h>

#include "mpi_functions.hpp"

namespace QComputations {

    template <>
    BLOCKED_Vector<COMPLEX> BLOCKED_Vector<COMPLEX>::operator*(COMPLEX num) const {
        BLOCKED_Vector<COMPLEX> res(*this);

        mpi::parallel_zscal(res.get_local_matrix().get_mass(), num, res.desc(), res.inc());

        return res;
    }

    template <>
    BLOCKED_Vector<double> BLOCKED_Vector<double>::operator*(double num) const {
        BLOCKED_Vector<double> res(*this);

        mpi::parallel_dscal(res.get_local_matrix().get_mass(), num, res.desc(), res.inc());

        return res;
    }

    template <>
    BLOCKED_Vector<double> BLOCKED_Vector<double>::operator+(const BLOCKED_Vector<double>& x) const {
        BLOCKED_Vector<double> res(*this);

        mpi::parallel_daxpy(x.get_local_matrix().get_mass(),
                            res.get_local_matrix().get_mass(),
                            x.desc(),
                            x.inc(),
                            res.desc(),
                            res.inc(),
                            1.0);

        return res;
    }

    template <>
    void BLOCKED_Vector<double>::operator+=(const BLOCKED_Vector<double>& x) {
        mpi::parallel_daxpy(x.get_local_matrix().get_mass(),
                            this->get_local_matrix().get_mass(),
                            x.desc(),
                            x.inc(),
                            this->desc(),
                            this->inc(),
                            1.0);
    }

    template <>
    BLOCKED_Vector<COMPLEX> BLOCKED_Vector<COMPLEX>::operator+(const BLOCKED_Vector<COMPLEX>& x) const {
        BLOCKED_Vector<COMPLEX> res(*this);

        mpi::parallel_zaxpy(x.get_local_matrix().get_mass(),
                            res.get_local_matrix().get_mass(),
                            x.desc(),
                            x.inc(),
                            res.desc(),
                            res.inc(),
                            1.0);

        return res;
    }

    template <>
    void BLOCKED_Vector<COMPLEX>::operator+=(const BLOCKED_Vector<COMPLEX>& x) {
        mpi::parallel_zaxpy(x.get_local_matrix().get_mass(),
                            this->get_local_matrix().get_mass(),
                            x.desc(),
                            x.inc(),
                            this->desc(),
                            this->inc(),
                            COMPLEX(1.0, 0));
    }

    // ------------------------------ FUNCTIONS -------------------------------------

    COMPLEX scalar_product(const BLOCKED_Vector<COMPLEX>& a, const BLOCKED_Vector<COMPLEX>& b) {
        return mpi::parallel_zdotc(
            a.get_local_matrix().get_mass(), b.get_local_matrix().get_mass(), a.desc(), a.inc(), b.desc(), b.inc());
    }

    double scalar_product(const BLOCKED_Vector<double>& a, const BLOCKED_Vector<double>& b) {
        return mpi::parallel_ddot(
            a.get_local_matrix().get_mass(), b.get_local_matrix().get_mass(), a.desc(), a.inc(), b.desc(), b.inc());
    }

    BLOCKED_Vector<double> blocked_matrix_get_row(ILP_TYPE ctxt, const BLOCKED_Matrix<double>& A, size_t row) {
        BLOCKED_Vector<double> res(ctxt, A.m());

        for (size_t i = 0; i < A.m(); i++) {
            res.set(i, A.get(row, i));
        }

        return res;
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

    std::vector<BLOCKED_Vector<double>> blocked_matrix_to_blocked_vectors(ILP_TYPE ctxt,
                                                                          const BLOCKED_Matrix<double>& A) {
        std::vector<BLOCKED_Vector<double>> res;

        for (size_t i = 0; i < A.m(); i++) {
            res.push_back(blocked_matrix_get_col(ctxt, A, i));
        }

        return res;
    }
    std::vector<BLOCKED_Vector<COMPLEX>> blocked_matrix_to_blocked_vectors(ILP_TYPE ctxt,
                                                                           const BLOCKED_Matrix<COMPLEX>& A) {
        std::vector<BLOCKED_Vector<COMPLEX>> res;

        for (size_t i = 0; i < A.m(); i++) {
            res.push_back(blocked_matrix_get_col(ctxt, A, i));
        }

        return res;
    }

}  // namespace QComputations

#endif
#endif
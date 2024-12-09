#pragma once

#ifdef ENABLE_MPI

#include <mpi.h>

#include <complex>
#include <functional>
#include <iostream>
#include <map>

#include "state.hpp"

namespace {
#ifdef MKL_ILP64
using ILP_TYPE = long long;
constexpr MPI_Datatype MPI_BCAST_DATATYPE = MPI_LONG_LONG;
#else
using ILP_TYPE = int;
constexpr MPI_Datatype MPI_BCAST_DATATYPE = MPI_INT;
#endif

using COMPLEX = std::complex<double>;
}  // namespace

namespace QComputations {

#ifdef ENABLE_CLUSTER
bool is_main_proc();

bool get_proc_rank();
#endif

namespace mpi {
struct MPI_Data {
    size_t n;
    std::function<COMPLEX(size_t, size_t)> func;
    TCH_State state;
    std::vector<double> timeline;
};

namespace MPI_Datatype_ID {
constexpr int INT = 1;
constexpr int DOUBLE = 2;
constexpr int DOUBLE_COMPLEX = 3;
}  // namespace MPI_Datatype_ID

constexpr int ROOT_ID = 0;

std::vector<COMPLEX> bcast_vector_complex(const std::vector<COMPLEX>& v = {});
std::vector<double> bcast_vector_double(const std::vector<double>& v = {});

#ifdef ENABLE_CLUSTER

// Печатает блок матрицы каждого процессора.
template <typename T>
void print_distributed_matrix(const Matrix<T>& A, const std::string& matrix_name, ILP_TYPE ctxt) {
    int myid, numproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &numproc);

    for (ILP_TYPE id = 0; id < numproc; ++id) {
        if (id == myid) {
            std::cout << matrix_name << " on proc " << myid << std::endl;
            A.show();
            std::flush(std::cout);
            std::cout << std::endl;
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }
}

// ON BLACS GRID
// Пытается подогнать под квадратную решётку
void init_grid(ILP_TYPE& ctxt, ILP_TYPE proc_rows = 0, ILP_TYPE proc_cols = 0);
// Пытается выстроить процессы в ряд
void init_vector_grid(ILP_TYPE& ctxt, ILP_TYPE proc_rows = 0, ILP_TYPE proc_cols = 0);
// Получить информацию о решётке
void blacs_gridinfo(const ILP_TYPE& ctxt, ILP_TYPE& proc_rows, ILP_TYPE& proc_cols, ILP_TYPE& myrow, ILP_TYPE& mycol);
ILP_TYPE numroc(ILP_TYPE N, ILP_TYPE NB, ILP_TYPE myindex, ILP_TYPE ZERO, ILP_TYPE size);
ILP_TYPE indxl2g(ILP_TYPE N, ILP_TYPE NB, ILP_TYPE myindx, ILP_TYPE RSRC, ILP_TYPE dim_counts);
ILP_TYPE indxg2p(ILP_TYPE N, ILP_TYPE NB, ILP_TYPE myindx, ILP_TYPE RSRC, ILP_TYPE dim_counts);
ILP_TYPE indxg2l(ILP_TYPE N, ILP_TYPE NB, ILP_TYPE myindx, ILP_TYPE RSRC, ILP_TYPE dim_counts);
void blacs_gridexit(ILP_TYPE& ctxt);
double pdelget(const Matrix<double>& A, ILP_TYPE i, ILP_TYPE j, const std::vector<ILP_TYPE>& desc);
COMPLEX pzelget(const Matrix<COMPLEX>& A, ILP_TYPE i, ILP_TYPE j, const std::vector<ILP_TYPE>& desc);
void pdelset(Matrix<double>& A, ILP_TYPE i, ILP_TYPE j, double num, const std::vector<ILP_TYPE>& desc);
void pzelset(Matrix<COMPLEX>& A, ILP_TYPE i, ILP_TYPE j, COMPLEX num, const std::vector<ILP_TYPE>& desc);

// Распределяет по Blacs решётке блоки матрицы. Реализовано только для Matrix<double> и Matrix<std::complex<double>>
template <typename T>
Matrix<T> scatter_blacs_matrix(const Matrix<T>& A,
                               ILP_TYPE& N,
                               ILP_TYPE& M,
                               ILP_TYPE& NB,
                               ILP_TYPE& MB,
                               ILP_TYPE& nrows,
                               ILP_TYPE& ncols,
                               ILP_TYPE& ctxt,
                               ILP_TYPE root_id,
                               MPI_Comm comm = MPI_COMM_WORLD);

template <>
Matrix<double> scatter_blacs_matrix<double>(const Matrix<double>& A,
                                            ILP_TYPE& N,
                                            ILP_TYPE& M,
                                            ILP_TYPE& NB,
                                            ILP_TYPE& MB,
                                            ILP_TYPE& nrows,
                                            ILP_TYPE& ncols,
                                            ILP_TYPE& ctxt,
                                            ILP_TYPE root_id,
                                            MPI_Comm comm);
template <>
Matrix<COMPLEX> scatter_blacs_matrix<COMPLEX>(const Matrix<COMPLEX>& A,
                                              ILP_TYPE& N,
                                              ILP_TYPE& M,
                                              ILP_TYPE& NB,
                                              ILP_TYPE& MB,
                                              ILP_TYPE& nrows,
                                              ILP_TYPE& ncols,
                                              ILP_TYPE& ctxt,
                                              ILP_TYPE root_id,
                                              MPI_Comm comm);

// Распределяет обратно.
template <typename T>
void gather_blacs_matrix(const Matrix<T>& localC,
                         Matrix<T>& C,
                         ILP_TYPE N,
                         ILP_TYPE M,
                         ILP_TYPE NB,
                         ILP_TYPE MB,
                         ILP_TYPE nrows,
                         ILP_TYPE ncols,
                         ILP_TYPE ctxt,
                         ILP_TYPE root_id);

template <>
void gather_blacs_matrix<double>(const Matrix<double>& localC,
                                 Matrix<double>& C,
                                 ILP_TYPE N,
                                 ILP_TYPE M,
                                 ILP_TYPE NB,
                                 ILP_TYPE MB,
                                 ILP_TYPE nrows,
                                 ILP_TYPE ncols,
                                 ILP_TYPE ctxt,
                                 ILP_TYPE root_id);
template <>
void gather_blacs_matrix<COMPLEX>(const Matrix<COMPLEX>& localC,
                                  Matrix<COMPLEX>& C,
                                  ILP_TYPE N,
                                  ILP_TYPE M,
                                  ILP_TYPE NB,
                                  ILP_TYPE MB,
                                  ILP_TYPE nrows,
                                  ILP_TYPE ncols,
                                  ILP_TYPE ctxt,
                                  ILP_TYPE root_id);

template <typename T>
std::vector<T> scatter_blacs_vector(const std::vector<T>& v,
                                    ILP_TYPE& N,
                                    ILP_TYPE& NB,
                                    ILP_TYPE& nrows,
                                    ILP_TYPE& ctxt,
                                    ILP_TYPE root_id,
                                    MPI_Comm comm = MPI_COMM_WORLD);

template <>
std::vector<double> scatter_blacs_vector<double>(const std::vector<double>& v,
                                                 ILP_TYPE& N,
                                                 ILP_TYPE& NB,
                                                 ILP_TYPE& nrows,
                                                 ILP_TYPE& ctxt,
                                                 ILP_TYPE root_id,
                                                 MPI_Comm comm);

template <>
std::vector<COMPLEX> scatter_blacs_vector<COMPLEX>(const std::vector<COMPLEX>& v,
                                                   ILP_TYPE& N,
                                                   ILP_TYPE& NB,
                                                   ILP_TYPE& nrows,
                                                   ILP_TYPE& ctxt,
                                                   ILP_TYPE root_id,
                                                   MPI_Comm comm);

template <typename T>
void gather_blacs_vector(const std::vector<T>& local_y,
                         std::vector<T>& y,
                         ILP_TYPE N,
                         ILP_TYPE NB,
                         ILP_TYPE nrows,
                         ILP_TYPE ctxt,
                         ILP_TYPE root_id);

template <>
void gather_blacs_vector<double>(const std::vector<double>& local_y,
                                 std::vector<double>& y,
                                 ILP_TYPE N,
                                 ILP_TYPE NB,
                                 ILP_TYPE nrows,
                                 ILP_TYPE ctxt,
                                 ILP_TYPE root_id);

template <>
void gather_blacs_vector<COMPLEX>(const std::vector<COMPLEX>& local_y,
                                  std::vector<COMPLEX>& y,
                                  ILP_TYPE N,
                                  ILP_TYPE NB,
                                  ILP_TYPE nrows,
                                  ILP_TYPE ctxt,
                                  ILP_TYPE root_id);

#endif

#ifdef ENABLE_CLUSTER

std::vector<ILP_TYPE> descinit(ILP_TYPE n,
                               ILP_TYPE m,
                               ILP_TYPE NB,
                               ILP_TYPE MB,
                               ILP_TYPE rsrc,
                               ILP_TYPE csrc,
                               ILP_TYPE ctxt,
                               ILP_TYPE LLD,
                               ILP_TYPE info);

// only for n x n matrix
template <typename T>
std::vector<T> get_diagonal_elements(Matrix<T>& localA, const std::vector<ILP_TYPE>& desca);

template <>
std::vector<double> get_diagonal_elements<double>(Matrix<double>& localA, const std::vector<ILP_TYPE>& desca);

template <>
std::vector<COMPLEX> get_diagonal_elements<COMPLEX>(Matrix<COMPLEX>& localA, const std::vector<ILP_TYPE>& desca);

// Parallel matrix computations
void parallel_dgeadd(const Matrix<double>& A,
                     Matrix<double>& C,
                     const std::vector<ILP_TYPE>& desca,
                     const std::vector<ILP_TYPE>& descc,
                     double alpha = 1.0,
                     double betta = 1.0,
                     char op_A = 'N');

// N - not trans. T - trans. C - hermit
void parallel_zgeadd(const Matrix<COMPLEX>& A,
                     Matrix<COMPLEX>& C,
                     const std::vector<ILP_TYPE>& desca,
                     const std::vector<ILP_TYPE>& descc,
                     COMPLEX alpha = COMPLEX(1, 0),
                     COMPLEX betta = COMPLEX(1, 0),
                     char op_A = 'N');

void parallel_dgemm(const Matrix<double>& A,
                    const Matrix<double>& B,
                    Matrix<double>& C,
                    const std::vector<ILP_TYPE>& desca,
                    const std::vector<ILP_TYPE>& descb,
                    const std::vector<ILP_TYPE>& descc,
                    double alpha = 1,
                    double betta = 0,
                    char op_A = 'N',
                    char op_B = 'N');

// N - not trans. T - trans. C - hermit
void parallel_zgemm(const Matrix<COMPLEX>& A,
                    const Matrix<COMPLEX>& B,
                    Matrix<COMPLEX>& C,
                    const std::vector<ILP_TYPE>& desca,
                    const std::vector<ILP_TYPE>& descb,
                    const std::vector<ILP_TYPE>& descc,
                    COMPLEX alpha = COMPLEX(1, 0),
                    COMPLEX betta = COMPLEX(0, 0),
                    char op_A = 'N',
                    char op_B = 'N');

void parallel_zhemm(char side,
                    const Matrix<COMPLEX>& A,
                    const Matrix<COMPLEX>& B,
                    Matrix<COMPLEX>& C,
                    const std::vector<ILP_TYPE>& desca,
                    const std::vector<ILP_TYPE>& descb,
                    const std::vector<ILP_TYPE>& descc,
                    COMPLEX alpha = COMPLEX(1, 0),
                    COMPLEX betta = COMPLEX(0, 0));

void parallel_dgemv(const Matrix<double>& A,
                    const std::vector<double>& x,
                    std::vector<double>& y,
                    const std::vector<ILP_TYPE>& desca,
                    const std::vector<ILP_TYPE>& descx,
                    const std::vector<ILP_TYPE>& descy,
                    char op_A = 'N');
void parallel_zgemv(const Matrix<COMPLEX>& A,
                    const std::vector<COMPLEX>& x,
                    std::vector<COMPLEX>& y,
                    const std::vector<ILP_TYPE>& desca,
                    const std::vector<ILP_TYPE>& descx,
                    const std::vector<ILP_TYPE>& descy,
                    char op_A = 'N');

// x*y (without conjugate)
COMPLEX parallel_zdotu(const std::vector<COMPLEX>& x,
                       const std::vector<COMPLEX>& y,
                       const std::vector<ILP_TYPE>& descx,
                       ILP_TYPE incx,
                       const std::vector<ILP_TYPE>& descy,
                       ILP_TYPE incy);

// <x|y>
COMPLEX parallel_zdotc(const std::vector<COMPLEX>& x,
                       const std::vector<COMPLEX>& y,
                       const std::vector<ILP_TYPE>& descx,
                       ILP_TYPE incx,
                       const std::vector<ILP_TYPE>& descy,
                       ILP_TYPE incy);
double parallel_ddot(const std::vector<double>& x,
                     const std::vector<double>& y,
                     const std::vector<ILP_TYPE>& descx,
                     ILP_TYPE incx,
                     const std::vector<ILP_TYPE>& descy,
                     ILP_TYPE incy);
COMPLEX parallel_zscal(std::vector<COMPLEX>& x, COMPLEX a, const std::vector<ILP_TYPE>& descx, ILP_TYPE incx);
double parallel_dscal(std::vector<double>& x, double a, const std::vector<ILP_TYPE>& descx, ILP_TYPE incx);
void parallel_daxpy(const std::vector<double>& x,
                    std::vector<double>& y,
                    const std::vector<ILP_TYPE>& descx,
                    ILP_TYPE incx,
                    const std::vector<ILP_TYPE>& descy,
                    ILP_TYPE incy,
                    double alpha = 1.0);
void parallel_zaxpy(const std::vector<COMPLEX>& x,
                    std::vector<COMPLEX>& y,
                    const std::vector<ILP_TYPE>& descx,
                    ILP_TYPE incx,
                    const std::vector<ILP_TYPE>& descy,
                    ILP_TYPE incy,
                    COMPLEX alpha = COMPLEX(1, 0));

#endif
}  // namespace mpi

}  // namespace QComputations

#endif  // ENABLE_MPI

#pragma once

#ifdef ENABLE_MPI

#include <mpi.h>
//#include </home/kuzmaline/.local/include/mpi.h>
//#include </usr/lib/x86_64-linux-gnu/openmpi/include/mpi.h>
#include <iostream>
#include <complex>
#include "state.hpp"
#include <map>
#include <functional>

namespace {
#ifdef MKL_ILP64
    using ILP_TYPE = long long;
    constexpr MPI_Datatype MPI_BCAST_DATATYPE = MPI_LONG_LONG;
#else
    using ILP_TYPE = int;
    constexpr MPI_Datatype MPI_BCAST_DATATYPE = MPI_INT;
#endif

    using COMPLEX = std::complex<double>;
}

namespace QComputations {

// COMMAND LIST
namespace COMMAND {
    constexpr int COMMANDS_COUNT = 11;

    enum COMMANDS { STOP = 100, GENERATE_H = 101, GENERATE_H_FUNC = 102,
                    SCHRODINGER = 103, QME = 104, CANNON_MULTIPLY = 105,
                    MATVEC = 106, MATNUM = 107, EXIT_FROM_FUNC = 108,
                    DIM_MULTIPLY = 109, P_GEMM_MULTIPLY = 110};
    // UNUSED, DELETE
    namespace DIM {
        constexpr int ROW = 0;
        constexpr int COL = 1;
    }
}

namespace mpi {
    struct MPI_Data {
        size_t n;
        std::function<COMPLEX(size_t, size_t)> func;
        State state;
        std::vector<double> timeline;
    };

    namespace MPI_Datatype_ID {
        constexpr int INT = 1;
        constexpr int DOUBLE = 2;
        constexpr int DOUBLE_COMPLEX = 3;
    }

    constexpr int ROOT_ID = 0;

    // Send command to other process
    void make_command(COMMAND::COMMANDS command);

    std::vector<COMPLEX> bcast_vector_complex(const std::vector<COMPLEX>& v = {});
    std::vector<double> bcast_vector_double(const std::vector<double>& v = {});
    State bcast_state(const State& state = State());


#ifdef ENABLE_CLUSTER

    // Печатает блок матрицы каждого процессора.
    template<typename T>
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

            //blacs_barrier(&ctxt, "All");
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    // ON BLACS GRID
    void init_grid(ILP_TYPE& ctxt, ILP_TYPE proc_rows = 0, ILP_TYPE proc_cols = 0);
    void init_vector_grid(ILP_TYPE& ctxt, ILP_TYPE proc_rows = 0, ILP_TYPE proc_cols = 0);
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
    template<typename T>
    Matrix<T> scatter_blacs_matrix(const Matrix<T>& A, ILP_TYPE& N, ILP_TYPE& M,
                             ILP_TYPE& NB, ILP_TYPE& MB, ILP_TYPE& nrows,
                             ILP_TYPE& ncols, ILP_TYPE& ctxt, ILP_TYPE root_id,
                             MPI_Comm comm = MPI_COMM_WORLD);

    template<>
    Matrix<double> scatter_blacs_matrix<double>(const Matrix<double>& A, ILP_TYPE& N, ILP_TYPE& M,
                                      ILP_TYPE& NB, ILP_TYPE& MB, ILP_TYPE& nrows,
                                      ILP_TYPE& ncols, ILP_TYPE& ctxt, ILP_TYPE root_id,
                                      MPI_Comm comm);
    template<>
    Matrix<COMPLEX> scatter_blacs_matrix<COMPLEX>(const Matrix<COMPLEX>& A, ILP_TYPE& N, ILP_TYPE& M,
                                       ILP_TYPE& NB, ILP_TYPE& MB, ILP_TYPE& nrows,
                                       ILP_TYPE& ncols, ILP_TYPE& ctxt, ILP_TYPE root_id,
                                       MPI_Comm comm);
    
    // Распределяет обратно.
    template<typename T>
    void gather_blacs_matrix(const Matrix<T>& localC, Matrix<T>& C, ILP_TYPE N, ILP_TYPE M,
                             ILP_TYPE NB, ILP_TYPE MB, ILP_TYPE nrows,
                             ILP_TYPE ncols, ILP_TYPE ctxt, ILP_TYPE root_id);

    template<>
    void gather_blacs_matrix<double>(const Matrix<double>& localC, Matrix<double>& C, ILP_TYPE N, ILP_TYPE M,
                                     ILP_TYPE NB, ILP_TYPE MB, ILP_TYPE nrows,
                                     ILP_TYPE ncols, ILP_TYPE ctxt, ILP_TYPE root_id);
    template<>
    void gather_blacs_matrix<COMPLEX>(const Matrix<COMPLEX>& localC, Matrix<COMPLEX>& C, ILP_TYPE N, ILP_TYPE M,
                                      ILP_TYPE NB, ILP_TYPE MB, ILP_TYPE nrows,
                                      ILP_TYPE ncols, ILP_TYPE ctxt, ILP_TYPE root_id);

    template<typename T>
    std::vector<T> scatter_blacs_vector(const std::vector<T>& v, ILP_TYPE& N,
                             ILP_TYPE& NB, ILP_TYPE& nrows, ILP_TYPE& ctxt,
                             ILP_TYPE root_id, MPI_Comm comm = MPI_COMM_WORLD);

    template<>
    std::vector<double> scatter_blacs_vector<double>(const std::vector<double>& v, ILP_TYPE& N,
                             ILP_TYPE& NB, ILP_TYPE& nrows, ILP_TYPE& ctxt,
                             ILP_TYPE root_id, MPI_Comm comm);
    
    template<>
    std::vector<COMPLEX> scatter_blacs_vector<COMPLEX>(const std::vector<COMPLEX>& v, ILP_TYPE& N,
                             ILP_TYPE& NB, ILP_TYPE& nrows, ILP_TYPE& ctxt,
                             ILP_TYPE root_id, MPI_Comm comm);

    template<typename T>
    void gather_blacs_vector(const std::vector<T>& local_y, std::vector<T>& y, ILP_TYPE N,
                             ILP_TYPE NB, ILP_TYPE nrows, ILP_TYPE ctxt, ILP_TYPE root_id);
    
    template<>
    void gather_blacs_vector<double>(const std::vector<double>& local_y, std::vector<double>& y, ILP_TYPE N,
                             ILP_TYPE NB, ILP_TYPE nrows, ILP_TYPE ctxt, ILP_TYPE root_id);

    template<>
    void gather_blacs_vector<COMPLEX>(const std::vector<COMPLEX>& local_y, std::vector<COMPLEX>& y, ILP_TYPE N,
                             ILP_TYPE NB, ILP_TYPE nrows, ILP_TYPE ctxt, ILP_TYPE root_id);

#endif

    // Stay to wait all MPI process until root process give commands. MPI_Init included
    void run_mpi_slaves(const std::map<int, std::vector<MPI_Data>>& data = {}); 

    // Stop MPI. MPI_Finalize() included
    void stop_mpi_slaves();

    /*
    void Cannon_Multiply(const Matrix<COMPLEX>& A, const Matrix<COMPLEX>& B, Matrix<COMPLEX>& C, int grid_size, int block_size, int n);
    void Cannon_Multiply(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C, int grid_size, int block_size, int n);
    void Cannon_Multiply(const Matrix<int>& A, const Matrix<int>& B, Matrix<int>& C, int grid_size, int block_size, int n);
    */

   template<typename T>
   void Cannon_Multiply(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C, int grid_size, int block_size, int n);

   template<>
   void Cannon_Multiply<COMPLEX>(const Matrix<COMPLEX>& A, const Matrix<COMPLEX>& B, Matrix<COMPLEX>& C, int grid_size, int block_size, int n);
   template<>
   void Cannon_Multiply<int>(const Matrix<int>& A, const Matrix<int>& B, Matrix<int>& C, int grid_size, int block_size, int n);
   template<>
   void Cannon_Multiply<double>(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C, int grid_size, int block_size, int n);

   template<typename T>
   void Dim_Multiply(const Matrix<T>& A, const Matrix<T>& B, Matrix<T>& C);

   template<>
   void Dim_Multiply<COMPLEX>(const Matrix<COMPLEX>& A, const Matrix<COMPLEX>& B, Matrix<COMPLEX>& C);

    /*
    template<typename T, typename V>
    std::vector<V> MPI_Runge_Kutt_4(const std::vector<T>& x, const V& y0,
                                    std::function<V(T, V)> f, int* proc_group,
                                    MPI_Comm comm) {
        int rank, numproc;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &numproc);
        size_t len = x.size();
        std::vector<V> y(len);
        y[0] = y0;

        for (size_t i = 0; i < len - 1; i++) {
            if (i % (len / 100) == 0) std::cout << i << " " << len << std::endl;
            //std::cout << i << " " << y[i] << " ";
            T h = x[i + 1] - x[i];

            if (rank <= proc_group[0]) V k1 = f(x[i], y[i]);
            if (proc_group[0] < rank and rank <= proc_group[1]) V k2 = f(x[i] + h / 2.0, y[i] + k1 * (h / 2.0));
            if (proc_group[1] < rank and rank <= proc_group[2]) V k3 = f(x[i] + h / 2.0, y[i] + k2 * (h / 2.0));
            if (proc_group[2] < rank and rank <= proc_group[3]) V k4 = f(x[i] + h, y[i] + k3 * h);
            
            y[i + 1] = y[i]  + (k1 + (k2 + k3) * 2 + k4) * (h / 6.0);
            //std::cout << h << " " << y[i + 1] << " " << 2 * x[i + 1] << std::endl;
            MPI_Barrier(MPI_COMM_WORLD);
        }

        return y;
    }
    */

#ifdef ENABLE_CLUSTER
    std::vector<ILP_TYPE> descinit(ILP_TYPE n, ILP_TYPE m, ILP_TYPE NB,
                                   ILP_TYPE MB, ILP_TYPE rsrc, ILP_TYPE csrc,
                                   ILP_TYPE ctxt, ILP_TYPE LLD, ILP_TYPE info);

    // only for n x n matrix
    template<typename T>
    std::vector<T> get_diagonal_elements(Matrix<T>& localA, const std::vector<ILP_TYPE>& desca);

    template<>
    std::vector<double> get_diagonal_elements<double>(Matrix<double>& localA, const std::vector<ILP_TYPE>& desca);

    template<>
    std::vector<COMPLEX> get_diagonal_elements<COMPLEX>(Matrix<COMPLEX>& localA, const std::vector<ILP_TYPE>& desca);

   // Parallel matrix computations
    void parallel_dgeadd(const Matrix<double>& A, Matrix<double>& C,
                        const std::vector<ILP_TYPE>& desca,
                        const std::vector<ILP_TYPE>& descc,
                        double alpha = 1.0, double betta = 1.0,
                        char op_A = 'N');

    void parallel_zgeadd(const Matrix<COMPLEX>& A, Matrix<COMPLEX>& C,
                        const std::vector<ILP_TYPE>& desca,
                        const std::vector<ILP_TYPE>& descc,
                        COMPLEX alpha = COMPLEX(1, 0), COMPLEX betta = COMPLEX(1, 0),
                        char op_A = 'N');

    void parallel_dgemm(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C,
                        const std::vector<ILP_TYPE>& desca,
                        const std::vector<ILP_TYPE>& descb, const std::vector<ILP_TYPE>& descc,
                        char op_A = 'N', char op_B = 'N');

    void parallel_zgemm(const Matrix<COMPLEX>& A, const Matrix<COMPLEX>& B, Matrix<COMPLEX>& C,
                        const std::vector<ILP_TYPE>& desca,
                        const std::vector<ILP_TYPE>& descb, const std::vector<ILP_TYPE>& descc,
                        char op_A = 'N', char op_B = 'N');
        
    void parallel_zhemm(char side, const Matrix<COMPLEX>& A, const Matrix<COMPLEX>& B, Matrix<COMPLEX>& C,
                        const std::vector<ILP_TYPE>& desca,
                        const std::vector<ILP_TYPE>& descb, const std::vector<ILP_TYPE>& descc,
                        char op_A = 'N', char op_B = 'N');

    void parallel_dgemv(const Matrix<double>& A, const std::vector<double>& x, std::vector<double>& y,
                        const std::vector<ILP_TYPE>& desca,
                        const std::vector<ILP_TYPE>& descx, const std::vector<ILP_TYPE>& descy,
                        char op_A = 'N');
    void parallel_zgemv(const Matrix<COMPLEX>& A, const std::vector<COMPLEX>& x, std::vector<COMPLEX>& y,
                        const std::vector<ILP_TYPE>& desca,
                        const std::vector<ILP_TYPE>& descx, const std::vector<ILP_TYPE>& descy,
                        char op_A = 'N');
    COMPLEX parallel_zdotu(const std::vector<COMPLEX>& x, const std::vector<COMPLEX>& y,
                        const std::vector<ILP_TYPE>& descx, ILP_TYPE incx,
                        const std::vector<ILP_TYPE>& descy, ILP_TYPE incy);
    COMPLEX parallel_zdotc(const std::vector<COMPLEX>& x, const std::vector<COMPLEX>& y,
                    const std::vector<ILP_TYPE>& descx, ILP_TYPE incx,
                    const std::vector<ILP_TYPE>& descy, ILP_TYPE incy);
    double parallel_ddot(const std::vector<double>& x, const std::vector<double>& y,
                        const std::vector<ILP_TYPE>& descx, ILP_TYPE incx,
                        const std::vector<ILP_TYPE>& descy, ILP_TYPE incy);
    COMPLEX parallel_zscal(std::vector<COMPLEX>& x, COMPLEX a,
                    const std::vector<ILP_TYPE>& descx, ILP_TYPE incx);
    double parallel_dscal(std::vector<double>& x, double a,
                const std::vector<ILP_TYPE>& descx, ILP_TYPE incx);
    void parallel_daxpy(const std::vector<double>& x, std::vector<double>& y,
                        const std::vector<ILP_TYPE>& descx, ILP_TYPE incx,
                        const std::vector<ILP_TYPE>& descy, ILP_TYPE incy, double alpha = 1.0);
    void parallel_zaxpy(const std::vector<COMPLEX>& x, std::vector<COMPLEX>& y,
                        const std::vector<ILP_TYPE>& descx, ILP_TYPE incx,
                        const std::vector<ILP_TYPE>& descy, ILP_TYPE incy, COMPLEX alpha = COMPLEX(1, 0));
#endif
}

} // namespace QComputations

#endif // ENABLE_MPI

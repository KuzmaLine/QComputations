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
        using LP_TYPE = long long;
        constexpr MPI_Datatype MPI_BCAST_DATATYPE = MPI_LONG_LONG;
    #else
        using LP_TYPE = int;
        constexpr MPI_Datatype MPI_BCAST_DATATYPE = MPI_INT;
    #endif

    using COMPLEX = std::complex<double>;
}

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
    // ON BLACS GRID

    void init_grid(LP_TYPE& ctxt);

    template<typename T>
    Matrix<T> scatter_blacs_matrix(const Matrix<T>& A, LP_TYPE& N, LP_TYPE& M,
                             LP_TYPE& NB, LP_TYPE& MB, LP_TYPE& nrows,
                             LP_TYPE& ncols, LP_TYPE& ctxt, LP_TYPE root_id,
                             LP_TYPE NB_FORCE = LP_TYPE(0), LP_TYPE MB_FORCE = LP_TYPE(0));

    template<>
    Matrix<double> scatter_blacs_matrix<double>(const Matrix<double>& A, LP_TYPE& N, LP_TYPE& M,
                                      LP_TYPE& NB, LP_TYPE& MB, LP_TYPE& nrows,
                                      LP_TYPE& ncols, LP_TYPE& ctxt, LP_TYPE root_id,
                                      LP_TYPE NB_FORCE, LP_TYPE MB_FORCE);
    template<>
    Matrix<COMPLEX> scatter_blacs_matrix<COMPLEX>(const Matrix<COMPLEX>& A, LP_TYPE& N, LP_TYPE& M,
                                       LP_TYPE& NB, LP_TYPE& MB, LP_TYPE& nrows,
                                       LP_TYPE& ncols, LP_TYPE& ctxt, LP_TYPE root_id,
                                       LP_TYPE NB_FORCE, LP_TYPE MB_FORCE);
    
    template<typename T>
    void gather_blacs_matrix(Matrix<T>& A, LP_TYPE& N, LP_TYPE& M,
                             LP_TYPE& NB, LP_TYPE& MB, LP_TYPE& nrows,
                             LP_TYPE& ncols, LP_TYPE root_id,
                             LP_TYPE NB_FORCE = 0, LP_TYPE MB_FORCE = 0);

    template<>
    void gather_blacs_matrix<double>(Matrix<double>& A, LP_TYPE& N, LP_TYPE& M,
                                     LP_TYPE& NB, LP_TYPE& MB, LP_TYPE& nrows,
                                     LP_TYPE& ncols, LP_TYPE root_id,
                                     LP_TYPE NB_FORCE, LP_TYPE MB_FORCE);
    template<>
    void gather_blacs_matrix<COMPLEX>(Matrix<COMPLEX>& A, LP_TYPE& N, LP_TYPE& M,
                                      LP_TYPE& NB, LP_TYPE& MB, LP_TYPE& nrows,
                                      LP_TYPE& ncols, LP_TYPE root_id,
                                      LP_TYPE NB_FORCE, LP_TYPE MB_FORCE);

#endif

    MPI_Datatype Create_Block_Type_double (LP_TYPE N, LP_TYPE M, LP_TYPE NB, LP_TYPE MB);
    MPI_Datatype Create_Block_Type_complex (LP_TYPE N, LP_TYPE M, LP_TYPE NB, LP_TYPE MB);

    // DELETE
    void RING_Bcast(double *buf, int count, MPI_Datatype type, int root, MPI_Comm comm);

    // Stay to wait all MPI process until root process give commands. MPI_Init included
    void run_mpi_slaves(const std::map<int, std::vector<MPI_Data>>& data); 

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
   void parallel_dgemm(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C,
                       bool is_distributed = false, LP_TYPE* desca = NULL, LP_TYPE* descb = NULL, LP_TYPE* descc = NULL);
   void parallel_zgemm(const Matrix<COMPLEX>& A, const Matrix<COMPLEX>& B, Matrix<COMPLEX>& C);

   void print_distributed_matrix(const Matrix<double>& A, const std::string& matrix_name, MPI_Comm comm = MPI_COMM_WORLD);
#endif
}

#endif // ENABLE_MPI

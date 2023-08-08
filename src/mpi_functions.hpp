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
    using COMPLEX = std::complex<double>;
}

// COMMAND LIST
namespace COMMAND {
    constexpr int COMMANDS_COUNT = 11;

    constexpr int STOP = 0;
    constexpr int GENERATE_H = 1;
    constexpr int GENERATE_H_FUNC = 2;
    constexpr int SCHRODINGER = 3;
    constexpr int QME = 4;
    constexpr int CANNON_MULTIPLY = 5;
    constexpr int MATVEC = 6;
    constexpr int MATNUM = 7;
    constexpr int EXIT_FROM_FUNC = 8;
    constexpr int DIM_MULTIPLY = 9;
    constexpr int P_GEMM_MULTIPLY = 10;

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
    void make_command(int command);

    std::vector<COMPLEX> bcast_vector_complex(const std::vector<COMPLEX>& v = {});
    std::vector<double> bcast_vector_double(const std::vector<double>& v = {});
    State bcast_state(const State& state = State());
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

#ifdef ENABLE_CLUSTER 
   void parallel_dgemm(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C);
   void parallel_zgemm(const Matrix<COMPLEX>& A, const Matrix<COMPLEX>& B, Matrix<COMPLEX>& C);
#endif
}

#endif // ENABLE_MPI

/*

Пример работы с blocked_matrix.hpp

GE - general matrix
HE - hermitian matrix (Внимание, в памяти нижняя треугольная часть матрицы не проинициализирована)
SY - symmetric matrix (Аналогично HE)

*/

#include <iostream>
#include <chrono>
#include <complex>
#include "/home/kuzmaline/Quantum/diploma/src/QComputations_CPU_CLUSTER_NO_PLOTS.hpp"

using COMPLEX = std::complex<double>;

int main(int argc, char** argv) {
    using namespace QComputations;
    int n = 4;
    int m = 4;
    int k = 4;
    int rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == mpi::ROOT_ID) {
        Matrix<COMPLEX> A(C_STYLE, n, m);
        Matrix<COMPLEX> B(C_STYLE, m, k);
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < m; j++) {
                A[i][j] = i + j + i % 3 + j % 2 + 1;
            }
        }

        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < k; j++) {
                B[i][j] = int(i) - int(j);
            }
        }

        A.show();
        B.show();
        auto C = A * B;

        C.show();

        C = A + B;
        
        C.show();

        std::cout << " ----------------------------------------- \n";
    }

    MPI_Barrier(MPI_COMM_WORLD);

    std::function<COMPLEX(size_t i, size_t j)> func = {[](size_t i, size_t j){ return i + j + i % 3 + j % 2 + 1; }};
    std::function<COMPLEX(size_t i, size_t j)> func_2 = {[](size_t i, size_t j){ return int(i) - int(j); }};

    int ctxt;
    mpi::init_grid(ctxt);

    BLOCKED_Matrix<COMPLEX> M(ctxt, GE, n, m, func);
    BLOCKED_Matrix<COMPLEX> K(ctxt, GE, m, k, func_2);

    //std::cout << rank << " - " << M.local_n() << " " << M.local_m() << " " << M.NB() << " " << M.MB() << std::endl;
    //M.print_distributed(ctxt, "M");
    M.show(mpi::ROOT_ID);
    K.show(mpi::ROOT_ID);

    //auto P = M * K;

    //P.show(mpi::ROOT_ID);
    //P.print_distributed(ctxt, "P");


    M *= K;

    M.show(mpi::ROOT_ID);

    MPI_Finalize();
    return 0;
}
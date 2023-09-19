#include <iostream>
#include <chrono>
#include "/home/kuzmaline/Quantum/diploma/src/QComputations_MPI_CLUSTER.hpp"

using COMPLEX = std::complex<double>;

#ifdef DOUBLE_TYPE
using NumType = double;
#else
using NumType = COMPLEX;
#endif

enum NUM_TASK { MM = 100, MV = 101, HM = 102};

int main(int argc, char** argv) {
    using namespace QComputations;
    int n_a = 4, m_a = 4;
    int n_b = 4, m_b = 1;
    NUM_TASK TASK = MV;
    bool IS_COUT = true;

    int rank, world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Matrix<NumType> A, B, C, localA, localB, localC;
    std::vector<NumType> b, y, localb, localy;
    if (rank == 0) {
#ifdef DOUBLE_TYPE
        if (TASK == MM) {
            A = matrix::create_rand_matrix<NumType>(n_a, m_a, 0, 5);
            B = matrix::create_rand_matrix<NumType>(n_b, m_b, 0, 5);
            A.show();
            //B.show();
        } else if (TASK == MV) {
            A = matrix::create_rand_matrix<NumType>(n_a, m_a, 0, 5);
            b = matrix::create_rand_matrix<NumType>(m_a, 1, 0, 5).col(0);
        }
#else
        if (TASK == MM) {
            A = matrix::create_rand_complex_matrix<NumType>(n_a, m_a, COMPLEX(0, 0), COMPLEX(5, 5));
            B = matrix::create_rand_complex_matrix<NumType>(n_b, m_b, COMPLEX(0, 0), COMPLEX(5, 5));
        } else if (TASK == MV) {
            A = matrix::create_rand_complex_matrix<NumType>(n_a, m_a, COMPLEX(0, 0), COMPLEX(5, 5));
            b = matrix::create_rand_complex_matrix<NumType>(n_b, 1, COMPLEX(0, 0), COMPLEX(5, 5)).col(0);
        }
#endif
        if (TASK == MM) {
            A.to_fortran_style();
            B.to_fortran_style();
            auto begin = std::chrono::steady_clock::now();
            C = A * B;
            auto end = std::chrono::steady_clock::now();
            std::cout << "COMMON: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
            if (IS_COUT) C.show();
        } else if (TASK == MV) {
            A.to_fortran_style();
            auto begin = std::chrono::steady_clock::now();
            y = A * b;
            //A.show();
            auto end = std::chrono::steady_clock::now();
            if (IS_COUT) std::cout << y << std::endl;
            std::cout << "COMMON: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
        }
    }

    int iZERO = 0;
    int ctxt, myrow, mycol, proc_rows, proc_cols;
    mpi::init_grid(ctxt);
    mpi::blacs_gridinfo(ctxt, proc_rows, proc_cols, myrow, mycol);

    int NB_A, MB_A, NB_B, MB_B, N_Y, M_Y;
    int nrows_A, ncols_A, nrows_B, ncols_B, nrows_y, ncols_y, nrows_C, ncols_C;

    if (TASK == MM) {
        localA = mpi::scatter_blacs_matrix<NumType>(A, n_a, m_a, NB_A, MB_A, nrows_A, ncols_A, ctxt, mpi::ROOT_ID, MPI_COMM_WORLD);
        localB = mpi::scatter_blacs_matrix<NumType>(B, n_b, m_b, NB_B, MB_B, nrows_B, ncols_B, ctxt, mpi::ROOT_ID, MPI_COMM_WORLD);


        //std::cout << rank << " - " << NB_A << " " << MB_A << std::endl;
        //mpi::print_distributed_matrix(localA, "localA", MPI_COMM_WORLD);
        //mpi::print_distributed_matrix(localB, "localB", MPI_COMM_WORLD);

        nrows_C = mpi::numroc(n_a, NB_A, myrow, iZERO, proc_rows);
        ncols_C = mpi::numroc(m_b, MB_B, mycol, iZERO, proc_cols);

        localC = Matrix<NumType>(FORTRAN_STYLE, nrows_C, ncols_C);
        //std::cout << localC.n() << " " << localC.m() << std::endl;
    } else if (TASK == MV) {
        localA = mpi::scatter_blacs_matrix<NumType>(A, n_a, m_a, NB_A, MB_A, nrows_A, ncols_A, ctxt, mpi::ROOT_ID, MPI_COMM_WORLD);
        //vector_to_matrix<NumType>(FORTRAN_STYLE, b).show();
        localb = mpi::scatter_blacs_vector<NumType>(b, n_b, NB_B, nrows_B, ctxt, mpi::ROOT_ID, MPI_COMM_WORLD);
        MB_B = 1;
        //mpi::print_distributed_matrix(localA, "localA", MPI_COMM_WORLD);
        //mpi::print_distributed_matrix(tmp, "tmp", MPI_COMM_WORLD);
        nrows_y = mpi::numroc(n_a, NB_A, myrow, iZERO, proc_rows);
        ncols_y = mpi::numroc(m_b, MB_B, mycol, iZERO, proc_cols);
        //std::cout << n_a << " " << NB_A << " " << nrows_y << " " << NB_B << std::endl;
        //auto ncols_C = numroc_(&m_b, &MB_B, &mycol, &iZERO, &proc_cols);

        localy = std::vector<NumType>(nrows_y);
    }

    int LLD_A = nrows_A;
    int LLD_B = nrows_B;
    int LLD_C;
    if (TASK == MM) {
        LLD_C = nrows_C;
    } else if (TASK == MV) {
        LLD_C = nrows_y;
    }

    int rsrc = 0, csrc = 0, info;
    //int* desca = new int[9];
    //int* descb = new int[9];
    //int* descc = new int[9];

    //std::cout << rank << " " << n_a << " " << m_a << " " << n_b << " " << m_b << " " << NB_A << " " << MB_A << " " << NB_B << " " << MB_B << std::endl;
    int* desca = mpi::descinit(n_a, m_a, NB_A, MB_A, rsrc, csrc, ctxt, LLD_A, info);
    if (info != 0) std::cout << "ERROR OF descinit__A: " << rank << " " << info << std::endl;
    int* descb = mpi::descinit(n_b, m_b, NB_B, MB_B, rsrc, csrc, ctxt, LLD_B, info);
    if (info != 0) std::cout << "ERROR OF descinit__B: " << rank << " " << info << std::endl;
    int* descc = mpi::descinit(n_a, m_b, NB_A, MB_B, rsrc, csrc, ctxt, LLD_C, info);
    if (info != 0) std::cout << "ERROR OF descinit__C: " << rank << " " << info << std::endl;

    auto diag = mpi::get_diagonal_elements<NumType>(localA, desca);
    if (rank == mpi::ROOT_ID) std::cout << diag << std::endl;

    auto begin = std::chrono::steady_clock::now();
#ifdef DOUBLE_TYPE
    if (TASK == MM) {
        mpi::parallel_dgemm(localA, localB, localC, true, desca, descb, descc);
    } else if (TASK == MV) {
        mpi::parallel_dgemv(localA, localb, localy, true, desca, descb, descc);
        //std::cout << localy << std::endl;
    }
#else
    if (TASK == MM) {
        mpi::parallel_zgemm(localA, localB, localC, true, desca, descb, descc);
    } else if (TASK == MV) {
        mpi::parallel_zgemv(localA, localb, localy, true, desca, descb, descc);
    }
#endif
    //pdgemv_(&N, &n_a, &m_a, &alpha, localA.data(), &iONE, &iONE, desca,
    //                                localX.data(), &iONE, &iONE, descb, &iONE,
    //                                &betta, localC.data(), &iONE, &iONE, descc, &iONE);
    auto end = std::chrono::steady_clock::now();

    if (TASK == MM) {
        if (IS_COUT) mpi::print_distributed_matrix<NumType>(localC, "localC", MPI_COMM_WORLD);
        if (rank == mpi::ROOT_ID) std::cout << "PGEMM: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
    } else if (TASK == MV) {
        //mpi::print_distributed_matrix<NumType>(vector_to_matrix(FORTRAN_STYLE, localy), "localy", MPI_COMM_WORLD); 
        if (rank == mpi::ROOT_ID) std::cout << "PGEMV: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
    }

    Matrix<NumType> C_res;
    std::vector<NumType> y_res;
    if (TASK == MM) {
        if (rank == mpi::ROOT_ID) C_res = Matrix<NumType>(FORTRAN_STYLE, n_a, m_b);
        mpi::gather_blacs_matrix<NumType>(localC, C_res, n_a, m_b, NB_A, MB_B, nrows_C, ncols_C, ctxt, mpi::ROOT_ID);
    } else if (TASK == MV) {
        if (rank == mpi::ROOT_ID) y_res = std::vector<NumType>(n_a);
        mpi::gather_blacs_vector<NumType>(localy, y_res, n_a, NB_A, nrows_y, ctxt, mpi::ROOT_ID);
    }
    delete [] desca;
    delete [] descb;
    delete [] descc;
    mpi::blacs_gridexit(ctxt);

    if (IS_COUT) {
        if (TASK == MM) {
            if (rank == mpi::ROOT_ID) C.show();
        } else if (TASK == MV) {
            if (rank == mpi::ROOT_ID) std::cout << y_res << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}


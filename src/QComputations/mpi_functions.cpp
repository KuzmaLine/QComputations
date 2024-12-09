#ifdef ENABLE_MPI

#define MKL_Complex16 std::complex<double>

#include "mpi_functions.hpp"

#include "dynamic.hpp"
#include "functions.hpp"
#include "hamiltonian.hpp"

#ifdef ENABLE_CLUSTER

#include <mkl_blacs.h>
#include <mkl_pblas.h>
#include <mkl_scalapack.h>

extern "C" {
void pdelget_(char*, char*, double*, const double*, int*, int*, const int*);
void pzelget_(char*, char*, COMPLEX*, const COMPLEX*, int*, int*, const int*);
void pdelset_(double*, int*, int*, const int*, double*);
void pzelset_(COMPLEX*, int*, int*, const int*, COMPLEX*);
ILP_TYPE indxl2g_(ILP_TYPE*, ILP_TYPE*, ILP_TYPE*, ILP_TYPE*, ILP_TYPE*);
ILP_TYPE indxg2p_(ILP_TYPE*, ILP_TYPE*, ILP_TYPE*, ILP_TYPE*, ILP_TYPE*);
ILP_TYPE indxg2l_(ILP_TYPE*, ILP_TYPE*, ILP_TYPE*, ILP_TYPE*, ILP_TYPE*);
}

#endif

namespace QComputations {

#ifdef ENABLE_CLUSTER
bool is_main_proc() {
    ILP_TYPE rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    return rank == 0;
}

bool get_proc_rank() {
    ILP_TYPE rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    return rank;
}
#endif

std::vector<COMPLEX> mpi::bcast_vector_complex(const std::vector<COMPLEX>& v) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size_t n = v.size();
    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, ROOT_ID, MPI_COMM_WORLD);

    std::vector<COMPLEX> res(n);
    if (rank == ROOT_ID) res = v;

    MPI_Bcast(res.data(), n, MPI_DOUBLE_COMPLEX, ROOT_ID, MPI_COMM_WORLD);

    return res;
}

std::vector<double> mpi::bcast_vector_double(const std::vector<double>& v) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size_t n = v.size();
    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, ROOT_ID, MPI_COMM_WORLD);

    std::vector<double> res(n);
    if (rank == ROOT_ID) res = v;

    MPI_Bcast(res.data(), n, MPI_DOUBLE, ROOT_ID, MPI_COMM_WORLD);

    return res;
}

// ##################################### BLACS, PBLAS ###########################################

#ifdef ENABLE_CLUSTER

// UNUSED
MPI_Datatype Create_Block_Type_double(ILP_TYPE N, ILP_TYPE M, ILP_TYPE NB, ILP_TYPE MB) {
    MPI_Datatype tmp_type, block_type;
    int starts[2] = {0, 0};
    int global_size[2] = {N, M};
    int local_size[2] = {NB, MB};

    MPI_Type_create_subarray(2, global_size, local_size, starts, MPI_ORDER_C, MPI_DOUBLE, &tmp_type);
    MPI_Type_commit(&block_type);

    return block_type;
}

namespace {
void my_dgesd2d(ILP_TYPE N,
                ILP_TYPE M,
                ILP_TYPE row_index,
                ILP_TYPE col_index,
                const Matrix<double>& A,
                ILP_TYPE send_id) {
    auto sub = A.submatrix(N, M, row_index, col_index);
    MPI_Send(sub.data(), M * N, MPI_DOUBLE, send_id, 0, MPI_COMM_WORLD);
}

void my_dgerv2d(ILP_TYPE N, ILP_TYPE M, ILP_TYPE offset, Matrix<double>& A, ILP_TYPE LDA, ILP_TYPE source_id) {
    Matrix<double> tmp(A.get_matrix_style(), N, M);
    ILP_TYPE row_offset, col_offset;
    if (A.is_c_style()) {
        row_offset = offset / LDA;
        col_offset = offset % LDA;
    } else {
        row_offset = offset % LDA;
        col_offset = offset / LDA;
    }
    MPI_Recv(tmp.data(), N * M, MPI_DOUBLE, source_id, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < M; j++) {
            A.data()[A.index(row_offset + i, col_offset + j)] = tmp.data()[tmp.index(i, j)];
        }
    }
}

void my_zgesd2d(ILP_TYPE N,
                ILP_TYPE M,
                ILP_TYPE row_index,
                ILP_TYPE col_index,
                const Matrix<COMPLEX>& A,
                ILP_TYPE send_id) {
    auto sub = A.submatrix(N, M, row_index, col_index);
    MPI_Send(sub.data(), M * N, MPI_DOUBLE_COMPLEX, send_id, 0, MPI_COMM_WORLD);
}

void my_zgerv2d(ILP_TYPE N, ILP_TYPE M, ILP_TYPE offset, Matrix<COMPLEX>& A, ILP_TYPE LDA, ILP_TYPE source_id) {
    Matrix<COMPLEX> tmp(A.get_matrix_style(), N, M);
    ILP_TYPE row_offset, col_offset;
    if (A.is_c_style()) {
        row_offset = offset / LDA;
        col_offset = offset % LDA;
    } else {
        row_offset = offset % LDA;
        col_offset = offset / LDA;
    }
    MPI_Recv(tmp.data(), N * M, MPI_DOUBLE_COMPLEX, source_id, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < M; j++) {
            A.data()[A.index(row_offset + i, col_offset + j)] = tmp.data()[tmp.index(i, j)];
        }
    }
}

// NEED TO REPLACE
void ScatterBLACSMatrix_double(const Matrix<double>& A,
                               ILP_TYPE NA,
                               ILP_TYPE MA,
                               Matrix<double>& localA,
                               ILP_TYPE NB_A,
                               ILP_TYPE MB_A,
                               ILP_TYPE nrows_A,
                               ILP_TYPE ncols_A,
                               ILP_TYPE myrow,
                               ILP_TYPE mycol,
                               ILP_TYPE proc_rows,
                               ILP_TYPE proc_cols,
                               ILP_TYPE rank,
                               ILP_TYPE* p_ctxt) {
    auto A_tmp = A;
    ILP_TYPE iZERO = 0;
    ILP_TYPE sendr = 0, sendc = 0, recvr = 0, recvc = 0;
    for (ILP_TYPE r = 0; r < NA; r += NB_A, sendr = (sendr + 1) % proc_rows) {
        sendc = 0;
        // Number of rows to be sent
        // Is this the last row block?
        ILP_TYPE nr = NB_A;
        if (NA - r < NB_A) nr = NA - r;

        for (ILP_TYPE c = 0; c < MA; c += MB_A, sendc = (sendc + 1) % proc_cols) {
            // Number of cols to be sent
            // Is this the last col block?
            ILP_TYPE nc = MB_A;
            if (MA - c < MB_A) nc = MA - c;

            if (rank == mpi::ROOT_ID) {
                // Send a nr-by-nc submatrix to process (sendr, sendc)
                ILP_TYPE send_id = blacs_pnum(p_ctxt, &sendr, &sendc);
                my_dgesd2d(nr, nc, r, c, A, send_id);
            }

            if (myrow == sendr && mycol == sendc) {
                //std::cout << "Recv " << myrow << " " << mycol << " " << recvr << " " << recvc << std::endl;
                // Receive the same data
                // The leading dimension of the local matrix is nrows!
                my_dgerv2d(nr, nc, ncols_A * recvr + recvc, localA, ncols_A, mpi::ROOT_ID);
                recvc = (recvc + nc) % ncols_A;
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }

        if (myrow == sendr) recvr = (recvr + nr) % nrows_A;
    }
}

// NEED TO REPLACE
void ScatterBLACSMatrix_COMPLEX(const Matrix<COMPLEX>& A,
                                ILP_TYPE NA,
                                ILP_TYPE MA,
                                Matrix<COMPLEX>& localA,
                                ILP_TYPE NB_A,
                                ILP_TYPE MB_A,
                                ILP_TYPE nrows_A,
                                ILP_TYPE ncols_A,
                                ILP_TYPE myrow,
                                ILP_TYPE mycol,
                                ILP_TYPE proc_rows,
                                ILP_TYPE proc_cols,
                                ILP_TYPE rank,
                                ILP_TYPE* p_ctxt) {
    ILP_TYPE sendr = 0, sendc = 0, recvr = 0, recvc = 0;
    for (ILP_TYPE r = 0; r < NA; r += NB_A, sendr = (sendr + 1) % proc_rows) {
        sendc = 0;
        // Number of rows to be sent
        // Is this the last row block?
        ILP_TYPE nr = NB_A;
        if (NA - r < NB_A) nr = NA - r;

        for (ILP_TYPE c = 0; c < MA; c += MB_A, sendc = (sendc + 1) % proc_cols) {
            // Number of cols to be sent
            // Is this the last col block?
            ILP_TYPE nc = MB_A;
            if (MA - c < MB_A) nc = MA - c;

            if (rank == mpi::ROOT_ID) {
                // Send a nr-by-nc submatrix to process (sendr, sendc)
                ILP_TYPE send_id = blacs_pnum(p_ctxt, &sendr, &sendc);
                my_zgesd2d(nr, nc, r, c, A, send_id);
            }

            if (myrow == sendr && mycol == sendc) {
                //std::cout << "Recv " << myrow << " " << mycol << " " << recvr << " " << recvc << std::endl;
                // Receive the same data
                // The leading dimension of the local matrix is nrows!
                my_zgerv2d(nr, nc, ncols_A * recvr + recvc, localA, ncols_A, mpi::ROOT_ID);
                recvc = (recvc + nc) % ncols_A;
            }
        }

        if (myrow == sendr) recvr = (recvr + nr) % nrows_A;
    }
}
}  // namespace

ILP_TYPE mpi::indxl2g(ILP_TYPE n, ILP_TYPE NB, ILP_TYPE myindx, ILP_TYPE RSRC, ILP_TYPE dim_size) {
    ILP_TYPE n_new = n + 1;
    return indxl2g_(&n_new, &NB, &myindx, &RSRC, &dim_size) - 1;
}

ILP_TYPE mpi::indxg2p(ILP_TYPE n, ILP_TYPE NB, ILP_TYPE myindx, ILP_TYPE RSRC, ILP_TYPE dim_size) {
    ILP_TYPE n_new = n + 1;
    return indxg2p_(&n_new, &NB, &myindx, &RSRC, &dim_size);
}

ILP_TYPE mpi::indxg2l(ILP_TYPE n, ILP_TYPE NB, ILP_TYPE myindx, ILP_TYPE RSRC, ILP_TYPE dim_size) {
    ILP_TYPE n_new = n + 1;
    return indxg2l_(&n_new, &NB, &myindx, &RSRC, &dim_size) - 1;
}

std::vector<ILP_TYPE> mpi::descinit(ILP_TYPE n,
                                    ILP_TYPE m,
                                    ILP_TYPE NB,
                                    ILP_TYPE MB,
                                    ILP_TYPE rsrc,
                                    ILP_TYPE csrc,
                                    ILP_TYPE ctxt,
                                    ILP_TYPE LLD,
                                    ILP_TYPE info) {
    std::vector<ILP_TYPE> desc(9);
    descinit_(desc.data(), &n, &m, &NB, &MB, &rsrc, &csrc, &ctxt, &LLD, &info);

    return desc;
}

void mpi::blacs_gridinfo(const ILP_TYPE& ctxt,
                         ILP_TYPE& proc_rows,
                         ILP_TYPE& proc_cols,
                         ILP_TYPE& myrow,
                         ILP_TYPE& mycol) {
    ::blacs_gridinfo(&ctxt, &proc_rows, &proc_cols, &myrow, &mycol);
}

ILP_TYPE mpi::numroc(ILP_TYPE N, ILP_TYPE NB, ILP_TYPE myindex, ILP_TYPE ZERO, ILP_TYPE size) {
    return ::numroc_(&N, &NB, &myindex, &ZERO, &size);
}

void mpi::blacs_gridexit(ILP_TYPE& ctxt) { ::blacs_gridexit(&ctxt); }

double mpi::pdelget(const Matrix<double>& A, ILP_TYPE i, ILP_TYPE j, const std::vector<ILP_TYPE>& desc) {
    char chA = 'A';
    char TopI = 'I';
    ILP_TYPE new_i = i + 1;
    ILP_TYPE new_j = j + 1;
    double res;
    ::pdelget_(&chA, &TopI, &res, A.data(), &new_i, &new_j, desc.data());
    return res;
}

COMPLEX mpi::pzelget(const Matrix<COMPLEX>& A, ILP_TYPE i, ILP_TYPE j, const std::vector<ILP_TYPE>& desc) {
    char chA = 'A';
    char TopI = 'I';
    ILP_TYPE new_i = i + 1;
    ILP_TYPE new_j = j + 1;
    COMPLEX res;
    ::pzelget_(&chA, &TopI, &res, A.data(), &new_i, &new_j, desc.data());
    return res;
}

void mpi::pdelset(Matrix<double>& A, ILP_TYPE i, ILP_TYPE j, double num, const std::vector<ILP_TYPE>& desc) {
    ILP_TYPE new_i = i + 1;
    ILP_TYPE new_j = j + 1;
    ::pdelset_(A.data(), &new_i, &new_j, desc.data(), &num);
}

void mpi::pzelset(Matrix<COMPLEX>& A, ILP_TYPE i, ILP_TYPE j, COMPLEX num, const std::vector<ILP_TYPE>& desc) {
    ILP_TYPE new_i = i + 1;
    ILP_TYPE new_j = j + 1;
    ::pzelset_(A.data(), &new_i, &new_j, desc.data(), &num);
}

void mpi::init_vector_grid(ILP_TYPE& ctxt, ILP_TYPE proc_rows, ILP_TYPE proc_cols) {
    ILP_TYPE iZERO = 0;
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ILP_TYPE myid, numproc, myrow, mycol;
    char order = 'R';
    if (proc_rows == 0 and proc_cols == 0) {
        proc_rows = world_size;
        proc_cols = 1;
    }

    blacs_pinfo(&myid, &numproc);
    ILP_TYPE iMINUS = -1;
    blacs_get(&iMINUS, &iZERO, &ctxt);
    blacs_gridinit(&ctxt, &order, &proc_rows, &proc_cols);
}

void mpi::init_grid(ILP_TYPE& ctxt, ILP_TYPE proc_rows, ILP_TYPE proc_cols) {
    ILP_TYPE iZERO = 0;
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ILP_TYPE myid, numproc, myrow, mycol;
    char order = 'R';
    if (proc_rows == 0 or proc_cols == 0) {
        proc_rows = std::sqrt(world_size);
        for (ILP_TYPE i = proc_rows; i <= world_size; i++) {
            proc_rows = i;

            if (world_size % proc_rows == 0) {
                break;
            }
        }
        proc_cols = world_size / proc_rows;
    } else {
        assert(proc_rows * proc_cols == world_size);
    }

    blacs_pinfo(&myid, &numproc);
    ILP_TYPE iMINUS = -1;
    blacs_get(&iMINUS, &iZERO, &ctxt);
    blacs_gridinit(&ctxt, &order, &proc_rows, &proc_cols);
}

template <>
Matrix<double> mpi::scatter_blacs_matrix<double>(const Matrix<double>& A,
                                                 ILP_TYPE& N,
                                                 ILP_TYPE& M,
                                                 ILP_TYPE& NB,
                                                 ILP_TYPE& MB,
                                                 ILP_TYPE& nrows,
                                                 ILP_TYPE& ncols,
                                                 ILP_TYPE& ctxt,
                                                 ILP_TYPE root_id,
                                                 MPI_Comm comm) {
    ILP_TYPE iZERO = 0;
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ILP_TYPE myrow, mycol;
    char order = 'R';
    ILP_TYPE proc_rows, proc_cols;
    blacs_gridinfo(ctxt, proc_rows, proc_cols, myrow, mycol);

    ILP_TYPE root_row, root_col;
    ILP_TYPE bcast_data[5];
    if (rank == root_id) {
        bcast_data[0] = A.n();
        bcast_data[1] = A.m();
        bcast_data[2] = A.get_matrix_style();
        bcast_data[3] = myrow;
        bcast_data[4] = mycol;
    }

    MPI_Bcast(&bcast_data, 5, MPI_BCAST_DATATYPE, mpi::ROOT_ID, comm);

    N = bcast_data[0];
    M = bcast_data[1];
    MATRIX_STYLE matrix_style = MATRIX_STYLE(bcast_data[2]);
    root_row = bcast_data[3];
    root_col = bcast_data[4];
    NB = N / proc_rows;
    MB = M / proc_cols;

    if (NB == 0) NB = 1;
    if (MB == 0) MB = 1;

    nrows = numroc_(&N, &NB, &myrow, &iZERO, &proc_rows);
    ncols = numroc_(&M, &MB, &mycol, &iZERO, &proc_cols);

    Matrix<double> localA(matrix_style, nrows, ncols);

    bool is_c_style = localA.is_c_style();
    ILP_TYPE A_LD = A.LD();
    ILP_TYPE localA_LD = localA.LD();
    ILP_TYPE sendr = 0, sendc = 0, recvr = 0, recvc = 0;
    for (ILP_TYPE r = 0; r < N; r += NB, sendr = (sendr + 1) % proc_rows) {
        sendc = 0;
        ILP_TYPE nr = NB;
        if (N - r < NB) nr = N - r;

        for (ILP_TYPE c = 0; c < M; c += MB, sendc = (sendc + 1) % proc_cols) {
            ILP_TYPE nc = MB;
            if (M - c < MB) nc = M - c;

            if (rank == root_id) {
                ILP_TYPE send_id = blacs_pnum(&ctxt, &sendr, &sendc);
                my_dgesd2d(nr, nc, r, c, A, send_id);
            }

            if (myrow == sendr && mycol == sendc) {
                if (is_c_style) {
                    my_dgerv2d(nr, nc, ncols * recvr + recvc, localA, localA_LD, root_id);
                } else {
                    my_dgerv2d(nr, nc, nrows * recvc + recvr, localA, localA_LD, root_id);
                }
                recvc = (recvc + nc) % ncols;
            }
        }

        if (myrow == sendr) recvr = (recvr + nr) % nrows;
    }

    return localA;
}

template <>
Matrix<COMPLEX> mpi::scatter_blacs_matrix<COMPLEX>(const Matrix<COMPLEX>& A,
                                                   ILP_TYPE& N,
                                                   ILP_TYPE& M,
                                                   ILP_TYPE& NB,
                                                   ILP_TYPE& MB,
                                                   ILP_TYPE& nrows,
                                                   ILP_TYPE& ncols,
                                                   ILP_TYPE& ctxt,
                                                   ILP_TYPE root_id,
                                                   MPI_Comm comm) {
    ILP_TYPE iZERO = 0;
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ILP_TYPE myrow, mycol;
    ILP_TYPE proc_rows, proc_cols;
    blacs_gridinfo(ctxt, proc_rows, proc_cols, myrow, mycol);

    ILP_TYPE root_row, root_col;
    ILP_TYPE bcast_data[5];
    if (rank == root_id) {
        bcast_data[0] = A.n();
        bcast_data[1] = A.m();
        bcast_data[2] = A.get_matrix_style();
        bcast_data[3] = myrow;
        bcast_data[4] = mycol;
    }

    MPI_Bcast(&bcast_data, 5, MPI_BCAST_DATATYPE, mpi::ROOT_ID, comm);

    N = bcast_data[0];
    M = bcast_data[1];
    MATRIX_STYLE matrix_style = MATRIX_STYLE(bcast_data[2]);
    root_row = bcast_data[3];
    root_col = bcast_data[4];
    NB = N / proc_rows;
    MB = M / proc_cols;

    if (NB == 0) NB = 1;
    if (MB == 0) MB = 1;

    nrows = numroc_(&N, &NB, &myrow, &iZERO, &proc_rows);
    ncols = numroc_(&M, &MB, &mycol, &iZERO, &proc_cols);

    Matrix<COMPLEX> localA(matrix_style, nrows, ncols);
    bool is_c_style = localA.is_c_style();

    ILP_TYPE A_LD = A.LD();
    ILP_TYPE localA_LD = localA.LD();

    ILP_TYPE sendr = 0, sendc = 0, recvr = 0, recvc = 0;
    for (ILP_TYPE r = 0; r < N; r += NB, sendr = (sendr + 1) % proc_rows) {
        sendc = 0;
        ILP_TYPE nr = NB;
        if (N - r < NB) nr = N - r;

        for (ILP_TYPE c = 0; c < M; c += MB, sendc = (sendc + 1) % proc_cols) {
            ILP_TYPE nc = MB;
            if (M - c < MB) nc = M - c;

            if (rank == root_id) {
                ILP_TYPE send_id = blacs_pnum(&ctxt, &sendr, &sendc);
                my_zgesd2d(nr, nc, r, c, A, send_id);
            }

            if (myrow == sendr && mycol == sendc) {
                if (is_c_style) {
                    my_zgerv2d(nr, nc, ncols * recvr + recvc, localA, localA_LD, root_id);
                } else {
                    my_zgerv2d(nr, nc, nrows * recvc + recvr, localA, localA_LD, root_id);
                }
                recvc = (recvc + nc) % ncols;
            }
        }

        if (myrow == sendr) recvr = (recvr + nr) % nrows;
    }

    return localA;
}

template <>
void mpi::gather_blacs_matrix<double>(const Matrix<double>& localC,
                                      Matrix<double>& C,
                                      ILP_TYPE N,
                                      ILP_TYPE M,
                                      ILP_TYPE NB,
                                      ILP_TYPE MB,
                                      ILP_TYPE nrows,
                                      ILP_TYPE ncols,
                                      ILP_TYPE ctxt,
                                      ILP_TYPE root_id) {
    ILP_TYPE iZERO = 0;
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ILP_TYPE myrow, mycol;
    ILP_TYPE proc_rows, proc_cols;
    blacs_gridinfo(ctxt, proc_rows, proc_cols, myrow, mycol);

    ILP_TYPE root_row, root_col;
    ILP_TYPE bcast_data[2];
    if (rank == root_id) {
        bcast_data[0] = myrow;
        bcast_data[1] = mycol;
    }

    MPI_Bcast(&bcast_data, 2, MPI_BCAST_DATATYPE, mpi::ROOT_ID, MPI_COMM_WORLD);

    MATRIX_STYLE matrix_style = localC.get_matrix_style();
    root_row = bcast_data[0];
    root_col = bcast_data[1];

    bool is_c_style = localC.is_c_style();

    ILP_TYPE C_LD;
    if (rank == root_id) {
        C_LD = C.LD();
    }

    ILP_TYPE localC_LD = localC.LD();

    ILP_TYPE sendr = 0, sendc = 0, recvr = 0, recvc = 0;
    for (ILP_TYPE r = 0; r < N; r += NB, sendr = (sendr + 1) % proc_rows) {
        sendc = 0;
        ILP_TYPE nr = NB;
        if (N - r < NB) nr = N - r;

        for (ILP_TYPE c = 0; c < M; c += MB, sendc = (sendc + 1) % proc_cols) {
            ILP_TYPE nc = MB;
            if (M - c < MB) nc = M - c;

            if (myrow == sendr && mycol == sendc) {
                my_dgesd2d(nr, nc, recvr, recvc, localC, root_id);
                recvc = (recvc + nc) % ncols;
            }

            if (rank == root_id) {
                ILP_TYPE source_id = blacs_pnum(&ctxt, &sendr, &sendc);
                if (is_c_style) {
                    my_dgerv2d(nr, nc, M * r + c, C, C_LD, source_id);
                } else {
                    my_dgerv2d(nr, nc, N * c + r, C, C_LD, source_id);
                }
            }
        }

        if (myrow == sendr) recvr = (recvr + nr) % nrows;
    }
}

template <>
void mpi::gather_blacs_matrix<COMPLEX>(const Matrix<COMPLEX>& localC,
                                       Matrix<COMPLEX>& C,
                                       ILP_TYPE N,
                                       ILP_TYPE M,
                                       ILP_TYPE NB,
                                       ILP_TYPE MB,
                                       ILP_TYPE nrows,
                                       ILP_TYPE ncols,
                                       ILP_TYPE ctxt,
                                       ILP_TYPE root_id) {
    ILP_TYPE iZERO = 0;
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ILP_TYPE myrow, mycol;
    ILP_TYPE proc_rows, proc_cols;
    blacs_gridinfo(ctxt, proc_rows, proc_cols, myrow, mycol);

    ILP_TYPE root_row, root_col;
    ILP_TYPE bcast_data[2];
    if (rank == root_id) {
        bcast_data[0] = myrow;
        bcast_data[1] = mycol;
    }

    MPI_Bcast(&bcast_data, 2, MPI_BCAST_DATATYPE, mpi::ROOT_ID, MPI_COMM_WORLD);

    MATRIX_STYLE matrix_style = localC.get_matrix_style();
    root_row = bcast_data[0];
    root_col = bcast_data[1];

    ILP_TYPE C_LD;
    if (rank == root_id) {
        C_LD = C.LD();
    }

    bool is_c_style = localC.is_c_style();
    ILP_TYPE localC_LD = localC.LD();

    ILP_TYPE sendr = 0, sendc = 0, recvr = 0, recvc = 0;
    for (ILP_TYPE r = 0; r < N; r += NB, sendr = (sendr + 1) % proc_rows) {
        sendc = 0;
        ILP_TYPE nr = NB;
        if (N - r < NB) nr = N - r;

        for (ILP_TYPE c = 0; c < M; c += MB, sendc = (sendc + 1) % proc_cols) {
            ILP_TYPE nc = MB;
            if (M - c < MB) nc = M - c;

            if (myrow == sendr && mycol == sendc) {
                my_zgesd2d(nr, nc, recvr, recvc, localC, root_id);
                recvc = (recvc + nc) % ncols;
            }

            if (rank == root_id) {
                ILP_TYPE source_id = blacs_pnum(&ctxt, &sendr, &sendc);
                if (is_c_style) {
                    my_zgerv2d(nr, nc, M * r + c, C, C_LD, source_id);
                } else {
                    my_zgerv2d(nr, nc, N * c + r, C, C_LD, source_id);
                }
            }
        }

        if (myrow == sendr) recvr = (recvr + nr) % nrows;
    }
}

template <>
std::vector<double> mpi::scatter_blacs_vector<double>(const std::vector<double>& v,
                                                      ILP_TYPE& N,
                                                      ILP_TYPE& NB,
                                                      ILP_TYPE& nrows,
                                                      ILP_TYPE& ctxt,
                                                      ILP_TYPE root_id,
                                                      MPI_Comm comm) {
    ILP_TYPE iZERO = 0;
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ILP_TYPE myrow, mycol;
    char order = 'R';
    ILP_TYPE proc_rows, proc_cols;
    blacs_gridinfo(ctxt, proc_rows, proc_cols, myrow, mycol);

    ILP_TYPE root_row, root_col;
    ILP_TYPE bcast_data[3];
    if (rank == root_id) {
        bcast_data[0] = v.size();
        bcast_data[1] = myrow;
        bcast_data[2] = mycol;
    }

    MPI_Bcast(&bcast_data, 3, MPI_BCAST_DATATYPE, mpi::ROOT_ID, comm);

    N = bcast_data[0];
    root_row = bcast_data[1];
    root_col = bcast_data[2];
    NB = N / proc_rows;

    if (NB == 0) NB = N;

    nrows = numroc_(&N, &NB, &myrow, &iZERO, &proc_rows);

    std::vector<double> local_v(nrows);

    ILP_TYPE sendr = 0, sendc = 0, recvr = 0, recvc = 0;
    for (ILP_TYPE r = 0; r < N; r += NB, sendr = (sendr + 1) % proc_rows) {
        sendc = 0;
        ILP_TYPE nr = NB;
        if (N - r < NB) nr = N - r;

        if (rank == root_id) {
            for (ILP_TYPE sendc = 0; sendc < proc_cols; sendc++) {
                ILP_TYPE send_id = blacs_pnum(&ctxt, &sendr, &sendc);
                MPI_Send(v.data() + r, nr, MPI_DOUBLE, send_id, 0, MPI_COMM_WORLD);
            }
        }

        if (myrow == sendr && mycol == sendc) {
            MPI_Recv(local_v.data() + recvr, nr, MPI_DOUBLE, root_id, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (myrow == sendr) recvr = (recvr + nr) % nrows;
    }

    return local_v;
}

template <>
std::vector<COMPLEX> mpi::scatter_blacs_vector<COMPLEX>(const std::vector<COMPLEX>& v,
                                                        ILP_TYPE& N,
                                                        ILP_TYPE& NB,
                                                        ILP_TYPE& nrows,
                                                        ILP_TYPE& ctxt,
                                                        ILP_TYPE root_id,
                                                        MPI_Comm comm) {
    ILP_TYPE iZERO = 0;
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ILP_TYPE myrow, mycol;
    char order = 'R';
    ILP_TYPE proc_rows, proc_cols;
    blacs_gridinfo(ctxt, proc_rows, proc_cols, myrow, mycol);

    ILP_TYPE root_row, root_col;
    ILP_TYPE bcast_data[3];
    if (rank == root_id) {
        bcast_data[0] = v.size();
        bcast_data[1] = myrow;
        bcast_data[2] = mycol;
    }

    MPI_Bcast(&bcast_data, 3, MPI_BCAST_DATATYPE, mpi::ROOT_ID, comm);

    N = bcast_data[0];
    root_row = bcast_data[1];
    root_col = bcast_data[2];
    NB = N / proc_rows;

    if (NB == 0) NB = N;

    nrows = numroc_(&N, &NB, &myrow, &iZERO, &proc_rows);

    std::vector<COMPLEX> local_v(nrows);

    ILP_TYPE sendr = 0, sendc = 0, recvr = 0, recvc = 0;
    for (ILP_TYPE r = 0; r < N; r += NB, sendr = (sendr + 1) % proc_rows) {
        sendc = 0;
        ILP_TYPE nr = NB;
        if (N - r < NB) nr = N - r;

        if (rank == root_id) {
            for (ILP_TYPE sendc = 0; sendc < proc_cols; sendc++) {
                ILP_TYPE send_id = blacs_pnum(&ctxt, &sendr, &sendc);
                MPI_Send(v.data() + r, nr, MPI_DOUBLE_COMPLEX, send_id, 0, MPI_COMM_WORLD);
            }
        }

        if (myrow == sendr && mycol == sendc) {
            MPI_Recv(local_v.data() + recvr,
                     nr,
                     MPI_DOUBLE_COMPLEX,
                     root_id,
                     MPI_ANY_TAG,
                     MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        }

        if (myrow == sendr) recvr = (recvr + nr) % nrows;
    }

    return local_v;
}

template <>
void mpi::gather_blacs_vector<double>(const std::vector<double>& local_y,
                                      std::vector<double>& y,
                                      ILP_TYPE N,
                                      ILP_TYPE NB,
                                      ILP_TYPE nrows,
                                      ILP_TYPE ctxt,
                                      ILP_TYPE root_id) {
    ILP_TYPE iZERO = 0;
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ILP_TYPE myrow, mycol;
    ILP_TYPE proc_rows, proc_cols;
    blacs_gridinfo(ctxt, proc_rows, proc_cols, myrow, mycol);

    ILP_TYPE root_row, root_col;
    ILP_TYPE bcast_data[2];
    if (rank == root_id) {
        bcast_data[0] = myrow;
        bcast_data[1] = mycol;
    }

    MPI_Bcast(&bcast_data, 2, MPI_BCAST_DATATYPE, mpi::ROOT_ID, MPI_COMM_WORLD);

    root_row = bcast_data[0];
    root_col = bcast_data[1];

    ILP_TYPE sendr = 0, sendc = 0, recvr = 0, recvc = 0;
    for (ILP_TYPE r = 0; r < N; r += NB, sendr = (sendr + 1) % proc_rows) {
        sendc = 0;
        ILP_TYPE nr = NB;
        if (N - r < NB) nr = N - r;

        if (myrow == sendr && mycol == 0) {
            MPI_Send(local_y.data() + recvr, nr, MPI_DOUBLE, root_id, 0, MPI_COMM_WORLD);
        }

        if (rank == root_id) {
            ILP_TYPE source_id = blacs_pnum(&ctxt, &sendr, &iZERO);
            MPI_Recv(y.data() + r, nr, MPI_DOUBLE, source_id, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (myrow == sendr) recvr = (recvr + nr) % nrows;
    }
}

template <>
void mpi::gather_blacs_vector<COMPLEX>(const std::vector<COMPLEX>& local_y,
                                       std::vector<COMPLEX>& y,
                                       ILP_TYPE N,
                                       ILP_TYPE NB,
                                       ILP_TYPE nrows,
                                       ILP_TYPE ctxt,
                                       ILP_TYPE root_id) {
    ILP_TYPE iZERO = 0;
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ILP_TYPE myrow, mycol;
    ILP_TYPE proc_rows, proc_cols;
    blacs_gridinfo(ctxt, proc_rows, proc_cols, myrow, mycol);

    ILP_TYPE root_row, root_col;
    ILP_TYPE bcast_data[2];
    if (rank == root_id) {
        bcast_data[0] = myrow;
        bcast_data[1] = mycol;
    }

    MPI_Bcast(&bcast_data, 2, MPI_BCAST_DATATYPE, mpi::ROOT_ID, MPI_COMM_WORLD);

    root_row = bcast_data[0];
    root_col = bcast_data[1];

    ILP_TYPE sendr = 0, sendc = 0, recvr = 0, recvc = 0;
    for (ILP_TYPE r = 0; r < N; r += NB, sendr = (sendr + 1) % proc_rows) {
        sendc = 0;
        ILP_TYPE nr = NB;
        if (N - r < NB) nr = N - r;

        if (myrow == sendr && mycol == 0) {
            MPI_Send(local_y.data() + recvr, nr, MPI_DOUBLE_COMPLEX, root_id, 0, MPI_COMM_WORLD);
        }

        if (rank == root_id) {
            ILP_TYPE source_id = blacs_pnum(&ctxt, &sendr, &iZERO);
            MPI_Recv(y.data() + r, nr, MPI_DOUBLE_COMPLEX, source_id, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (myrow == sendr) recvr = (recvr + nr) % nrows;
    }
}

template <>
std::vector<double> mpi::get_diagonal_elements<double>(Matrix<double>& localA, const std::vector<ILP_TYPE>& desca) {
    std::vector<double> res(desca[2]);
    char chA = 'A';
    char TopI = 'I';
    for (int i = 0; i < desca[2]; i++) {
        int index = i + 1;
        pdelget_(&chA, &TopI, &res[i], localA.data(), &index, &index, desca.data());
    }

    return res;
}

template <>
std::vector<COMPLEX> mpi::get_diagonal_elements<COMPLEX>(Matrix<COMPLEX>& localA, const std::vector<ILP_TYPE>& desca) {
    std::vector<COMPLEX> res(desca[2]);
    char chA = 'A';
    char TopI = 'I';
    for (int i = 0; i < desca[2]; i++) {
        int index = i + 1;
        pzelget_(&chA, &TopI, &res[i], localA.data(), &index, &index, desca.data());
    }

    return res;
}

void mpi::parallel_dgemm(const Matrix<double>& A,
                         const Matrix<double>& B,
                         Matrix<double>& C,
                         const std::vector<ILP_TYPE>& desca,
                         const std::vector<ILP_TYPE>& descb,
                         const std::vector<ILP_TYPE>& descc,
                         double alpha,
                         double betta,
                         char op_A,
                         char op_B) {
    ILP_TYPE NA, MA, NB, MB, NB_A, MB_A, NB_B, MB_B;

    NA = desca[2];
    MA = desca[3];
    NB = descb[2];
    MB = descb[3];

    char N = 'N';
    ILP_TYPE iONE = 1;

    pdgemm_(&op_A,
            &op_B,
            &NA,
            &MB,
            &MA,
            &alpha,
            A.data(),
            &iONE,
            &iONE,
            desca.data(),
            B.data(),
            &iONE,
            &iONE,
            descb.data(),
            &betta,
            C.data(),
            &iONE,
            &iONE,
            descc.data());
}

void mpi::parallel_zgemm(const Matrix<COMPLEX>& A,
                         const Matrix<COMPLEX>& B,
                         Matrix<COMPLEX>& C,
                         const std::vector<ILP_TYPE>& desca,
                         const std::vector<ILP_TYPE>& descb,
                         const std::vector<ILP_TYPE>& descc,
                         COMPLEX alpha,
                         COMPLEX betta,
                         char op_A,
                         char op_B) {
    ILP_TYPE iZERO = 0;
    ILP_TYPE NA, MA, NB, MB, NB_A, MB_A, NB_B, MB_B;

    NA = desca[2];
    MA = desca[3];
    NB = descb[2];
    MB = descb[3];

    ILP_TYPE iONE = 1;

    pzgemm_(&op_A,
            &op_B,
            &NA,
            &MB,
            &MA,
            &alpha,
            A.data(),
            &iONE,
            &iONE,
            desca.data(),
            B.data(),
            &iONE,
            &iONE,
            descb.data(),
            &betta,
            C.data(),
            &iONE,
            &iONE,
            descc.data());
}

void mpi::parallel_zhemm(char side,
                         const Matrix<COMPLEX>& A,
                         const Matrix<COMPLEX>& B,
                         Matrix<COMPLEX>& C,
                         const std::vector<ILP_TYPE>& desca,
                         const std::vector<ILP_TYPE>& descb,
                         const std::vector<ILP_TYPE>& descc,
                         COMPLEX alpha,
                         COMPLEX betta) {
    ILP_TYPE NA, MA, NB, MB, NB_A, MB_A, NB_B, MB_B;

    NA = desca[2];
    MA = desca[3];
    NB = descb[2];
    MB = descb[3];

    ILP_TYPE iONE = 1;
    char uplo = 'U';

    pzhemm_(&side,
            &uplo,
            &NA,
            &MB,
            &alpha,
            A.data(),
            &iONE,
            &iONE,
            desca.data(),
            B.data(),
            &iONE,
            &iONE,
            descb.data(),
            &betta,
            C.data(),
            &iONE,
            &iONE,
            descc.data());
}

void mpi::parallel_dgeadd(const Matrix<double>& A,
                          Matrix<double>& C,
                          const std::vector<ILP_TYPE>& desca,
                          const std::vector<ILP_TYPE>& descc,
                          double alpha,
                          double betta,
                          char op_A) {
    ILP_TYPE iONE = 1;
    ILP_TYPE NA = desca[2];
    ILP_TYPE MA = desca[3];

    pdgeadd(
        &op_A, &NA, &MA, &alpha, A.data(), &iONE, &iONE, desca.data(), &betta, C.data(), &iONE, &iONE, descc.data());
}

void mpi::parallel_zgeadd(const Matrix<COMPLEX>& A,
                          Matrix<COMPLEX>& C,
                          const std::vector<ILP_TYPE>& desca,
                          const std::vector<ILP_TYPE>& descc,
                          COMPLEX alpha,
                          COMPLEX betta,
                          char op_A) {
    ILP_TYPE iONE = 1;
    ILP_TYPE NA = desca[2];
    ILP_TYPE MA = desca[3];

    pzgeadd(
        &op_A, &NA, &MA, &alpha, A.data(), &iONE, &iONE, desca.data(), &betta, C.data(), &iONE, &iONE, descc.data());
}

// only for distributed version
void mpi::parallel_dgemv(const Matrix<double>& A,
                         const std::vector<double>& x,
                         std::vector<double>& y,
                         const std::vector<ILP_TYPE>& desca,
                         const std::vector<ILP_TYPE>& descx,
                         const std::vector<ILP_TYPE>& descy,
                         char op_A) {
    ILP_TYPE N_A, M_A;
    N_A = desca[2];
    M_A = desca[3];

    int iONE = 1;
    double alpha = 1.0;
    double betta = 0;

    pdgemv_(&op_A,
            &N_A,
            &M_A,
            &alpha,
            A.data(),
            &iONE,
            &iONE,
            desca.data(),
            x.data(),
            &iONE,
            &iONE,
            descx.data(),
            &iONE,
            &betta,
            y.data(),
            &iONE,
            &iONE,
            descy.data(),
            &iONE);
}

// only for distributed version
void mpi::parallel_zgemv(const Matrix<COMPLEX>& A,
                         const std::vector<COMPLEX>& x,
                         std::vector<COMPLEX>& y,
                         const std::vector<ILP_TYPE>& desca,
                         const std::vector<ILP_TYPE>& descx,
                         const std::vector<ILP_TYPE>& descy,
                         char op_A) {
    ILP_TYPE N_A, M_A;
    N_A = desca[2];
    M_A = desca[3];

    int iONE = 1;
    COMPLEX alpha(1.0, 0);
    COMPLEX betta(0, 0);

    pzgemv_(&op_A,
            &N_A,
            &M_A,
            &alpha,
            A.data(),
            &iONE,
            &iONE,
            desca.data(),
            x.data(),
            &iONE,
            &iONE,
            descx.data(),
            &iONE,
            &betta,
            y.data(),
            &iONE,
            &iONE,
            descy.data(),
            &iONE);
}

COMPLEX mpi::parallel_zdotu(const std::vector<COMPLEX>& x,
                            const std::vector<COMPLEX>& y,
                            const std::vector<ILP_TYPE>& descx,
                            ILP_TYPE incx,
                            const std::vector<ILP_TYPE>& descy,
                            ILP_TYPE incy) {
    ILP_TYPE n = descx[2];
    int iONE = 1;
    COMPLEX dotu;

    pzdotu(&n, &dotu, x.data(), &iONE, &iONE, descx.data(), &incx, y.data(), &iONE, &iONE, descy.data(), &incy);

    return dotu;
}

COMPLEX mpi::parallel_zdotc(const std::vector<COMPLEX>& x,
                            const std::vector<COMPLEX>& y,
                            const std::vector<ILP_TYPE>& descx,
                            ILP_TYPE incx,
                            const std::vector<ILP_TYPE>& descy,
                            ILP_TYPE incy) {
    ILP_TYPE n = descx[2];
    int iONE = 1;
    COMPLEX dotc;

    pzdotc(&n, &dotc, x.data(), &iONE, &iONE, descx.data(), &incx, y.data(), &iONE, &iONE, descy.data(), &incy);

    return dotc;
}

double mpi::parallel_ddot(const std::vector<double>& x,
                          const std::vector<double>& y,
                          const std::vector<ILP_TYPE>& descx,
                          ILP_TYPE incx,
                          const std::vector<ILP_TYPE>& descy,
                          ILP_TYPE incy) {
    ILP_TYPE n = descx[2];
    int iONE = 1;
    double dotu;

    pddot(&n, &dotu, x.data(), &iONE, &iONE, descx.data(), &incx, y.data(), &iONE, &iONE, descy.data(), &incy);

    return dotu;
}

COMPLEX mpi::parallel_zscal(std::vector<COMPLEX>& x, COMPLEX a, const std::vector<ILP_TYPE>& descx, ILP_TYPE incx) {
    ILP_TYPE n = descx[2];
    int iONE = 1;

    pzscal(&n, &a, x.data(), &iONE, &iONE, descx.data(), &incx);

    return a;
}

double mpi::parallel_dscal(std::vector<double>& x, double a, const std::vector<ILP_TYPE>& descx, ILP_TYPE incx) {
    ILP_TYPE n = descx[2];
    int iONE = 1;

    pdscal(&n, &a, x.data(), &iONE, &iONE, descx.data(), &incx);

    return a;
}

void mpi::parallel_daxpy(const std::vector<double>& x,
                         std::vector<double>& y,
                         const std::vector<ILP_TYPE>& descx,
                         ILP_TYPE incx,
                         const std::vector<ILP_TYPE>& descy,
                         ILP_TYPE incy,
                         double alpha) {
    ILP_TYPE iONE = 1;
    ILP_TYPE n = descx[2];
    pdaxpy(&n, &alpha, x.data(), &iONE, &iONE, descx.data(), &incx, y.data(), &iONE, &iONE, descy.data(), &incy);
}
void mpi::parallel_zaxpy(const std::vector<COMPLEX>& x,
                         std::vector<COMPLEX>& y,
                         const std::vector<ILP_TYPE>& descx,
                         ILP_TYPE incx,
                         const std::vector<ILP_TYPE>& descy,
                         ILP_TYPE incy,
                         COMPLEX alpha) {
    ILP_TYPE iONE = 1;
    ILP_TYPE n = descx[2];
    pzaxpy(&n, &alpha, x.data(), &iONE, &iONE, descx.data(), &incx, y.data(), &iONE, &iONE, descy.data(), &incy);
}

#endif  // ENABLE_CLUSTER

}  // namespace QComputations
#endif  // ENABLE_MPI

#ifdef ENABLE_MPI

#define MKL_Complex16 std::complex<double>

#include "mpi_functions.hpp"
#include "functions.hpp"
#include "hamiltonian.hpp"
#include "dynamic.hpp"

#ifdef ENABLE_CLUSTER

#include <mkl_pblas.h>
#include <mkl_scalapack.h>
#include <mkl_blacs.h>

/*
extern "C" {
    int numroc_( int *n, int *nb, int *iproc, int *isrcproc, int *nprocs);
    int indxl2g_(int*, int*, int*, int*, int*);
    void descinit_(int *desc, int *m, int *n, int *mb, int *nb, int *irsrc, int *icsrc, int *ictxt, int
*lld, int *info);
    void pdgemm_( char *TRANSA,char *TRANSB,int *M,int *N,int *K,double
*ALPHA,double *A,int *IA,int *JA,int *DESCA,
 double * B, int * IB, int * JB, int * DESCB,double * BETA,double * C, int * IC, int *
JC, int * DESCC );
}
*/

#endif

void mpi::make_command(COMMAND::COMMANDS command) {
    MPI_Bcast(&command, 1, MPI_INT, ROOT_ID, MPI_COMM_WORLD);
}

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

void mpi::RING_Bcast(double *buf, int count, MPI_Datatype type, int root, MPI_Comm comm) {
  int me, np;
  MPI_Status status;

  MPI_Comm_rank(comm, &me);
  MPI_Comm_size(comm, &np);
  if(me != root)
    MPI_Recv(buf, count, type, (me-1+np)%np, MPI_ANY_TAG, comm, &status);
  if( (me+1)%np != root)
    MPI_Send(buf, count, type, (me+1)%np, 0, comm);
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

State mpi::bcast_state(const State& state) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //std::cout << "RANK = " << rank << std::endl;
    const int data_size = 3;
    size_t data[data_size];
    std::vector<size_t> grid_config;
    std::vector<COMPLEX> gamma;
    if (rank == ROOT_ID) {
        data[0] = state.max_N();
        data[1] = state.min_N();
        data[2] = state.cavities_count();
        std::cout << data[2] << std::endl;
        gamma = state.get_gamma().get_mass();

        for (size_t i = 0; i < data[2]; i++) {
            grid_config.emplace_back(state.m(i));
        }
    }

    MPI_Bcast(data, data_size, MPI_UNSIGNED_LONG, ROOT_ID, MPI_COMM_WORLD);

    auto cavities_count = data[2];

    if (rank != ROOT_ID) {
        grid_config.resize(cavities_count);
        gamma.resize(cavities_count * cavities_count);
        //std::cout << "SIZE: " << rank << " " << gamma.size() << " " << cavities_count << std::endl;
    }
    //std::cout << "DATA - " << rank << " " << data[0] << " " << data[1] << " " << data[2] << std::endl;
    MPI_Bcast(grid_config.data(), cavities_count, MPI_UNSIGNED_LONG, ROOT_ID, MPI_COMM_WORLD);
    MPI_Bcast(gamma.data(), cavities_count * cavities_count, MPI_DOUBLE_COMPLEX, ROOT_ID, MPI_COMM_WORLD);

    //std::cout << "GRID - " << rank << " : ";
    //show_vector(grid_config);
    State res(grid_config);
    res.set_max_N(data[0]);
    res.set_min_N(data[1]);
    res.set_gamma(Matrix<COMPLEX>(gamma, cavities_count, cavities_count, true)); // true - c_style

    //std::cout << rank << " " << res.to_string() << std::endl;
    return res;
}

// ------------------------------------ RUN_MPI_SLAVES --------------------------

void mpi::run_mpi_slaves(const std::map<int, std::vector<MPI_Data>>& data) {
    std::vector<int> commands_count(COMMAND::COMMANDS_COUNT, 0);
    while(true) {
        COMMAND::COMMANDS command;
        MPI_Bcast(&command, 1, MPI_INT, ROOT_ID, MPI_COMM_WORLD);
        int command_id = commands_count[command];

        if (command == COMMAND::STOP) {
            break;
        }

        if (command == COMMAND::GENERATE_H) {
            State tmp = bcast_state();
            //std::cout << tmp.to_string() << std::endl;
            H_TCH H(tmp);
        } else if (command == COMMAND::GENERATE_H_FUNC) {
            H_by_func H(data.at(command)[command_id].n, data.at(command)[command_id].func);
        } else if (command == COMMAND::SCHRODINGER) {
            auto init_state = bcast_vector_complex();
            //show_vector(init_state);
            auto time_vec = bcast_vector_double();
            auto H = Hamiltonian();
            Evolution::schrodinger(init_state, H, time_vec);

        // not ready, hard realization
        } else if (command == COMMAND::QME) {
            auto init_state = bcast_vector_complex();

        // NOT EFFECTIVE, DELETE
        } else if (command == COMMAND::CANNON_MULTIPLY) {
            int bcastdata[4];
            MPI_Bcast(&bcastdata, 4, MPI_INT, mpi::ROOT_ID, MPI_COMM_WORLD);
            if (bcastdata[3] == MPI_Datatype_ID::INT) {
                Matrix<int> tmp;
                Cannon_Multiply(tmp, tmp, tmp, bcastdata[0], bcastdata[1], bcastdata[2]);
            } else if (bcastdata[3] == MPI_Datatype_ID::DOUBLE) {
                Matrix<double> tmp;
                Cannon_Multiply(tmp, tmp, tmp, bcastdata[0], bcastdata[1], bcastdata[2]);
            } else if (bcastdata[3] == MPI_Datatype_ID::DOUBLE_COMPLEX) {
                Matrix<COMPLEX> tmp;
                Cannon_Multiply(tmp, tmp, tmp, bcastdata[0], bcastdata[1], bcastdata[2]);
            } else {
                MPI_Abort(MPI_COMM_WORLD, 4);
            }

        // NOT EFFECTIVE, DELETE
        } else if(command == COMMAND::DIM_MULTIPLY) {
            int bcast_data[3];
            MPI_Bcast(&bcast_data, 3, MPI_INT, mpi::ROOT_ID, MPI_COMM_WORLD);
            int n = bcast_data[0], m = bcast_data[1];
            Matrix<COMPLEX> B(n, m);

            if (bcast_data[2] == MPI_Datatype_ID::DOUBLE_COMPLEX) {
                MPI_Bcast(B.data(), n * m, MPI_DOUBLE_COMPLEX, mpi::ROOT_ID, MPI_COMM_WORLD);
                Matrix<COMPLEX> tmp;
                Dim_Multiply(tmp, B, tmp);
            } else {
                MPI_Abort(MPI_COMM_WORLD, 5);
            }
#ifdef ENABLE_CLUSTER 
        } else if (command == COMMAND::P_GEMM_MULTIPLY) {
            int datatype;
            MPI_Bcast(&datatype, 1, MPI_INT, mpi::ROOT_ID, MPI_COMM_WORLD);

            if (datatype == MPI_Datatype_ID::DOUBLE_COMPLEX) {
                Matrix<COMPLEX> tmp;
                parallel_zgemm(tmp, tmp, tmp);
            } else if (datatype == MPI_Datatype_ID::DOUBLE) {
                Matrix<double> tmp;
                parallel_dgemm(tmp, tmp, tmp);
            } else {
                MPI_Abort(MPI_COMM_WORLD, 6);
            }
#endif
        } else if (command == COMMAND::EXIT_FROM_FUNC) {
            return;
        } else {
            std::cerr << "UNKNOWN/UNAVAILABLE COMMAND - " << command << std::endl;
        }

        commands_count[command]++;
    }
}

void mpi::stop_mpi_slaves() {
    auto command = COMMAND::STOP;
    make_command(command);
    MPI_Finalize();
}

template<>
void mpi::Cannon_Multiply <COMPLEX>(const Matrix<COMPLEX>& A, const Matrix<COMPLEX>& B, Matrix<COMPLEX>& C, int grid_size, int block_size, int n) {
    MPI_Datatype datatype = MPI_DOUBLE_COMPLEX;
    
    auto begin = std::chrono::steady_clock::now();

    int rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int dims[2] = {grid_size, grid_size};
    int periods[2] = {1, 1};
    MPI_Comm COMM_2D;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &COMM_2D);

    MPI_Datatype tmp_type, block_type;
    int starts[2] = {0, 0};
    int global_size[2] = {n, n};
    int local_size[2] = {block_size, block_size};

    MPI_Type_create_subarray(2, global_size, local_size, starts, MPI_ORDER_C, datatype, &tmp_type);
    MPI_Type_create_resized(tmp_type, 0, block_size * sizeof(COMPLEX), &block_type);
    MPI_Type_commit(&block_type);

    Matrix<COMPLEX> localA(block_size, block_size);
    Matrix<COMPLEX> localB(block_size, block_size);
    Matrix<COMPLEX> localC(block_size, block_size, COMPLEX(0));

    std::vector<int> sendcounts(world_size);
    std::vector<int> displs(world_size);

    if (rank == 0) {
        sendcounts = std::vector<int>(world_size, 1);

        int disp = 0;
        for (size_t i = 0; i < grid_size; i++) {
            for (size_t j = 0; j < grid_size; j++) {
                displs[i * grid_size + j] = disp;
                disp++;
            }

            disp += (block_size - 1) * grid_size;
        }
    }

    MPI_Scatterv(A.data(), sendcounts.data(), displs.data(), block_type, localA.data(),
    n * n / (world_size), datatype,
    0, COMM_2D);
	MPI_Scatterv(B.data(), sendcounts.data(), displs.data(), block_type, localB.data(),
    n * n / (world_size), datatype,
    0, COMM_2D);

    int rank2d;
    int coords[2];
    int left, right, up, down;
    MPI_Comm_rank(COMM_2D, &rank2d);
    MPI_Cart_coords(COMM_2D, rank2d, 2, coords);
	MPI_Cart_shift(COMM_2D, 1, coords[0], &left, &right);
	MPI_Sendrecv_replace(localA.data(), block_size * block_size, datatype, left, 1, right, 1, COMM_2D, MPI_STATUS_IGNORE);
	MPI_Cart_shift(COMM_2D, 0, coords[1], &up, &down);
	MPI_Sendrecv_replace(localB.data(), block_size * block_size, datatype, up, 1, down, 1, COMM_2D, MPI_STATUS_IGNORE);

    auto end = std::chrono::steady_clock::now();
    if (rank2d == 0) {
        std::cout << "SCATTER: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
    }

    for (size_t i = 0; i < grid_size; i++) {
        COMPLEX alpha(1, 0);
        COMPLEX betta(1, 0);
        begin = std::chrono::steady_clock::now();
        //Matrix<COMPLEX> tmp(block_size, block_size);
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                block_size, block_size, block_size, &alpha, localA.data(),
                block_size, localB.data(), block_size, &betta,
                localC.data(), block_size);
        //Multiply_Matrix(localA, localB, localC);
        end = std::chrono::steady_clock::now();
        std::cout << "cblas: " << rank2d << " " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;

        //std::cout << coords[0] << " " << coords[1] << ": \n";
        //localC.show();
        MPI_Cart_shift(COMM_2D, 1, 1, &left, &right);
        MPI_Cart_shift(COMM_2D, 0, 1, &up, &down);
        MPI_Sendrecv_replace(localA.data(), block_size * block_size, datatype, left, 1, right, 1, COMM_2D, MPI_STATUS_IGNORE);
        MPI_Sendrecv_replace(localB.data(), block_size * block_size, datatype, up, 1, down, 1, COMM_2D, MPI_STATUS_IGNORE);
    }

    MPI_Gatherv(localC.data(), n * n / world_size, datatype,
	C.data(), sendcounts.data(), displs.data(), block_type,
	0, COMM_2D);
}

template<>
void mpi::Cannon_Multiply <double>(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C, int grid_size, int block_size, int n) {
    MPI_Datatype datatype = MPI_DOUBLE;
    
    int rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int dims[2] = {grid_size, grid_size};
    int periods[2] = {1, 1};
    MPI_Comm COMM_2D;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &COMM_2D);

    MPI_Datatype tmp_type, block_type;
    int starts[2] = {0, 0};
    int global_size[2] = {n, n};
    int local_size[2] = {block_size, block_size};

    MPI_Type_create_subarray(2, global_size, local_size, starts, MPI_ORDER_C, datatype, &tmp_type);
    MPI_Type_create_resized(tmp_type, 0, block_size * sizeof(double), &block_type);
    MPI_Type_commit(&block_type);

    Matrix<double> localA(block_size, block_size);
    Matrix<double> localB(block_size, block_size);

    Matrix<double> localC(block_size, block_size, 0.0);

    std::vector<int> sendcounts(world_size);
    std::vector<int> displs(world_size);

    if (rank == 0) {
        sendcounts = std::vector<int>(world_size, 1);

        int disp = 0;
        for (size_t i = 0; i < grid_size; i++) {
            for (size_t j = 0; j < grid_size; j++) {
                displs[i * grid_size + j] = disp;
                disp++;
            }

            disp += (block_size - 1) * grid_size;
        }
    }

    MPI_Scatterv(A.data(), sendcounts.data(), displs.data(), block_type, localA.data(),
    n * n / (world_size), datatype,
    0, MPI_COMM_WORLD);
	MPI_Scatterv(B.data(), sendcounts.data(), displs.data(), block_type, localB.data(),
    n * n / (world_size), datatype,
    0, MPI_COMM_WORLD);

    int rank2d;
    int coords[2];
    int left, right, up, down;
    MPI_Comm_rank(COMM_2D, &rank2d);
    MPI_Cart_coords(COMM_2D, rank2d, 2, coords);
	MPI_Cart_shift(COMM_2D, 1, coords[0], &left, &right);
	MPI_Sendrecv_replace(localA.data(), block_size * block_size, datatype, left, 1, right, 1, COMM_2D, MPI_STATUS_IGNORE);
	MPI_Cart_shift(COMM_2D, 0, coords[1], &up, &down);
	MPI_Sendrecv_replace(localB.data(), block_size * block_size, datatype, up, 1, down, 1, COMM_2D, MPI_STATUS_IGNORE);

    for (size_t i = 0; i < grid_size; i++) {
        cblas_MM_double(localA.data(),
                        localB.data(), localC.data(),
                        block_size, block_size, block_size, 1.0, 1.0);

        MPI_Cart_shift(COMM_2D, 1, 1, &left, &right);
        MPI_Cart_shift(COMM_2D, 0, 1, &up, &down);
        MPI_Sendrecv_replace(localA.data(), block_size * block_size, datatype, left, 1, right, 1, COMM_2D, MPI_STATUS_IGNORE);
        MPI_Sendrecv_replace(localB.data(), block_size * block_size, datatype, up, 1, down, 1, COMM_2D, MPI_STATUS_IGNORE);
    }

    MPI_Gatherv(localC.data(), n * n / world_size, datatype,
	C.data(), sendcounts.data(), displs.data(), block_type,
	0, MPI_COMM_WORLD);
}

template<>
void mpi::Cannon_Multiply <int>(const Matrix<int>& A, const Matrix<int>& B, Matrix<int>& C, int grid_size, int block_size, int n) {
    MPI_Datatype datatype = MPI_INT;
    
    int rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int dims[2] = {grid_size, grid_size};
    int periods[2] = {1, 1};
    MPI_Comm COMM_2D;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &COMM_2D);

    MPI_Datatype tmp_type, block_type;
    int starts[2] = {0, 0};
    int global_size[2] = {n, n};
    int local_size[2] = {block_size, block_size};

    MPI_Type_create_subarray(2, global_size, local_size, starts, MPI_ORDER_C, datatype, &tmp_type);
    MPI_Type_create_resized(tmp_type, 0, block_size * sizeof(int), &block_type);
    MPI_Type_commit(&block_type);

    Matrix<int> localA(block_size, block_size);
    Matrix<int> localB(block_size, block_size);
    Matrix<int> localC(block_size, block_size, 0);

    std::vector<int> sendcounts(world_size);
    std::vector<int> displs(world_size);

    if (rank == 0) {
        sendcounts = std::vector<int>(world_size, 1);

        int disp = 0;
        for (size_t i = 0; i < grid_size; i++) {
            for (size_t j = 0; j < grid_size; j++) {
                displs[i * grid_size + j] = disp;
                disp++;
            }

            disp += (block_size - 1) * grid_size;
        }
    }

    MPI_Scatterv(A.data(), sendcounts.data(), displs.data(), block_type, localA.data(),
    n * n / (world_size), datatype,
    0, MPI_COMM_WORLD);
	MPI_Scatterv(B.data(), sendcounts.data(), displs.data(), block_type, localB.data(),
    n * n / (world_size), datatype,
    0, MPI_COMM_WORLD);

    int rank2d;
    int coords[2];
    int left, right, up, down;
    MPI_Comm_rank(COMM_2D, &rank2d);
    MPI_Cart_coords(COMM_2D, rank2d, 2, coords);
	MPI_Cart_shift(COMM_2D, 1, coords[0], &left, &right);
	MPI_Sendrecv_replace(localA.data(), block_size * block_size, datatype, left, 1, right, 1, COMM_2D, MPI_STATUS_IGNORE);
	MPI_Cart_shift(COMM_2D, 0, coords[1], &up, &down);
	MPI_Sendrecv_replace(localB.data(), block_size * block_size, datatype, up, 1, down, 1, COMM_2D, MPI_STATUS_IGNORE);

    for (size_t i = 0; i < grid_size; i++) {
        cblas_MM_int(localA.data(),
                        localB.data(), localC.data(),
                        block_size, block_size, block_size, 1.0, 1.0);

        MPI_Cart_shift(COMM_2D, 1, 1, &left, &right);
        MPI_Cart_shift(COMM_2D, 0, 1, &up, &down);
        MPI_Sendrecv_replace(localA.data(), block_size * block_size, datatype, left, 1, right, 1, COMM_2D, MPI_STATUS_IGNORE);
        MPI_Sendrecv_replace(localB.data(), block_size * block_size, datatype, up, 1, down, 1, COMM_2D, MPI_STATUS_IGNORE);
    }

    MPI_Gatherv(localC.data(), n * n / world_size, datatype,
	C.data(), sendcounts.data(), displs.data(), block_type,
	0, MPI_COMM_WORLD);
}

template<>
void mpi::Dim_Multiply<COMPLEX>(const Matrix<COMPLEX>& A, const Matrix<COMPLEX>& B, Matrix<COMPLEX>& C) {
    MPI_Datatype datatype = MPI_DOUBLE_COMPLEX;
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int bcast_data[3];
    if (rank == mpi::ROOT_ID) {
        bcast_data[0] = A.n();
        bcast_data[1] = A.m();
        bcast_data[2] = B.m();
    }

    MPI_Bcast(&bcast_data, 3, MPI_INT, ROOT_ID, MPI_COMM_WORLD);

    int n = bcast_data[0], k = bcast_data[1], m = bcast_data[2];

    if (n >= k) {
        // Rows parallel
        std::vector<int> rows_map(world_size, n / world_size);
        std::vector<int> displs(world_size);
        int index = 0;
        for (size_t i = 0; i < world_size; i++) {
            displs[i] = index;
            if (n % world_size > i) {
                rows_map[i]++;
            }

            index += rows_map[i];
        }

        MPI_Datatype tmp_type, row_type;
        int global_size[2] = {n, k};
        int local_size[2] = {1, k};
        int starts[2] = {0, 0};

        MPI_Type_create_subarray(2, global_size, local_size, starts, MPI_ORDER_C, datatype, &tmp_type);
        MPI_Type_create_resized(tmp_type, 0, k * sizeof(COMPLEX), &row_type);
        MPI_Type_commit(&row_type);

        Matrix<COMPLEX> localA(rows_map[rank], k);
        Matrix<COMPLEX> localC(rows_map[rank], m);

        MPI_Scatterv(A.data(), rows_map.data(), displs.data(), row_type, localA.data(),
        rows_map[rank] * k, datatype,
        0, MPI_COMM_WORLD);

        COMPLEX alpha(1, 0);
        COMPLEX betta(1, 0);
        //Matrix<COMPLEX> tmp(block_size, block_size);
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            rows_map[rank], k, m, &alpha, localA.data(),
            k, B.data(), m, &betta,
            localC.data(), m);

        MPI_Gatherv(localC.data(), rows_map[rank] * k, datatype,
        C.data(), rows_map.data(), displs.data(), row_type,
        0, MPI_COMM_WORLD);
    } else {
        // Cols parallel

    }
}

// -------------------------------------------- BLACS, PBLAS -----------------------------------------------------

#ifdef ENABLE_CLUSTER
    

MPI_Datatype Create_Block_Type_double (LP_TYPE N, LP_TYPE M, LP_TYPE NB, LP_TYPE MB) {
    MPI_Datatype tmp_type, block_type;
    int starts[2] = {0, 0};
    int global_size[2] = {N, M};
    int local_size[2] = {NB, MB};

    MPI_Type_create_subarray(2, global_size, local_size, starts, MPI_ORDER_C, MPI_DOUBLE, &tmp_type);
    MPI_Type_commit(&block_type);

    return block_type;
}

namespace {

    // FOR ROW MAJOR ORDER 

    void my_dgesd2d(LP_TYPE N, LP_TYPE M, LP_TYPE row_index, LP_TYPE col_index, const Matrix<double>& A, LP_TYPE send_id) {
        auto sub = A.submatrix(N, M, row_index, col_index);
        //sub.show();
        //auto block_type = Create_Block_Type_double(bcast_data[0], bcast_data[1], bcast_data[2], bcast_data[3]);
        //MPI_Send(A.data() + A.index(row_index, col_index), 1, block_type, send_id, 0, MPI_COMM_WORLD);
        MPI_Send(sub.data(), M * N, MPI_DOUBLE, send_id, 0, MPI_COMM_WORLD);
    }

    void my_dgerv2d(LP_TYPE N, LP_TYPE M, LP_TYPE offset, Matrix<double>& A, LP_TYPE LDA, LP_TYPE source_id) {
        Matrix<double> tmp(A.is_c_style(), N, M);
        MPI_Recv(tmp.data(), N * M, MPI_DOUBLE, source_id, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                A[offset / LDA + i][offset % LDA + j] = tmp[i][j];
            }
        }
    }

    void my_zgesd2d(LP_TYPE N, LP_TYPE M, LP_TYPE row_index, LP_TYPE col_index, const Matrix<COMPLEX>& A, LP_TYPE LDA, LP_TYPE send_id) {
        auto sub = A.submatrix(N, M, row_index, col_index);
        //sub.show();
        MPI_Send(sub.data(), M * N, MPI_DOUBLE_COMPLEX, send_id, 0, MPI_COMM_WORLD);
    }

    void my_zgerv2d(LP_TYPE N, LP_TYPE M, LP_TYPE offset, Matrix<COMPLEX>& A, LP_TYPE LDA, LP_TYPE source_id) {
        MPI_Recv(A.data() + offset, M * N, MPI_DOUBLE_COMPLEX, source_id, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }


    // NEED TO REPLACE
    void ScatterBLACSMatrix_double(const Matrix<double>& A, LP_TYPE NA,
                                    LP_TYPE MA, Matrix<double>& localA,
                                    LP_TYPE NB_A, LP_TYPE MB_A, LP_TYPE nrows_A,
                                    LP_TYPE ncols_A, LP_TYPE myrow, LP_TYPE mycol,
                                    LP_TYPE proc_rows, LP_TYPE proc_cols,
                                    LP_TYPE rank, LP_TYPE* p_ctxt) {

        auto A_tmp = A;
        LP_TYPE iZERO = 0;
        LP_TYPE sendr = 0, sendc = 0, recvr = 0, recvc = 0;
        for (LP_TYPE r = 0; r < NA; r += NB_A, sendr = (sendr + 1) % proc_rows) {
            sendc = 0;
            // Number of rows to be sent
            // Is this the last row block?
            LP_TYPE nr = NB_A;
            if (NA - r < NB_A)
                nr = NA - r;
    
            for (LP_TYPE c = 0; c < MA; c += MB_A, sendc = (sendc + 1) % proc_cols) {
                // Number of cols to be sent
                // Is this the last col block?
                LP_TYPE nc = MB_A;
                if (MA - c < MB_A)
                    nc = MA - c;
    
                if (rank == mpi::ROOT_ID) {
                    // Send a nr-by-nc submatrix to process (sendr, sendc)
                    LP_TYPE send_id = blacs_pnum(p_ctxt, &sendr, &sendc);
                    //std::cout << "Send to " << send_id << " : " << sendr << " " << sendc << std::endl;
                    //std::cout << nr << " " << nc << " " << r << " " << c << std::endl;
                    my_dgesd2d(nr, nc, r, c, A, send_id);
                    //dgesd2d(p_ctxt, &nr, &nc, A_tmp.data() + NA * c + r, &NA, &sendr, &sendc);
                }
    
                if (myrow == sendr && mycol == sendc) {
                    //std::cout << "Recv " << myrow << " " << mycol << " " << recvr << " " << recvc << std::endl;
                    // Receive the same data
                    // The leading dimension of the local matrix is nrows!
                    my_dgerv2d(nr, nc, ncols_A * recvr + recvc, localA, ncols_A, mpi::ROOT_ID);
                    //dgerv2d(p_ctxt, &nr, &nc, localA.data() + nrows_A * recvc + recvr, &nrows_A, &iZERO, &iZERO);
                    recvc = (recvc + nc) % ncols_A;
                }

                MPI_Barrier(MPI_COMM_WORLD);
            }

            if (myrow == sendr)
                recvr = (recvr + nr) % nrows_A;
        }
    }

    // NEED TO REPLACE
    void ScatterBLACSMatrix_COMPLEX(const Matrix<COMPLEX>& A, LP_TYPE NA,
                                    LP_TYPE MA, Matrix<COMPLEX>& localA,
                                    LP_TYPE NB_A, LP_TYPE MB_A, LP_TYPE nrows_A,
                                    LP_TYPE ncols_A, LP_TYPE myrow, LP_TYPE mycol,
                                    LP_TYPE proc_rows, LP_TYPE proc_cols,
                                    LP_TYPE rank, LP_TYPE* p_ctxt) {

        LP_TYPE sendr = 0, sendc = 0, recvr = 0, recvc = 0;
        for (LP_TYPE r = 0; r < NA; r += NB_A, sendr = (sendr + 1) % proc_rows) {
            sendc = 0;
            // Number of rows to be sent
            // Is this the last row block?
            LP_TYPE nr = NB_A;
            if (NA - r < NB_A)
                nr = NA - r;
    
            for (LP_TYPE c = 0; c < MA; c += MB_A, sendc = (sendc + 1) % proc_cols) {
                // Number of cols to be sent
                // Is this the last col block?
                LP_TYPE nc = MB_A;
                if (MA - c < MB_A)
                    nc = MA - c;
    
                if (rank == mpi::ROOT_ID) {
                    // Send a nr-by-nc submatrix to process (sendr, sendc)
                    LP_TYPE send_id = blacs_pnum(p_ctxt, &sendr, &sendc);
                    //std::cout << "Send to " << send_id << " : " << sendr << " " << sendc << std::endl;
                    my_zgesd2d(nr, nc, r, c, A, NA, send_id);
                    //zgesd2d(&ctxt, &nr, &nc, A.data() + NA * c + r, NA, sendr, sendc);
                }
    
                if (myrow == sendr && mycol == sendc) {
                    //std::cout << "Recv " << myrow << " " << mycol << " " << recvr << " " << recvc << std::endl;
                    // Receive the same data
                    // The leading dimension of the local matrix is nrows!
                    my_zgerv2d(nr, nc, ncols_A * recvr + recvc, localA, ncols_A, mpi::ROOT_ID);
                    //zgerv2d(&ctxt, &nr, &nc, localA.data() + nrows_A * recvc + recvr, nrows_A, 0, 0);
                    recvc = (recvc + nc) % ncols_A;
                }
            }

            if (myrow == sendr)
                recvr = (recvr + nr) % nrows_A;
        }
    }
}

void print_distributed_matrix(const Matrix<double>& A, const std::string& matrix_name, MPI_Comm comm) {
    int myid, numproc;
    MPI_Comm_rank(comm, &myid);
    MPI_Comm_size(comm, &numproc);

    for (LP_TYPE id = 0; id < numproc; ++id) {
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

void init_grid(LP_TYPE& ctxt) {
    LP_TYPE iZERO = 0;
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    LP_TYPE myid, numproc, myrow, mycol;
    char order = 'R';
    LP_TYPE proc_rows = std::sqrt(world_size), proc_cols = world_size / proc_rows;
    //std::cout << rank << " Here1\n";
    blacs_pinfo(&myid, &numproc);
    LP_TYPE iMINUS = -1;
    blacs_get(&iMINUS, &iZERO, &ctxt);
    //std::cout << rank << " Here3\n";
    blacs_gridinit(&ctxt, &order, &proc_rows, &proc_cols);
}

template<>
Matrix<double> mpi::scatter_blacs_matrix<double>(const Matrix<double>& A, LP_TYPE& N, LP_TYPE& M,
                                      LP_TYPE& NB, LP_TYPE& MB, LP_TYPE& nrows,
                                      LP_TYPE& ncols, LP_TYPE& ctxt, LP_TYPE root_id,
                                      LP_TYPE NB_FORCE, LP_TYPE MB_FORCE) {
    LP_TYPE iZERO = 0;
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    LP_TYPE myrow, mycol;
    char order = 'R';
    LP_TYPE proc_rows, proc_cols;
    blacs_gridinfo(&ctxt, &proc_rows, &proc_cols, &myrow, &mycol);

    LP_TYPE root_row, root_col;
    LP_TYPE bcast_data[2];
    if (rank == root_id) {
        bcast_data[0] = A.n();
        bcast_data[1] = A.m();
        bcast_data[2] = myrow;
        bcast_data[3] = mycol;
    }

    MPI_Bcast(&bcast_data, 4, MPI_BCAST_DATATYPE, mpi::ROOT_ID, MPI_COMM_WORLD);

    N = bcast_data[0];
    M = bcast_data[1];
    root_row = bcast_data[2];
    root_col = bcast_data[3];
    NB = N / proc_rows;
    MB = M / proc_cols;

    nrows = numroc_(&N, &NB, &myrow, &iZERO, &proc_rows);
    ncols = numroc_(&M, &MB, &mycol, &iZERO, &proc_cols);

    Matrix<double> localA(A.is_c_style(), nrows, ncols);

    LP_TYPE A_LD = A.LD();
    LP_TYPE localA_LD = localA.LD();

    LP_TYPE sendr = 0, sendc = 0, recvr = 0, recvc = 0;
    for (LP_TYPE r = 0; r < N; r += NB, sendr = (sendr + 1) % proc_rows) {
        sendc = 0;
        // Number of rows to be sent
        // Is this the last row block?
        LP_TYPE nr = NB;
        if (N - r < NB)
            nr = N - r;

        for (LP_TYPE c = 0; c < M; c += MB, sendc = (sendc + 1) % proc_cols) {
            // Number of cols to be sent
            // Is this the last col block?
            LP_TYPE nc = MB;
            if (M - c < MB)
                nc = M - c;

            if (rank == root_id) {
                // Send a nr-by-nc submatrix to process (sendr, sendc)
                LP_TYPE send_id = blacs_pnum(&ctxt, &sendr, &sendc);
                //std::cout << "Send to " << send_id << " : " << sendr << " " << sendc << std::endl;
                if (A.is_c_style()) {
                    my_dgesd2d(nr, nc, r, c, A, send_id);
                } else {
                    dgesd2d(&ctxt, &nr, &nc, A.data() + A.index(r, c), &A_LD, &sendr, &sendc);
                }
            }

            if (myrow == sendr && mycol == sendc) {
                //std::cout << "Recv " << myrow << " " << mycol << " " << recvr << " " << recvc << std::endl;
                // Receive the same data
                // The leading dimension of the local matrix is nrows!
                if (localA.is_c_style()) {
                    my_dgerv2d(nr, nc, ncols * recvr + recvc, localA, ncols, root_id);
                } else {
                    dgerv2d(&ctxt, &nr, &nc, localA.data() + localA.index(recvr, recvc), &localA_LD, &root_row, &root_col);
                }
                recvc = (recvc + nc) % ncols;
            }
        }

        if (myrow == sendr)
            recvr = (recvr + nr) % nrows;
    }

    return localA;
}

void mpi::parallel_dgemm(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C,
                         bool is_distributed, LP_TYPE* desca, LP_TYPE* descb, LP_TYPE* descc) {
    LP_TYPE iZERO = 0;
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    LP_TYPE ctxt;
    LP_TYPE NA, MA, NB, MB, NB_A, MB_A, NB_B, MB_B;
    LP_TYPE LLD_A, LLD_B, LLD_C;
    LP_TYPE nrows_A, nrows_B, nrows_C, ncols_A, ncols_B, ncols_C;
    LP_TYPE proc_rows, proc_cols, myrow, mycol;
    Matrix<double> localA;
    Matrix<double> localB;
    Matrix<double> localC;
    if (!is_distributed) {
        init_grid(ctxt);
        blacs_gridinfo(&ctxt, &proc_rows, &proc_cols, &myrow, &mycol);
        localA = mpi::scatter_blacs_matrix<double>(A, NA, MA, NB_A, MB_A, nrows_A, ncols_A, ctxt, mpi::ROOT_ID);
        localB = mpi::scatter_blacs_matrix<double>(B, NB, MB, NB_B, MB_B, nrows_B, ncols_B, ctxt, mpi::ROOT_ID);

        nrows_C = numroc_(&NA, &NB_A, &myrow, &iZERO, &proc_rows);
        ncols_C = numroc_(&MB, &MB_B, &mycol, &iZERO, &proc_cols);

        localC = Matrix<double>(nrows_C, ncols_C);

        LLD_A = nrows_A;
        LLD_B = nrows_B;
        LLD_C = nrows_C;

        LP_TYPE rsrc = 0, csrc = 0, info;
        descinit_(desca, &NA, &MA, &NB_A, &MB_A, &rsrc, &csrc, &ctxt, &LLD_A, &info);
        if (info != 0) std::cout << "ERROR OF descinit__A: " << rank << " " << info << std::endl;
        descinit_(descb, &NB, &MB, &NB_B, &MB_B, &rsrc, &csrc, &ctxt, &LLD_B, &info);
        if (info != 0) std::cout << "ERROR OF descinit__B: " << rank << " " << info << std::endl;
        descinit_(descc, &NA, &MB, &NB_A, &MB_B, &rsrc, &csrc, &ctxt, &LLD_C, &info);
        if (info != 0) std::cout << "ERROR OF descinit__C: " << rank << " " << info << std::endl;

        print_distributed_matrix(localA, "A");
        print_distributed_matrix(localB, "B");
    } else {
        NA = desca[2];
        MA = desca[3];
        NB = descb[2];
        MB = descb[3];
    }
    /*
    //std::cout << world_size << std::endl;
    LP_TYPE myid, numproc, ctxt, myrow, mycol;
    char order = 'R';
    LP_TYPE proc_rows = std::sqrt(world_size), proc_cols = world_size / proc_rows;
    //std::cout << rank << " Here1\n";
    blacs_pinfo(&myid, &numproc);
    LP_TYPE iMINUS = -1;
    blacs_get(&iMINUS, &iZERO, &ctxt);
    //std::cout << rank << " Here3\n";
    blacs_gridinit(&ctxt, &order, &proc_rows, &proc_cols);
    //std::cout << rank << " Here4\n";
    blacs_gridinfo(&ctxt, &proc_rows, &proc_cols, &myrow, &mycol);

    int bcast_data[4];
    if (rank == mpi::ROOT_ID) {
        bcast_data[0] = A.n();
        bcast_data[1] = A.m();
        bcast_data[2] = B.n();
        bcast_data[3] = B.m();
    }

    MPI_Bcast(&bcast_data, 4, MPI_INT, mpi::ROOT_ID, MPI_COMM_WORLD);

    LP_TYPE NA = bcast_data[0], MA = bcast_data[1], NB = bcast_data[2], MB = bcast_data[3];
    LP_TYPE NB_A = NA / proc_rows, MB_A = MA / proc_cols, NB_B = NB / proc_rows, MB_B = MB / proc_cols;

    NB_A = 4000;
    MB_A = 4000;
    NB_B = 4000;
    MB_B = 4000;
    LP_TYPE nrows_A = numroc_(&NA, &NB_A, &myrow, &iZERO, &proc_rows);
    LP_TYPE ncols_A = numroc_(&MA, &MB_A, &mycol, &iZERO, &proc_cols);
    LP_TYPE nrows_B = numroc_(&NB, &NB_B, &myrow, &iZERO, &proc_rows);
    LP_TYPE ncols_B = numroc_(&MB, &MB_B, &mycol, &iZERO, &proc_cols);
    LP_TYPE nrows_C = numroc_(&NA, &NB_A, &myrow, &iZERO, &proc_rows);
    LP_TYPE ncols_C = numroc_(&MB, &MB_B, &mycol, &iZERO, &proc_cols);

    LP_TYPE LLD_A = std::max<LP_TYPE>(1, nrows_A);
    LP_TYPE LLD_B = std::max<LP_TYPE>(1, nrows_B);
    LP_TYPE LLD_C = std::max<LP_TYPE>(1, nrows_C);

    Matrix<double> localA(nrows_A, ncols_A);
    Matrix<double> localB(nrows_B, ncols_B);
    Matrix<double> localC(nrows_C, ncols_C);

    */

    /*
    for (LP_TYPE id = 0; id < numproc; ++id) {
        if (id == myid) {
            std::cout << "data on node " << myid << std::endl;
            std::cout << myrow << " " << mycol << std::endl;
            std::cout << nrows_A << " " << ncols_A << " " << LLD_A << std::endl;
            std::cout << nrows_B << " " << ncols_B << " " << LLD_B << std::endl;
            std::cout << nrows_C << " " << ncols_C << " " << LLD_C << std::endl;
            std::flush(std::cout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    */
    /*
    ScatterBLACSMatrix_double(A, NA, MA, localA, NB_A, MB_A,
                               nrows_A, ncols_A, myrow, mycol,
                               proc_rows, proc_cols, myid, &ctxt);

    ScatterBLACSMatrix_double(B, NB, MB, localB, NB_B, MB_B,
                               nrows_B, ncols_B, myrow, mycol,
                               proc_rows, proc_cols, myid, &ctxt);

    LP_TYPE rsrc = 0, csrc = 0, info;
    descinit_(desca, &NA, &MA, &NB_A, &MB_A, &rsrc, &csrc, &ctxt, &LLD_A, &info);
    if (info != 0) std::cout << "ERROR OF descinit__A: " << rank << " " << info << std::endl;
    descinit_(descb, &NB, &MB, &NB_B, &MB_B, &rsrc, &csrc, &ctxt, &LLD_B, &info);
    if (info != 0) std::cout << "ERROR OF descinit__B: " << rank << " " << info << std::endl;
    descinit_(descc, &NA, &MB, &NB_A, &MB_B, &rsrc, &csrc, &ctxt, &LLD_C, &info);
    if (info != 0) std::cout << "ERROR OF descinit__C: " << rank << " " << info << std::endl;
    */
    /*
    for (size_t id = 0; id < numproc; id++) {
        if (id == myid) {
            std::cout << "NODE - " << id << std::endl;
            for (size_t i = 0; i < 9; i++) {
                std::cout << desca[i] << " ";
            }

            std::cout << std::endl;
            for (size_t i = 0; i < 9; i++) {
                std::cout << descb[i] << " ";
            }
            std::cout << std::endl;
            for (size_t i = 0; i < 9; i++) {
                std::cout << descc[i] << " ";
            }
            std::cout << std::endl;
            std::flush(std::cout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    */
    /*
    Matrix<double> TA(NA, MA);
    Matrix<double> TB(NB, MB);

    if (rank == mpi::ROOT_ID) { TA = A; TB = B;}

    MPI_Bcast(TA.data(), NA * MA, MPI_DOUBLE, mpi::ROOT_ID, MPI_COMM_WORLD);
    MPI_Bcast(TB.data(), NB * MB, MPI_DOUBLE, mpi::ROOT_ID, MPI_COMM_WORLD);

    for (LP_TYPE iloc = 0; iloc < nrows_A; iloc++) {
        for (LP_TYPE jloc = 0; jloc < ncols_A; jloc++) {
            LP_TYPE fortidl = iloc + 1;
            LP_TYPE fortjdl = jloc + 1;
            LP_TYPE i = indxl2g_(&fortidl, &NB_A, &myrow, &iZERO, &proc_rows)-1;
            LP_TYPE j = indxl2g_(&fortjdl, &MB_A, &mycol, &iZERO, &proc_cols)-1;
            localA[iloc][jloc]=TA[i][j];
        }
    }

    for (LP_TYPE iloc = 0; iloc < nrows_A; iloc++) {
        for (LP_TYPE jloc = 0; jloc < ncols_A; jloc++) {
            LP_TYPE fortidl = iloc + 1;
            LP_TYPE fortjdl = jloc + 1;
            LP_TYPE i = indxl2g_(&fortidl, &NB_B, &myrow, &iZERO, &proc_rows)-1;
            LP_TYPE j = indxl2g_(&fortjdl, &MB_B, &mycol, &iZERO, &proc_cols)-1;
            localB[iloc][jloc]=TB[i][j];
        }
    }
    */

    /*
    for (LP_TYPE id = 0; id < numproc; ++id) {
       if (id == myid) {
       std::cout << "A_loc on node " << myid << std::endl;
           for (LP_TYPE r = 0; r < nrows_A; ++r) {
               for (LP_TYPE c = 0; c < ncols_A; ++c)
                   std::cout << std::setw(config::WIDTH) << localA[r][c] << " ";
           std::cout << std::endl;
           std::flush(std::cout);
            }
           std::cout << std::endl;
       }

    //blacs_barrier(&ctxt, "All");
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (LP_TYPE id = 0; id < numproc; ++id) {
        if (id == myid) {
        std::cout << "B_loc on node " << myid << std::endl;
        for (LP_TYPE r = 0; r < nrows_B; ++r) {
           for (LP_TYPE c = 0; c < ncols_B; ++c)
            std::cout << std::setw(config::WIDTH) << localB[r][c] << " ";
            std::cout << std::endl;
            std::flush(std::cout);
       }
       std::cout << std::endl;
       }

    //blacs_barrier(&ctxt, "All");
    MPI_Barrier(MPI_COMM_WORLD);
    }
    */

   bool return_to_c_style = false;

    if (localA.is_c_style()) {
        return_to_c_style = true;
        localA.to_fortran_style();
    }

    if (localB.is_c_style()) {
        localB.to_fortran_style();
    }

/*
    1 2 3 4
    5 6 7 8
    9 10 11 12

    1 2 3 4 5 6 7 8 9 10 11 12
    1 5 9 2 6 10 3 7 11 4 8 12
*/
    /*
    if (rank == 0) {
    localA.set_multiply_mode(config::COMMON_MODE);
    auto check = localA * localB;

    for (int id = 0; id < numproc; ++id) {
        if (id == myid) {
            std::cout << "check_loc on node " << myid << std::endl;
            for (int r = 0; r < nrows_C; ++r) {
                for (int c = 0; c < ncols_C; ++c)
                    std::cout << std::setw(config::WIDTH) << check[r][c] << " ";
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }

        //blacs_barrier(&ctxt, "All");
    }

    }
    MPI_Barrier(MPI_COMM_WORLD);
    */

    /*
    for (LP_TYPE id = 0; id < numproc; ++id) {
       if (id == myid) {
       std::cout << "A_loc on node " << myid << std::endl;
           for (LP_TYPE r = 0; r < nrows_A; ++r) {
               for (LP_TYPE c = 0; c < ncols_A; ++c)
                   std::cout << std::setw(config::WIDTH) << localA.data()[c * nrows_A + r] << " ";
           std::cout << std::endl;
           std::flush(std::cout);
            }
           std::cout << std::endl;
       }

    //blacs_barrier(&ctxt, "All");
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (LP_TYPE id = 0; id < numproc; ++id) {
        if (id == myid) {
        std::cout << "B_loc on node " << myid << std::endl;
        for (LP_TYPE r = 0; r < nrows_B; ++r) {
           for (LP_TYPE c = 0; c < ncols_B; ++c)
            std::cout << std::setw(config::WIDTH) << localB.data()[c * nrows_B + r] << " ";
            std::cout << std::endl;
            std::flush(std::cout);
       }
       std::cout << std::endl;
       }

    //blacs_barrier(&ctxt, "All");
    MPI_Barrier(MPI_COMM_WORLD);
    }
    */
    //_MKL_Complex16 alpha;
    //alpha.real = 1;
    //alpha.imag = 0;
    //_MKL_Complex16 betta;
    //betta.real = 0;
    //betta.imag = 0;

    char N = 'N';
    LP_TYPE iONE = 1;
    double alpha = 1.0;
    double betta = 0;

    if (!is_distributed) {
        auto begin = std::chrono::steady_clock::now();
        pdgemm_(&N, &N, &NA, &MB, &MA, &alpha, localA.data(), &iONE, &iONE, desca,
                                        localB.data(), &iONE, &iONE, descb,
                                        &betta, localC.data(), &iONE, &iONE, descc);
        auto end = std::chrono::steady_clock::now();
        if (rank == mpi::ROOT_ID) std::cout << "PDGEMM: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
    } else {
        auto begin = std::chrono::steady_clock::now();
        pdgemm_(&N, &N, &NA, &MB, &MA, &alpha, A.data(), &iONE, &iONE, desca,
                                        B.data(), &iONE, &iONE, descb,
                                        &betta, C.data(), &iONE, &iONE, descc);
        auto end = std::chrono::steady_clock::now();
        if (rank == mpi::ROOT_ID) std::cout << "PDGEMM: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;  
    }

    if (return_to_c_style) {
        localC.to_c_style();
    }
    //pzgemm(&N, &N, &NA, &MB, &MA, &alpha, reinterpret_cast<_MKL_Complex16*>(localA.data()), &iONE, &iONE, desca,
    //                                reinterpret_cast<_MKL_Complex16*>(localB.data()), &iONE, &iONE, descb,
    //                                &betta, reinterpret_cast<_MKL_Complex16*>(localC.data()), &iONE, &iONE, descc);
    /*
    int sendr = 0, sendc = 0, recvr = 0, recvc = 0;
    for (int r = 0; r < NA; r += NB_A, sendr = (sendr + 1) % proc_rows) {
        sendc = 0;
        // Number of rows to be sent
        // Is this the last row block?
        int nr = NB_A;
        if (NA - r < NB_A)
            nr = NA - r;
 
        for (int c = 0; c < MA; c += MB_A, sendc = (sendc + 1) % proc_cols) {
            // Number of cols to be sent
            // Is this the last col block?
            int nc = MB_A;
            if (MA - c < MB_A)
                nc = MA - c;
 
            if (rank == mpi::ROOT_ID) {
                // Send a nr-by-nc submatrix to process (sendr, sendc)
                int send_id = blacs_pnum(&ctxt, &sendr, &sendc);
                std::cout << "Send to " << send_id << " : " << sendr << " " << sendc << std::endl;
                my_zgesd2d(nr, nc, r, c, A, NA, send_id);
                //zgesd2d(&ctxt, &nr, &nc, A.data() + NA * c + r, NA, sendr, sendc);
            }
 
            if (myrow == sendr && mycol == sendc) {
                std::cout << "Recv " << myrow << " " << mycol << " " << recvr << " " << recvc << std::endl;
                // Receive the same data
                // The leading dimension of the local matrix is nrows!
                my_zgerv2d(nr, nc, ncols_A * recvr + recvc, localA, ncols_A, mpi::ROOT_ID);
                //zgerv2d(&ctxt, &nr, &nc, localA.data() + nrows_A * recvc + recvr, nrows_A, 0, 0);
                recvc = (recvc + nc) % ncols_A;
            }
        }

        if (myrow == sendr)
            recvr = (recvr + nr) % nrows_A;
    }
    */
    print_distributed_matrix(localC, "C");
    if (!is_distributed) {
        blacs_gridexit(&ctxt);
    }
    //blacs_exit(&iZERO);
}

void mpi::parallel_zgemm(const Matrix<COMPLEX>& A, const Matrix<COMPLEX>& B, Matrix<COMPLEX>& C) {
    using M_COMPLEX = MKL_Complex16;
    LP_TYPE iZERO = 0;
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    //std::cout << world_size << std::endl;
    LP_TYPE myid, numproc, ctxt, myrow, mycol;
    char order = 'R';
    LP_TYPE proc_rows = sqrt(world_size), proc_cols = sqrt(world_size);
    //std::cout << rank << " Here1\n";
    blacs_pinfo(&myid, &numproc);
    blacs_get(&iZERO, &iZERO, &ctxt);
    //std::cout << rank << " Here3\n";
    blacs_gridinit(&ctxt, &order, &proc_rows, &proc_cols);
    //std::cout << rank << " Here4\n";
    blacs_gridinfo(&ctxt, &proc_rows, &proc_cols, &myrow, &mycol);
    //std::cout << rank << " Here5\n";

    int bcast_data[4];
    if (rank == mpi::ROOT_ID) {
        bcast_data[0] = A.n();
        bcast_data[1] = A.m();
        bcast_data[2] = B.n();
        bcast_data[3] = B.m();
    }

    MPI_Bcast(&bcast_data, 4, MPI_INT, mpi::ROOT_ID, MPI_COMM_WORLD);

    LP_TYPE NA = bcast_data[0], MA = bcast_data[1], NB = bcast_data[2], MB = bcast_data[3];
    LP_TYPE NB_A = NA / proc_rows, MB_A = MA / proc_cols, NB_B = NB / proc_rows, MB_B = MB / proc_cols;

    NB_A = 2;
    MB_A = 2;
    NB_B = 2;
    MB_B = 2;
    LP_TYPE nrows_A = numroc_(&NA, &NB_A, &myrow, &iZERO, &proc_rows);
    LP_TYPE ncols_A = numroc_(&MA, &MB_A, &mycol, &iZERO, &proc_cols);
    LP_TYPE nrows_B = numroc_(&NB, &NB_B, &myrow, &iZERO, &proc_rows);
    LP_TYPE ncols_B = numroc_(&MB, &MB_B, &mycol, &iZERO, &proc_cols);
    LP_TYPE nrows_C = numroc_(&NA, &NB_A, &myrow, &iZERO, &proc_rows);
    LP_TYPE ncols_C = numroc_(&MB, &MB_B, &mycol, &iZERO, &proc_cols);

    LP_TYPE LLD_A = std::max<LP_TYPE>(1, nrows_A);
    LP_TYPE LLD_B = std::max<LP_TYPE>(1, nrows_B);
    LP_TYPE LLD_C = std::max<LP_TYPE>(1, nrows_C);

    Matrix<COMPLEX> localA(nrows_A, ncols_A);
    Matrix<COMPLEX> localB(nrows_B, ncols_B);
    Matrix<COMPLEX> localC(nrows_C, ncols_C);
    LP_TYPE desca[9];
    LP_TYPE descb[9];
    LP_TYPE descc[9];

    for (LP_TYPE id = 0; id < numproc; ++id) {
        if (id == myid) {
            std::cout << "data on node " << myid << std::endl;
            std::cout << myrow << " " << mycol << std::endl;
            std::cout << nrows_A << " " << ncols_A << " " << LLD_A << std::endl;
            std::cout << nrows_B << " " << ncols_B << " " << LLD_B << std::endl;
            std::cout << nrows_C << " " << ncols_C << " " << LLD_C << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    LP_TYPE rsrc = 0, csrc = 0, info;
    descinit_(desca, &NA, &MA, &NB_A, &MB_A, &rsrc, &csrc, &ctxt, &LLD_A, &info);
    if (info != 0) std::cout << "ERROR OF descinit__A: " << rank << " " << info << std::endl;
    descinit_(descb, &NB, &MB, &NB_B, &MB_B, &rsrc, &csrc, &ctxt, &LLD_B, &info);
    if (info != 0) std::cout << "ERROR OF descinit__B: " << rank << " " << info << std::endl;
    descinit_(descc, &NA, &MB, &NB_A, &MB_B, &rsrc, &csrc, &ctxt, &LLD_C, &info);
    if (info != 0) std::cout << "ERROR OF descinit__C: " << rank << " " << info << std::endl;

    ScatterBLACSMatrix_COMPLEX(A, NA, MA, localA, NB_A, MB_A,
                               nrows_A, ncols_A, myrow, mycol,
                               proc_rows, proc_cols, myid, &ctxt);

    ScatterBLACSMatrix_COMPLEX(B, NB, MB, localB, NB_B, MB_B,
                               nrows_B, ncols_B, myrow, mycol,
                               proc_rows, proc_cols, myid, &ctxt);

    /*
    if (rank == 0) {
    localA.set_multiply_mode(config::COMMON_MODE);
    auto check = localA * localB;

    for (int id = 0; id < numproc; ++id) {
        if (id == myid) {
            std::cout << "check_loc on node " << myid << std::endl;
            for (int r = 0; r < nrows_C; ++r) {
                for (int c = 0; c < ncols_C; ++c)
                    std::cout << std::setw(config::WIDTH) << check[r][c] << " ";
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }

        //blacs_barrier(&ctxt, "All");
    }

    }
    MPI_Barrier(MPI_COMM_WORLD);
    */

    for (LP_TYPE id = 0; id < numproc; ++id) {
       if (id == myid) {
       std::cout << "A_loc on node " << myid << std::endl;
           for (LP_TYPE r = 0; r < nrows_A; ++r) {
               for (LP_TYPE c = 0; c < ncols_A; ++c)
                   std::cout << std::setw(config::WIDTH) << localA[r][c] << " ";
           std::cout << std::endl;
           std::flush(std::cout);
            }
           std::cout << std::endl;
       }

    //blacs_barrier(&ctxt, "All");
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (LP_TYPE id = 0; id < numproc; ++id) {
        if (id == myid) {
        std::cout << "B_loc on node " << myid << std::endl;
        for (LP_TYPE r = 0; r < nrows_B; ++r) {
           for (LP_TYPE c = 0; c < ncols_B; ++c)
            std::cout << std::setw(config::WIDTH) << localB[r][c] << " ";
            std::cout << std::endl;
            std::flush(std::cout);
       }
       std::cout << std::endl;
       }

    //blacs_barrier(&ctxt, "All");
    MPI_Barrier(MPI_COMM_WORLD);
    }

    //_MKL_Complex16 alpha;
    //alpha.real = 1;
    //alpha.imag = 0;
    //_MKL_Complex16 betta;
    //betta.real = 0;
    //betta.imag = 0;

    char N = 'N';
    LP_TYPE iONE = 1;
    COMPLEX alpha(1, 0);
    COMPLEX betta(0, 0);
    //pzgemm(&N, &N, &NA, &MB, &MA, &alpha, localA.data(), &iONE, &iONE, desca,
    //                                localB.data(), &iONE, &iONE, descb,
    //                                &betta, localC.data(), &iONE, &iONE, descc);
    //pzgemm(&N, &N, &NA, &MB, &MA, &alpha, reinterpret_cast<_MKL_Complex16*>(localA.data()), &iONE, &iONE, desca,
    //                                reinterpret_cast<_MKL_Complex16*>(localB.data()), &iONE, &iONE, descb,
    //                                &betta, reinterpret_cast<_MKL_Complex16*>(localC.data()), &iONE, &iONE, descc);
    /*
    int sendr = 0, sendc = 0, recvr = 0, recvc = 0;
    for (int r = 0; r < NA; r += NB_A, sendr = (sendr + 1) % proc_rows) {
        sendc = 0;
        // Number of rows to be sent
        // Is this the last row block?
        int nr = NB_A;
        if (NA - r < NB_A)
            nr = NA - r;
 
        for (int c = 0; c < MA; c += MB_A, sendc = (sendc + 1) % proc_cols) {
            // Number of cols to be sent
            // Is this the last col block?
            int nc = MB_A;
            if (MA - c < MB_A)
                nc = MA - c;
 
            if (rank == mpi::ROOT_ID) {
                // Send a nr-by-nc submatrix to process (sendr, sendc)
                int send_id = blacs_pnum(&ctxt, &sendr, &sendc);
                std::cout << "Send to " << send_id << " : " << sendr << " " << sendc << std::endl;
                my_zgesd2d(nr, nc, r, c, A, NA, send_id);
                //zgesd2d(&ctxt, &nr, &nc, A.data() + NA * c + r, NA, sendr, sendc);
            }
 
            if (myrow == sendr && mycol == sendc) {
                std::cout << "Recv " << myrow << " " << mycol << " " << recvr << " " << recvc << std::endl;
                // Receive the same data
                // The leading dimension of the local matrix is nrows!
                my_zgerv2d(nr, nc, ncols_A * recvr + recvc, localA, ncols_A, mpi::ROOT_ID);
                //zgerv2d(&ctxt, &nr, &nc, localA.data() + nrows_A * recvc + recvr, nrows_A, 0, 0);
                recvc = (recvc + nc) % ncols_A;
            }
        }

        if (myrow == sendr)
            recvr = (recvr + nr) % nrows_A;
    }
    */

    MPI_Barrier(MPI_COMM_WORLD);
    for (LP_TYPE id = 0; id < numproc; ++id) {
        if (id == myid) {
            std::cout << "C_loc on node " << myid << std::endl;
            for (LP_TYPE r = 0; r < nrows_C; ++r) {
                for (LP_TYPE c = 0; c < ncols_C; ++c)
                    std::cout << std::setw(config::WIDTH) << localC[r][c] << " ";
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }

        blacs_barrier(&ctxt, "All");
        //MPI_Barrier(MPI_COMM_WORLD);
    }

    blacs_gridexit(&ctxt);
    //blacs_exit(&iZERO);
}

#endif // ENABLE_CLUSTER

#endif // ENABLE_MPI

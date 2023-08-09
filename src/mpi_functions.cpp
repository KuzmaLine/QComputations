#ifdef ENABLE_MPI

#define MKL_Complex16 std::complex<double>

#include "mpi_functions.hpp"
#include "functions.hpp"
#include "hamiltonian.hpp"
#include "dynamic.hpp"

#ifdef ENABLE_CLUSTER

//#include <mkl_pblas.h>
//#include <mkl_scalapack.h>
#include <mkl_blacs.h>
#include <mkl.h>

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

#endif

void mpi::make_command(int command) {
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
    res.set_gamma(Matrix<COMPLEX>(gamma, cavities_count, cavities_count));

    //std::cout << rank << " " << res.to_string() << std::endl;
    return res;
}

// ------------------------------------ RUN_MPI_SLAVES --------------------------

void mpi::run_mpi_slaves(const std::map<int, std::vector<MPI_Data>>& data) {
    std::vector<int> commands_count(COMMAND::COMMANDS_COUNT, 0);
    while(true) {
        int command;
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
        } else if (command == COMMAND::QME) {
            auto init_state = bcast_vector_complex();
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
        } else if(command == COMMAND::DIM_MULTIPLY) {
            int bcast_data[3];
            MPI_Bcast(&bcast_data, 3, MPI_INT, mpi::ROOT_ID, MPI_COMM_WORLD);
            int n = bcast_data[0], m = bcast_data[1];
            Matrix<COMPLEX> B(n, m);

            if (bcast_data[2] == MPI_Datatype_ID::DOUBLE_COMPLEX) {
                MPI_Bcast(B.mass_data(), n * m, MPI_DOUBLE_COMPLEX, mpi::ROOT_ID, MPI_COMM_WORLD);
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
        } else {
            std::cerr << "UNKNOWN/UNAVAILABLE COMMAND - " << command << std::endl;
        }

        commands_count[command]++;
    }
}

void mpi::stop_mpi_slaves() {
    int command = COMMAND::STOP;
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

    MPI_Scatterv(A.mass_data(), sendcounts.data(), displs.data(), block_type, localA.mass_data(),
    n * n / (world_size), datatype,
    0, COMM_2D);
	MPI_Scatterv(B.mass_data(), sendcounts.data(), displs.data(), block_type, localB.mass_data(),
    n * n / (world_size), datatype,
    0, COMM_2D);

    int rank2d;
    int coords[2];
    int left, right, up, down;
    MPI_Comm_rank(COMM_2D, &rank2d);
    MPI_Cart_coords(COMM_2D, rank2d, 2, coords);
	MPI_Cart_shift(COMM_2D, 1, coords[0], &left, &right);
	MPI_Sendrecv_replace(localA.mass_data(), block_size * block_size, datatype, left, 1, right, 1, COMM_2D, MPI_STATUS_IGNORE);
	MPI_Cart_shift(COMM_2D, 0, coords[1], &up, &down);
	MPI_Sendrecv_replace(localB.mass_data(), block_size * block_size, datatype, up, 1, down, 1, COMM_2D, MPI_STATUS_IGNORE);

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
                block_size, block_size, block_size, &alpha, localA.mass_data(),
                block_size, localB.mass_data(), block_size, &betta,
                localC.mass_data(), block_size);
        //Multiply_Matrix(localA, localB, localC);
        end = std::chrono::steady_clock::now();
        std::cout << "cblas: " << rank2d << " " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;

        //std::cout << coords[0] << " " << coords[1] << ": \n";
        //localC.show();
        MPI_Cart_shift(COMM_2D, 1, 1, &left, &right);
        MPI_Cart_shift(COMM_2D, 0, 1, &up, &down);
        MPI_Sendrecv_replace(localA.mass_data(), block_size * block_size, datatype, left, 1, right, 1, COMM_2D, MPI_STATUS_IGNORE);
        MPI_Sendrecv_replace(localB.mass_data(), block_size * block_size, datatype, up, 1, down, 1, COMM_2D, MPI_STATUS_IGNORE);
    }

    MPI_Gatherv(localC.mass_data(), n * n / world_size, datatype,
	C.mass_data(), sendcounts.data(), displs.data(), block_type,
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

    MPI_Scatterv(A.mass_data(), sendcounts.data(), displs.data(), block_type, localA.mass_data(),
    n * n / (world_size), datatype,
    0, MPI_COMM_WORLD);
	MPI_Scatterv(B.mass_data(), sendcounts.data(), displs.data(), block_type, localB.mass_data(),
    n * n / (world_size), datatype,
    0, MPI_COMM_WORLD);

    int rank2d;
    int coords[2];
    int left, right, up, down;
    MPI_Comm_rank(COMM_2D, &rank2d);
    MPI_Cart_coords(COMM_2D, rank2d, 2, coords);
	MPI_Cart_shift(COMM_2D, 1, coords[0], &left, &right);
	MPI_Sendrecv_replace(localA.mass_data(), block_size * block_size, datatype, left, 1, right, 1, COMM_2D, MPI_STATUS_IGNORE);
	MPI_Cart_shift(COMM_2D, 0, coords[1], &up, &down);
	MPI_Sendrecv_replace(localB.mass_data(), block_size * block_size, datatype, up, 1, down, 1, COMM_2D, MPI_STATUS_IGNORE);

    for (size_t i = 0; i < grid_size; i++) {
        cblas_MM_double(localA.mass_data(),
                        localB.mass_data(), localC.mass_data(),
                        block_size, block_size, block_size, 1.0, 1.0);

        MPI_Cart_shift(COMM_2D, 1, 1, &left, &right);
        MPI_Cart_shift(COMM_2D, 0, 1, &up, &down);
        MPI_Sendrecv_replace(localA.mass_data(), block_size * block_size, datatype, left, 1, right, 1, COMM_2D, MPI_STATUS_IGNORE);
        MPI_Sendrecv_replace(localB.mass_data(), block_size * block_size, datatype, up, 1, down, 1, COMM_2D, MPI_STATUS_IGNORE);
    }

    MPI_Gatherv(localC.mass_data(), n * n / world_size, datatype,
	C.mass_data(), sendcounts.data(), displs.data(), block_type,
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

    MPI_Scatterv(A.mass_data(), sendcounts.data(), displs.data(), block_type, localA.mass_data(),
    n * n / (world_size), datatype,
    0, MPI_COMM_WORLD);
	MPI_Scatterv(B.mass_data(), sendcounts.data(), displs.data(), block_type, localB.mass_data(),
    n * n / (world_size), datatype,
    0, MPI_COMM_WORLD);

    int rank2d;
    int coords[2];
    int left, right, up, down;
    MPI_Comm_rank(COMM_2D, &rank2d);
    MPI_Cart_coords(COMM_2D, rank2d, 2, coords);
	MPI_Cart_shift(COMM_2D, 1, coords[0], &left, &right);
	MPI_Sendrecv_replace(localA.mass_data(), block_size * block_size, datatype, left, 1, right, 1, COMM_2D, MPI_STATUS_IGNORE);
	MPI_Cart_shift(COMM_2D, 0, coords[1], &up, &down);
	MPI_Sendrecv_replace(localB.mass_data(), block_size * block_size, datatype, up, 1, down, 1, COMM_2D, MPI_STATUS_IGNORE);

    for (size_t i = 0; i < grid_size; i++) {
        cblas_MM_int(localA.mass_data(),
                        localB.mass_data(), localC.mass_data(),
                        block_size, block_size, block_size, 1.0, 1.0);

        MPI_Cart_shift(COMM_2D, 1, 1, &left, &right);
        MPI_Cart_shift(COMM_2D, 0, 1, &up, &down);
        MPI_Sendrecv_replace(localA.mass_data(), block_size * block_size, datatype, left, 1, right, 1, COMM_2D, MPI_STATUS_IGNORE);
        MPI_Sendrecv_replace(localB.mass_data(), block_size * block_size, datatype, up, 1, down, 1, COMM_2D, MPI_STATUS_IGNORE);
    }

    MPI_Gatherv(localC.mass_data(), n * n / world_size, datatype,
	C.mass_data(), sendcounts.data(), displs.data(), block_type,
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

        MPI_Scatterv(A.mass_data(), rows_map.data(), displs.data(), row_type, localA.mass_data(),
        rows_map[rank] * k, datatype,
        0, MPI_COMM_WORLD);

        COMPLEX alpha(1, 0);
        COMPLEX betta(1, 0);
        //Matrix<COMPLEX> tmp(block_size, block_size);
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            rows_map[rank], k, m, &alpha, localA.mass_data(),
            k, B.mass_data(), m, &betta,
            localC.mass_data(), m);

        MPI_Gatherv(localC.mass_data(), rows_map[rank] * k, datatype,
        C.mass_data(), rows_map.data(), displs.data(), row_type,
        0, MPI_COMM_WORLD);
    } else {
        // Cols parallel

    }
}

#ifdef ENABLE_CLUSTER

namespace {

#ifdef MKL_ILP64
    using LP_TYPE = long long;
#else
    using LP_TYPE = int;
#endif

    void my_dgesd2d(LP_TYPE M, LP_TYPE N, LP_TYPE row_index, LP_TYPE col_index, const Matrix<double>& A, LP_TYPE LDA, LP_TYPE send_id) {
        auto sub = A.submatrix(M, N, row_index, col_index);
        //sub.show();
        MPI_Send(sub.mass_data(), M * N, MPI_DOUBLE, send_id, 0, MPI_COMM_WORLD);
    }

    void my_dgerv2d(LP_TYPE M, LP_TYPE N, LP_TYPE offset, Matrix<double>& A, LP_TYPE LDA, LP_TYPE source_id) {
        Matrix<double> tmp(M, N);
        MPI_Recv(tmp.mass_data(), M * N, MPI_DOUBLE, source_id, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                A[offset / LDA + i][offset % LDA + j] = tmp[i][j];
            }
        }
    }

    void my_zgesd2d(LP_TYPE M, LP_TYPE N, LP_TYPE row_index, LP_TYPE col_index, const Matrix<COMPLEX>& A, LP_TYPE LDA, LP_TYPE send_id) {
        auto sub = A.submatrix(M, N, row_index, col_index);
        //sub.show();
        MPI_Send(sub.mass_data(), M * N, MPI_DOUBLE_COMPLEX, send_id, 0, MPI_COMM_WORLD);
    }

    void my_zgerv2d(LP_TYPE M, LP_TYPE N, LP_TYPE offset, Matrix<COMPLEX>& A, LP_TYPE LDA, LP_TYPE source_id) {
        MPI_Recv(A.mass_data() + offset, M * N, MPI_DOUBLE_COMPLEX, source_id, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

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
                    my_dgesd2d(nr, nc, r, c, A, NA, send_id);
                    //dgesd2d(p_ctxt, &nr, &nc, A_tmp.mass_data() + NA * c + r, &NA, &sendr, &sendc);
                }
    
                if (myrow == sendr && mycol == sendc) {
                    //std::cout << "Recv " << myrow << " " << mycol << " " << recvr << " " << recvc << std::endl;
                    // Receive the same data
                    // The leading dimension of the local matrix is nrows!
                    my_dgerv2d(nr, nc, ncols_A * recvr + recvc, localA, ncols_A, mpi::ROOT_ID);
                    //dgerv2d(p_ctxt, &nr, &nc, localA.mass_data() + nrows_A * recvc + recvr, &nrows_A, &iZERO, &iZERO);
                    recvc = (recvc + nc) % ncols_A;
                }

                MPI_Barrier(MPI_COMM_WORLD);
            }

            if (myrow == sendr)
                recvr = (recvr + nr) % nrows_A;
        }
    }

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
                    //zgesd2d(&ctxt, &nr, &nc, A.mass_data() + NA * c + r, NA, sendr, sendc);
                }
    
                if (myrow == sendr && mycol == sendc) {
                    //std::cout << "Recv " << myrow << " " << mycol << " " << recvr << " " << recvc << std::endl;
                    // Receive the same data
                    // The leading dimension of the local matrix is nrows!
                    my_zgerv2d(nr, nc, ncols_A * recvr + recvc, localA, ncols_A, mpi::ROOT_ID);
                    //zgerv2d(&ctxt, &nr, &nc, localA.mass_data() + nrows_A * recvc + recvr, nrows_A, 0, 0);
                    recvc = (recvc + nc) % ncols_A;
                }
            }

            if (myrow == sendr)
                recvr = (recvr + nr) % nrows_A;
        }
    }
}

void mpi::parallel_dgemm(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C) {
    LP_TYPE iZERO = 0;
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

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
    MB_A = 2000;
    NB_B = 4000;
    MB_B = 2000;
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
            std::flush(std::cout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

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

    /*
    Matrix<double> TA(NA, MA);
    Matrix<double> TB(NB, MB);

    if (rank == mpi::ROOT_ID) { TA = A; TB = B;}

    MPI_Bcast(TA.mass_data(), NA * MA, MPI_DOUBLE, mpi::ROOT_ID, MPI_COMM_WORLD);
    MPI_Bcast(TB.mass_data(), NB * MB, MPI_DOUBLE, mpi::ROOT_ID, MPI_COMM_WORLD);

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
    localA.to_fortran();
    localB.to_fortran();

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
                   std::cout << std::setw(config::WIDTH) << localA.mass_data()[c * nrows_A + r] << " ";
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
            std::cout << std::setw(config::WIDTH) << localB.mass_data()[c * nrows_B + r] << " ";
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

    auto begin = std::chrono::steady_clock::now();
    pdgemm_(&N, &N, &NA, &MB, &MA, &alpha, localA.mass_data(), &iONE, &iONE, desca,
                                    localB.mass_data(), &iONE, &iONE, descb,
                                    &betta, localC.mass_data(), &iONE, &iONE, descc);
    auto end = std::chrono::steady_clock::now();
    if (rank == mpi::ROOT_ID) std::cout << "PDGEMM: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
    localC.from_fortran();
    //pzgemm(&N, &N, &NA, &MB, &MA, &alpha, reinterpret_cast<_MKL_Complex16*>(localA.mass_data()), &iONE, &iONE, desca,
    //                                reinterpret_cast<_MKL_Complex16*>(localB.mass_data()), &iONE, &iONE, descb,
    //                                &betta, reinterpret_cast<_MKL_Complex16*>(localC.mass_data()), &iONE, &iONE, descc);
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
                //zgesd2d(&ctxt, &nr, &nc, A.mass_data() + NA * c + r, NA, sendr, sendc);
            }
 
            if (myrow == sendr && mycol == sendc) {
                std::cout << "Recv " << myrow << " " << mycol << " " << recvr << " " << recvc << std::endl;
                // Receive the same data
                // The leading dimension of the local matrix is nrows!
                my_zgerv2d(nr, nc, ncols_A * recvr + recvc, localA, ncols_A, mpi::ROOT_ID);
                //zgerv2d(&ctxt, &nr, &nc, localA.mass_data() + nrows_A * recvc + recvr, nrows_A, 0, 0);
                recvc = (recvc + nc) % ncols_A;
            }
        }

        if (myrow == sendr)
            recvr = (recvr + nr) % nrows_A;
    }
    */

    /*
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
    */
    blacs_gridexit(&ctxt);
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
    //pzgemm(&N, &N, &NA, &MB, &MA, &alpha, localA.mass_data(), &iONE, &iONE, desca,
    //                                localB.mass_data(), &iONE, &iONE, descb,
    //                                &betta, localC.mass_data(), &iONE, &iONE, descc);
    //pzgemm(&N, &N, &NA, &MB, &MA, &alpha, reinterpret_cast<_MKL_Complex16*>(localA.mass_data()), &iONE, &iONE, desca,
    //                                reinterpret_cast<_MKL_Complex16*>(localB.mass_data()), &iONE, &iONE, descb,
    //                                &betta, reinterpret_cast<_MKL_Complex16*>(localC.mass_data()), &iONE, &iONE, descc);
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
                //zgesd2d(&ctxt, &nr, &nc, A.mass_data() + NA * c + r, NA, sendr, sendc);
            }
 
            if (myrow == sendr && mycol == sendc) {
                std::cout << "Recv " << myrow << " " << mycol << " " << recvr << " " << recvc << std::endl;
                // Receive the same data
                // The leading dimension of the local matrix is nrows!
                my_zgerv2d(nr, nc, ncols_A * recvr + recvc, localA, ncols_A, mpi::ROOT_ID);
                //zgerv2d(&ctxt, &nr, &nc, localA.mass_data() + nrows_A * recvc + recvr, nrows_A, 0, 0);
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

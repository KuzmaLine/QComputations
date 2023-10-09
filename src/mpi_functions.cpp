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

extern "C" {
    void pdelget_(char*, char*, double*, const double*, int*, int*, const int*);
    void pzelget_(char*, char*, COMPLEX*, const COMPLEX*, int*, int*, const int*);
    ILP_TYPE indxl2g_(ILP_TYPE*, ILP_TYPE*, ILP_TYPE*, ILP_TYPE*, ILP_TYPE*);
    ILP_TYPE indxg2p_(ILP_TYPE*, ILP_TYPE*, ILP_TYPE*, ILP_TYPE*, ILP_TYPE*);
    ILP_TYPE indxg2l_(ILP_TYPE*, ILP_TYPE*, ILP_TYPE*, ILP_TYPE*, ILP_TYPE*);
}

#endif

namespace QComputations {

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
        //std::cout << data[2] << std::endl;
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
    res.set_gamma(Matrix<COMPLEX>(gamma, cavities_count, cavities_count, C_STYLE)); // true - c_style

    std::vector<COMPLEX> leak_from_cavity(cavities_count, 0);
    std::vector<COMPLEX> gain_to_cavity(cavities_count, 0);

    if (rank == ROOT_ID) {
        auto leak_cavities_id = state.get_cavities_with_leak();
        auto gain_cavities_id = state.get_cavities_with_gain();
        for (auto id: leak_cavities_id) {
            leak_from_cavity[id] = state.get_leak_gamma(id);
        }
        for (auto id: gain_cavities_id) {
            gain_to_cavity[id] = state.get_gain_gamma(id);
        }
    }

    MPI_Bcast(leak_from_cavity.data(), cavities_count, MPI_DOUBLE_COMPLEX, ROOT_ID, MPI_COMM_WORLD);
    MPI_Bcast(gain_to_cavity.data(), cavities_count, MPI_DOUBLE_COMPLEX, ROOT_ID, MPI_COMM_WORLD);

    for (size_t i = 0; i < cavities_count; i++) {
        if (!is_zero(leak_from_cavity[i])) {
            res.set_leak_for_cavity(i, leak_from_cavity[i]);
        }

        if (!is_zero(gain_to_cavity[i])) {
            res.set_gain_for_cavity(i, gain_to_cavity[i]);
        }
    }

    //std::cout << rank << " " << res.to_string() << std::endl;
    return res;
}

// ################################ RUN_MPI_SLAVES #######################################

void mpi::run_mpi_slaves(const std::map<int, std::vector<MPI_Data>>& data) {
    std::vector<int> commands_count(COMMAND::COMMANDS_COUNT, 0);
    while(true) {
        //std::cout << "NEW_COMMAND\n";
        COMMAND::COMMANDS command;
        MPI_Bcast(&command, 1, MPI_INT, ROOT_ID, MPI_COMM_WORLD);
        //std::cout << "START_COMMAND\n";
        int command_id = commands_count[command];

        if (command == COMMAND::STOP) {
            break;
        }

// ----------------------------------- GENERATE_H --------------------------------------
        if (command == COMMAND::GENERATE_H) {
            State tmp = bcast_state();
            //std::cout << tmp.to_string() << std::endl;
            H_TCH H(tmp);

// --------------------------------- GENERATE_H_FUNC -----------------------------------
        } else if (command == COMMAND::GENERATE_H_FUNC) {
            H_by_func H(data.at(command)[command_id].n, data.at(command)[command_id].func);

// ---------------------------------- SCHRODINGER --------------------------------------
        } else if (command == COMMAND::SCHRODINGER) {
            auto init_state = bcast_vector_complex();
            //show_vector(init_state);
            auto time_vec = bcast_vector_double();
            auto H = Hamiltonian();
            Evolution::schrodinger(init_state, H, time_vec);

// ------------------------------------- QME -------------------------------------------
#ifdef ENABLE_CLUSTER
        } else if (command == COMMAND::QME) {
            auto init_state = bcast_vector_complex();
            auto time_vec = bcast_vector_double();

            bool is_full_rho;
            MPI_Bcast(&is_full_rho, 1, MPI_C_BOOL, mpi::ROOT_ID, MPI_COMM_WORLD);

            auto H_tmp = Hamiltonian();
            Evolution::Parallel_QME(init_state, H_tmp, time_vec, is_full_rho);
#endif
// ---------------------------------- DELETE --------------------------------------
// --------------------------------------------------------------------------------
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
            Matrix<COMPLEX> B(C_STYLE, n, m);

            if (bcast_data[2] == MPI_Datatype_ID::DOUBLE_COMPLEX) {
                MPI_Bcast(B.data(), n * m, MPI_DOUBLE_COMPLEX, mpi::ROOT_ID, MPI_COMM_WORLD);
                Matrix<COMPLEX> tmp;
                Dim_Multiply(tmp, B, tmp);
            } else {
                MPI_Abort(MPI_COMM_WORLD, 5);
            }

// ---------------------------------------------------------------------------------
// --------------------------------- DELETE -------------------------------------
#ifdef ENABLE_CLUSTER 

// ----------------------------- P_GEMM -----------------------------------------
        } else if (command == COMMAND::P_GEMM_MULTIPLY) {
            int datatype;
            MPI_Bcast(&datatype, 1, MPI_INT, mpi::ROOT_ID, MPI_COMM_WORLD);

            if (datatype == MPI_Datatype_ID::DOUBLE_COMPLEX) {
                Matrix<COMPLEX> tmp;
                //parallel_zgemm(tmp, tmp, tmp);
            } else if (datatype == MPI_Datatype_ID::DOUBLE) {
                Matrix<double> tmp;
                //parallel_dgemm(tmp, tmp, tmp);
            } else {
                MPI_Abort(MPI_COMM_WORLD, 6);
            }
#endif

// -------------------------- EXIT_FROM_FUNC ---------------------------------
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

    Matrix<COMPLEX> localA(C_STYLE, block_size, block_size);
    Matrix<COMPLEX> localB(C_STYLE, block_size, block_size);
    Matrix<COMPLEX> localC(C_STYLE, block_size, block_size, COMPLEX(0));

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

    Matrix<double> localA(C_STYLE, block_size, block_size);
    Matrix<double> localB(C_STYLE, block_size, block_size);
    Matrix<double> localC(C_STYLE, block_size, block_size, 0.0);

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

    Matrix<int> localA(C_STYLE, block_size, block_size);
    Matrix<int> localB(C_STYLE, block_size, block_size);
    Matrix<int> localC(C_STYLE, block_size, block_size, 0);

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

        Matrix<COMPLEX> localA(C_STYLE, rows_map[rank], k);
        Matrix<COMPLEX> localC(C_STYLE, rows_map[rank], m);

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

// ##################################### BLACS, PBLAS ###########################################

#ifdef ENABLE_CLUSTER
    
// UNUSED
MPI_Datatype Create_Block_Type_double (ILP_TYPE N, ILP_TYPE M, ILP_TYPE NB, ILP_TYPE MB) {
    MPI_Datatype tmp_type, block_type;
    int starts[2] = {0, 0};
    int global_size[2] = {N, M};
    int local_size[2] = {NB, MB};

    MPI_Type_create_subarray(2, global_size, local_size, starts, MPI_ORDER_C, MPI_DOUBLE, &tmp_type);
    MPI_Type_commit(&block_type);

    return block_type;
}

namespace {
    void my_dgesd2d(ILP_TYPE N, ILP_TYPE M, ILP_TYPE row_index, ILP_TYPE col_index, const Matrix<double>& A, ILP_TYPE send_id) {
        auto sub = A.submatrix(N, M, row_index, col_index);
        //sub.show();
        //auto block_type = Create_Block_Type_double(bcast_data[0], bcast_data[1], bcast_data[2], bcast_data[3]);
        //MPI_Send(A.data() + A.index(row_index, col_index), 1, block_type, send_id, 0, MPI_COMM_WORLD);
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

    void my_zgesd2d(ILP_TYPE N, ILP_TYPE M, ILP_TYPE row_index, ILP_TYPE col_index, const Matrix<COMPLEX>& A, ILP_TYPE send_id) {
        auto sub = A.submatrix(N, M, row_index, col_index);
        //sub.show();
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
    void ScatterBLACSMatrix_double(const Matrix<double>& A, ILP_TYPE NA,
                                    ILP_TYPE MA, Matrix<double>& localA,
                                    ILP_TYPE NB_A, ILP_TYPE MB_A, ILP_TYPE nrows_A,
                                    ILP_TYPE ncols_A, ILP_TYPE myrow, ILP_TYPE mycol,
                                    ILP_TYPE proc_rows, ILP_TYPE proc_cols,
                                    ILP_TYPE rank, ILP_TYPE* p_ctxt) {

        auto A_tmp = A;
        ILP_TYPE iZERO = 0;
        ILP_TYPE sendr = 0, sendc = 0, recvr = 0, recvc = 0;
        for (ILP_TYPE r = 0; r < NA; r += NB_A, sendr = (sendr + 1) % proc_rows) {
            sendc = 0;
            // Number of rows to be sent
            // Is this the last row block?
            ILP_TYPE nr = NB_A;
            if (NA - r < NB_A)
                nr = NA - r;
    
            for (ILP_TYPE c = 0; c < MA; c += MB_A, sendc = (sendc + 1) % proc_cols) {
                // Number of cols to be sent
                // Is this the last col block?
                ILP_TYPE nc = MB_A;
                if (MA - c < MB_A)
                    nc = MA - c;
    
                if (rank == mpi::ROOT_ID) {
                    // Send a nr-by-nc submatrix to process (sendr, sendc)
                    ILP_TYPE send_id = blacs_pnum(p_ctxt, &sendr, &sendc);
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
    void ScatterBLACSMatrix_COMPLEX(const Matrix<COMPLEX>& A, ILP_TYPE NA,
                                    ILP_TYPE MA, Matrix<COMPLEX>& localA,
                                    ILP_TYPE NB_A, ILP_TYPE MB_A, ILP_TYPE nrows_A,
                                    ILP_TYPE ncols_A, ILP_TYPE myrow, ILP_TYPE mycol,
                                    ILP_TYPE proc_rows, ILP_TYPE proc_cols,
                                    ILP_TYPE rank, ILP_TYPE* p_ctxt) {

        ILP_TYPE sendr = 0, sendc = 0, recvr = 0, recvc = 0;
        for (ILP_TYPE r = 0; r < NA; r += NB_A, sendr = (sendr + 1) % proc_rows) {
            sendc = 0;
            // Number of rows to be sent
            // Is this the last row block?
            ILP_TYPE nr = NB_A;
            if (NA - r < NB_A)
                nr = NA - r;
    
            for (ILP_TYPE c = 0; c < MA; c += MB_A, sendc = (sendc + 1) % proc_cols) {
                // Number of cols to be sent
                // Is this the last col block?
                ILP_TYPE nc = MB_A;
                if (MA - c < MB_A)
                    nc = MA - c;
    
                if (rank == mpi::ROOT_ID) {
                    // Send a nr-by-nc submatrix to process (sendr, sendc)
                    ILP_TYPE send_id = blacs_pnum(p_ctxt, &sendr, &sendc);
                    //std::cout << "Send to " << send_id << " : " << sendr << " " << sendc << std::endl;
                    my_zgesd2d(nr, nc, r, c, A, send_id);
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

ILP_TYPE mpi::indxl2g(ILP_TYPE n, ILP_TYPE NB, ILP_TYPE myindx, ILP_TYPE RSRC, ILP_TYPE dim_size) {
    ILP_TYPE n_new = n + 1;
    return indxl2g_(&n_new, &NB, &myindx, &RSRC, &dim_size) - 1;
}

ILP_TYPE mpi::indxg2p(ILP_TYPE n, ILP_TYPE NB, ILP_TYPE myindx, ILP_TYPE RSRC, ILP_TYPE dim_size) {
    ILP_TYPE n_new = n + 1;
    return indxg2p_(&n_new, &NB, &myindx, &RSRC, &dim_size) - 1;
}

ILP_TYPE mpi::indxg2l(ILP_TYPE n, ILP_TYPE NB, ILP_TYPE myindx, ILP_TYPE RSRC, ILP_TYPE dim_size) {
    ILP_TYPE n_new = n + 1;
    return indxg2l_(&n_new, &NB, &myindx, &RSRC, &dim_size) - 1;
}

std::vector<ILP_TYPE> mpi::descinit(ILP_TYPE n, ILP_TYPE m, ILP_TYPE NB,
                                    ILP_TYPE MB, ILP_TYPE rsrc, ILP_TYPE csrc,
                                    ILP_TYPE ctxt, ILP_TYPE LLD, ILP_TYPE info) {
    std::vector<ILP_TYPE> desc(9);
    descinit_(desc.data(), &n, &m, &NB, &MB, &rsrc, &csrc, &ctxt, &LLD, &info);

    return desc;
}

void mpi::blacs_gridinfo(const ILP_TYPE& ctxt, ILP_TYPE& proc_rows, ILP_TYPE& proc_cols, ILP_TYPE& myrow, ILP_TYPE& mycol) {
    ::blacs_gridinfo(&ctxt, &proc_rows, &proc_cols, &myrow, &mycol);
}

ILP_TYPE mpi::numroc(ILP_TYPE N, ILP_TYPE NB, ILP_TYPE myindex, ILP_TYPE ZERO, ILP_TYPE size) {
    return ::numroc_(&N, &NB, &myindex, &ZERO, &size);
}

void mpi::blacs_gridexit(ILP_TYPE& ctxt) {
    ::blacs_gridexit(&ctxt);
}

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

void mpi::init_grid(ILP_TYPE& ctxt) {
    ILP_TYPE iZERO = 0;
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ILP_TYPE myid, numproc, myrow, mycol;
    char order = 'R';
    ILP_TYPE proc_rows = std::sqrt(world_size), proc_cols = world_size / proc_rows;
    //std::cout << rank << " Here1\n";
    blacs_pinfo(&myid, &numproc);
    ILP_TYPE iMINUS = -1;
    blacs_get(&iMINUS, &iZERO, &ctxt);
    //std::cout << rank << " Here3\n";
    blacs_gridinit(&ctxt, &order, &proc_rows, &proc_cols);
}

template<>
Matrix<double> mpi::scatter_blacs_matrix<double>(const Matrix<double>& A, ILP_TYPE& N, ILP_TYPE& M,
                                      ILP_TYPE& NB, ILP_TYPE& MB, ILP_TYPE& nrows,
                                      ILP_TYPE& ncols, ILP_TYPE& ctxt, ILP_TYPE root_id, MPI_Comm comm) {
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
        if (N - r < NB)
            nr = N - r;

        for (ILP_TYPE c = 0; c < M; c += MB, sendc = (sendc + 1) % proc_cols) {
            ILP_TYPE nc = MB;
            if (M - c < MB)
                nc = M - c;

            if (rank == root_id) {
                ILP_TYPE send_id = blacs_pnum(&ctxt, &sendr, &sendc);
                my_dgesd2d(nr, nc, r, c, A, send_id);
            }

            if (myrow == sendr && mycol == sendc) {
                if (is_c_style) {
                    my_dgerv2d(nr, nc, ncols * recvr + recvc, localA, localA_LD, root_id);
                } else {
                    my_dgerv2d(nr, nc, nrows * recvc + recvr, localA, localA_LD, root_id);
                    //dgerv2d(&ctxt, &nr, &nc, localA.data() + localA.index(recvr, recvc), &localA_LD, &root_row, &root_col);
                }
                recvc = (recvc + nc) % ncols;
            }
        }

        if (myrow == sendr)
            recvr = (recvr + nr) % nrows;
    }

    return localA;
}

template<>
Matrix<COMPLEX> mpi::scatter_blacs_matrix<COMPLEX>(const Matrix<COMPLEX>& A, ILP_TYPE& N, ILP_TYPE& M,
                                      ILP_TYPE& NB, ILP_TYPE& MB, ILP_TYPE& nrows,
                                      ILP_TYPE& ncols, ILP_TYPE& ctxt, ILP_TYPE root_id, MPI_Comm comm) {
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
        if (N - r < NB)
            nr = N - r;

        for (ILP_TYPE c = 0; c < M; c += MB, sendc = (sendc + 1) % proc_cols) {
            ILP_TYPE nc = MB;
            if (M - c < MB)
                nc = M - c;

            if (rank == root_id) {
                ILP_TYPE send_id = blacs_pnum(&ctxt, &sendr, &sendc);
                my_zgesd2d(nr, nc, r, c, A, send_id);
            }

            if (myrow == sendr && mycol == sendc) {
                if (is_c_style) {
                    my_zgerv2d(nr, nc, ncols * recvr + recvc, localA, localA_LD, root_id);
                } else {
                    my_zgerv2d(nr, nc, nrows * recvc + recvr, localA, localA_LD, root_id);
                    //zgerv2d(&ctxt, &nr, &nc, localA.data() + localA.index(recvr, recvc), &localA_LD, &root_row, &root_col);
                }
                recvc = (recvc + nc) % ncols;
            }
        }

        if (myrow == sendr)
            recvr = (recvr + nr) % nrows;
    }

    return localA;
}

template<>
void mpi::gather_blacs_matrix<double>(const Matrix<double>& localC, Matrix<double>& C, 
                                      ILP_TYPE N, ILP_TYPE M,
                                      ILP_TYPE NB, ILP_TYPE MB, ILP_TYPE nrows,
                                      ILP_TYPE ncols, ILP_TYPE ctxt, ILP_TYPE root_id) {
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
        if (N - r < NB)
            nr = N - r;

        for (ILP_TYPE c = 0; c < M; c += MB, sendc = (sendc + 1) % proc_cols) {
            ILP_TYPE nc = MB;
            if (M - c < MB)
                nc = M - c;

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
                    //dgerv2d(&ctxt, &nr, &nc, localA.data() + localA.index(recvr, recvc), &localA_LD, &root_row, &root_col);
                }
            }
        }

        if (myrow == sendr)
            recvr = (recvr + nr) % nrows;
    }
}

template<>
void mpi::gather_blacs_matrix<COMPLEX>(const Matrix<COMPLEX>& localC, Matrix<COMPLEX>& C, 
                                      ILP_TYPE N, ILP_TYPE M,
                                      ILP_TYPE NB, ILP_TYPE MB, ILP_TYPE nrows,
                                      ILP_TYPE ncols, ILP_TYPE ctxt, ILP_TYPE root_id) {
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
        if (N - r < NB)
            nr = N - r;

        for (ILP_TYPE c = 0; c < M; c += MB, sendc = (sendc + 1) % proc_cols) {
            ILP_TYPE nc = MB;
            if (M - c < MB)
                nc = M - c;

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
                    //zgerv2d(&ctxt, &nr, &nc, localA.data() + localA.index(recvr, recvc), &localA_LD, &root_row, &root_col);
                }
            }
        }

        if (myrow == sendr)
            recvr = (recvr + nr) % nrows;
    }
}

template<>
std::vector<double> mpi::scatter_blacs_vector<double>(const std::vector<double>& v, ILP_TYPE& N,
                            ILP_TYPE& NB, ILP_TYPE& nrows, ILP_TYPE& ctxt,
                            ILP_TYPE root_id, MPI_Comm comm) {
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
        if (N - r < NB)
            nr = N - r;

        if (rank == root_id) {
            for (ILP_TYPE sendc = 0; sendc < proc_cols; sendc++) {
                ILP_TYPE send_id = blacs_pnum(&ctxt, &sendr, &sendc);
                MPI_Send(v.data() + r, nr, MPI_DOUBLE, send_id, 0, MPI_COMM_WORLD);
            }
        }

        if (myrow == sendr && mycol == sendc) {
            MPI_Recv(local_v.data() + recvr, nr, MPI_DOUBLE, root_id, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (myrow == sendr)
            recvr = (recvr + nr) % nrows;
    }

    return local_v;
}

template<>
std::vector<COMPLEX> mpi::scatter_blacs_vector<COMPLEX>(const std::vector<COMPLEX>& v, ILP_TYPE& N,
                            ILP_TYPE& NB, ILP_TYPE& nrows, ILP_TYPE& ctxt,
                            ILP_TYPE root_id, MPI_Comm comm) {
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
        if (N - r < NB)
            nr = N - r;

        if (rank == root_id) {
            for (ILP_TYPE sendc = 0; sendc < proc_cols; sendc++) {
                ILP_TYPE send_id = blacs_pnum(&ctxt, &sendr, &sendc);
                MPI_Send(v.data() + r, nr, MPI_DOUBLE_COMPLEX, send_id, 0, MPI_COMM_WORLD);
            }
        }

        if (myrow == sendr && mycol == sendc) {
            MPI_Recv(local_v.data() + recvr, nr, MPI_DOUBLE_COMPLEX, root_id, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (myrow == sendr)
            recvr = (recvr + nr) % nrows;
    }

    return local_v;
}

template<>
void mpi::gather_blacs_vector<double>(const std::vector<double>& local_y, std::vector<double>& y, ILP_TYPE N,
                            ILP_TYPE NB, ILP_TYPE nrows, ILP_TYPE ctxt, ILP_TYPE root_id) {
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
        if (N - r < NB)
            nr = N - r;

        if (myrow == sendr && mycol == 0) {
            //my_zgesd2d(nr, nc, recvr, recvc, localC, root_id);
            MPI_Send(local_y.data() + recvr, nr, MPI_DOUBLE, root_id, 0, MPI_COMM_WORLD);
        }

        if (rank == root_id) {
            ILP_TYPE source_id = blacs_pnum(&ctxt, &sendr, &iZERO);
            //std::cout << source_id << " " << r << " " << nr << " " << N << std::endl;
            MPI_Recv(y.data() + r, nr, MPI_DOUBLE, source_id, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    

        if (myrow == sendr)
            recvr = (recvr + nr) % nrows;
    }
}

template<>
void mpi::gather_blacs_vector<COMPLEX>(const std::vector<COMPLEX>& local_y, std::vector<COMPLEX>& y, ILP_TYPE N,
                            ILP_TYPE NB, ILP_TYPE nrows, ILP_TYPE ctxt, ILP_TYPE root_id) {
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
        if (N - r < NB)
            nr = N - r;

        if (myrow == sendr && mycol == 0) {
            //my_zgesd2d(nr, nc, recvr, recvc, localC, root_id);
            MPI_Send(local_y.data() + recvr, nr, MPI_DOUBLE_COMPLEX, root_id, 0, MPI_COMM_WORLD);
        }

        if (rank == root_id) {
            ILP_TYPE source_id = blacs_pnum(&ctxt, &sendr, &iZERO);
            //std::cout << source_id << " " << r << " " << nr << " " << N << std::endl;
            MPI_Recv(y.data() + r, nr, MPI_DOUBLE_COMPLEX, source_id, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        if (myrow == sendr)
            recvr = (recvr + nr) % nrows;
    }
}

template<>
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

template<>
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

void mpi::parallel_dgemm(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C, const std::vector<ILP_TYPE>& desca,
                         const std::vector<ILP_TYPE>& descb, const std::vector<ILP_TYPE>& descc,
                         char op_A, char op_B) {
    //ILP_TYPE iZERO = 0;
    //int rank, world_size;
    //MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    //ILP_TYPE ctxt;
    ILP_TYPE NA, MA, NB, MB, NB_A, MB_A, NB_B, MB_B;
    //ILP_TYPE LLD_A, LLD_B, LLD_C;
    //ILP_TYPE nrows_A, nrows_B, nrows_C, ncols_A, ncols_B, ncols_C;
    //ILP_TYPE proc_rows, proc_cols, myrow, mycol;
    //Matrix<double> localA;
    //Matrix<double> localB;
    //Matrix<double> localC;
    /*
    if (!is_distributed) {
        init_grid(ctxt);
        blacs_gridinfo(ctxt, proc_rows, proc_cols, myrow, mycol);
        //std::cout << "HERE\n";
        localA = mpi::scatter_blacs_matrix<double>(A, NA, MA, NB_A, MB_A, nrows_A, ncols_A, ctxt, mpi::ROOT_ID);
        localB = mpi::scatter_blacs_matrix<double>(B, NB, MB, NB_B, MB_B, nrows_B, ncols_B, ctxt, mpi::ROOT_ID);

        nrows_C = numroc_(&NA, &NB_A, &myrow, &iZERO, &proc_rows);
        ncols_C = numroc_(&MB, &MB_B, &mycol, &iZERO, &proc_cols);

        localC = Matrix<double>(FORTRAN_STYLE, nrows_C, ncols_C);

        LLD_A = nrows_A;
        LLD_B = nrows_B;
        LLD_C = nrows_C;

        ILP_TYPE rsrc = 0, csrc = 0, info;
        desca = descinit(NA, MA, NB_A, MB_A, rsrc, csrc, ctxt, LLD_A, info);
        if (info != 0) std::cout << "ERROR OF descinit__A: " << rank << " " << info << std::endl;
        descb = descinit(NB, MB, NB_B, MB_B, rsrc, csrc, ctxt, LLD_B, info);
        //descinit_(descb, &NB, &MB, &NB_B, &MB_B, &rsrc, &csrc, &ctxt, &LLD_B, &info);
        if (info != 0) std::cout << "ERROR OF descinit__B: " << rank << " " << info << std::endl;
        descc = descinit(NA, MB, NB_A, MB_B, rsrc, csrc, ctxt, LLD_C, info);
        //descinit_(descc, &NA, &MB, &NB_A, &MB_B, &rsrc, &csrc, &ctxt, &LLD_C, &info);
        if (info != 0) std::cout << "ERROR OF descinit__C: " << rank << " " << info << std::endl;

        
        //mpi::print_distributed_matrix<double>(localA, "A", MPI_COMM_WORLD);
        //mpi::print_distributed_matrix<double>(localB, "B", MPI_COMM_WORLD);
    */
    //} else {
    NA = desca[2];
    MA = desca[3];
    NB = descb[2];
    MB = descb[3];
    //}

    /*
    bool return_to_c_style = false;

    if (localA.is_c_style()) {
        return_to_c_style = true;
        localA.to_fortran_style();
    }

    if (!is_distributed andlocalB.is_c_style()) {
        localB.to_fortran_style();
    }
    */
    char N = 'N';
    ILP_TYPE iONE = 1;
    double alpha = 1.0;
    double betta = 0;

    //if (!is_distributed) {
        //auto begin = std::chrono::steady_clock::now();
    //    pdgemm_(&op_A, &op_B, &NA, &MB, &MA, &alpha, localA.data(), &iONE, &iONE, desca.data(),
    //                                    localB.data(), &iONE, &iONE, descb.data(),
    //                                    &betta, localC.data(), &iONE, &iONE, descc.data());
        //auto end = std::chrono::steady_clock::now();
        //if (rank == mpi::ROOT_ID) std::cout << "PDGEMM: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
    //} else {
        //auto begin = std::chrono::steady_clock::now();
    pdgemm_(&op_A, &op_B, &NA, &MB, &MA, &alpha, A.data(), &iONE, &iONE, desca.data(),
                                    B.data(), &iONE, &iONE, descb.data(),
                                    &betta, C.data(), &iONE, &iONE, descc.data());
        //auto end = std::chrono::steady_clock::now();
        //if (rank == mpi::ROOT_ID) std::cout << "PDGEMM: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;  
    //}

    //if (return_to_c_style) {
    //    localC.to_c_style();
    //}

    //print_distributed_matrix<double>(localC, "C", MPI_COMM_WORLD);

    /*
    if (!is_distributed) {
        if (rank == mpi::ROOT_ID) C = Matrix<double>(localC.get_matrix_style(), NA, MB);
        gather_blacs_matrix<double>(localC, C, NA, MB, NB_A, MB_B, nrows_C, ncols_C, ctxt, mpi::ROOT_ID);
        blacs_gridexit(ctxt);
    }
    */
    //blacs_exit(&iZERO);
}

void mpi::parallel_zgemm(const Matrix<COMPLEX>& A, const Matrix<COMPLEX>& B, Matrix<COMPLEX>& C,
                        const std::vector<ILP_TYPE>& desca,
                         const std::vector<ILP_TYPE>& descb, const std::vector<ILP_TYPE>& descc,
                         char op_A, char op_B) {
    ILP_TYPE iZERO = 0;
    //int rank, world_size;
    //MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    //std::cout << rank << " HERE\n";

    //ILP_TYPE ctxt;
    ILP_TYPE NA, MA, NB, MB, NB_A, MB_A, NB_B, MB_B;
    //ILP_TYPE LLD_A, LLD_B, LLD_C;
    //ILP_TYPE nrows_A, nrows_B, nrows_C, ncols_A, ncols_B, ncols_C;
    //ILP_TYPE proc_rows, proc_cols, myrow, mycol;
    //Matrix<COMPLEX> localA;
    //Matrix<COMPLEX> localB;
    //Matrix<COMPLEX> localC;
    /*
    if (!is_distributed) {
        init_grid(ctxt);
        blacs_gridinfo(ctxt, proc_rows, proc_cols, myrow, mycol);
        //std::cout << "HERE\n";
        localA = mpi::scatter_blacs_matrix<COMPLEX>(A, NA, MA, NB_A, MB_A, nrows_A, ncols_A, ctxt, mpi::ROOT_ID);
        localB = mpi::scatter_blacs_matrix<COMPLEX>(B, NB, MB, NB_B, MB_B, nrows_B, ncols_B, ctxt, mpi::ROOT_ID);

        nrows_C = numroc_(&NA, &NB_A, &myrow, &iZERO, &proc_rows);
        ncols_C = numroc_(&MB, &MB_B, &mycol, &iZERO, &proc_cols);

        localC = Matrix<COMPLEX>(FORTRAN_STYLE, nrows_C, ncols_C);

        LLD_A = nrows_A;
        LLD_B = nrows_B;
        LLD_C = nrows_C;

        ILP_TYPE rsrc = 0, csrc = 0, info;
        desca = descinit(NA, MA, NB_A, MB_A, rsrc, csrc, ctxt, LLD_A, info);
        if (info != 0) std::cout << "ERROR OF descinit__A: " << rank << " " << info << std::endl;
        descb = descinit(NB, MB, NB_B, MB_B, rsrc, csrc, ctxt, LLD_B, info);
        //descinit_(descb, &NB, &MB, &NB_B, &MB_B, &rsrc, &csrc, &ctxt, &LLD_B, &info);
        if (info != 0) std::cout << "ERROR OF descinit__B: " << rank << " " << info << std::endl;
        descc = descinit(NA, MB, NB_A, MB_B, rsrc, csrc, ctxt, LLD_C, info);
        //descinit_(descc, &NA, &MB, &NB_A, &MB_B, &rsrc, &csrc, &ctxt, &LLD_C, &info);
        if (info != 0) std::cout << "ERROR OF descinit__C: " << rank << " " << info << std::endl;

        
        //mpi::print_distributed_matrix<COMPLEX>(localA, "A", MPI_COMM_WORLD);
        //mpi::print_distributed_matrix<COMPLEX>(localB, "B", MPI_COMM_WORLD);
    */
    //} else {
    NA = desca[2];
    MA = desca[3];
    NB = descb[2];
    MB = descb[3];
    //}

    /*
    bool return_to_c_style = false;

    if (!is_distributed and localA.is_c_style()) {
        return_to_c_style = true;
        localA.to_fortran_style();
    }

    if (!is_distributed and localB.is_c_style()) {
        localB.to_fortran_style();
    }
    */


    ILP_TYPE iONE = 1;
    COMPLEX alpha(1.0, 0);
    COMPLEX betta(0, 0);

    //if (!is_distributed) {
    //    auto begin = std::chrono::steady_clock::now();
    //    pzgemm_(&op_A, &op_B, &NA, &MB, &MA, &alpha, localA.data(), &iONE, &iONE, desca.data(),
    //                                    localB.data(), &iONE, &iONE, descb.data(),
    //                                    &betta, localC.data(), &iONE, &iONE, descc.data());
    //    auto end = std::chrono::steady_clock::now();
        //if (rank == mpi::ROOT_ID) std::cout << "PZGEMM: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
    //} else {
    //    auto begin = std::chrono::steady_clock::now();
    pzgemm_(&op_A, &op_B, &NA, &MB, &MA, &alpha, A.data(), &iONE, &iONE, desca.data(),
                                    B.data(), &iONE, &iONE, descb.data(),
                                    &betta, C.data(), &iONE, &iONE, descc.data());
    //    auto end = std::chrono::steady_clock::now();
        //if (rank == mpi::ROOT_ID) std::cout << "PZGEMM: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;  
    //}

    //if (!is_distributed and return_to_c_style) {
    //    localC.to_c_style();
    //}

    //print_distributed_matrix<COMPLEX>(localC, "C", MPI_COMM_WORLD);

    /*
    if (!is_distributed) {
        if (rank == mpi::ROOT_ID) C = Matrix<COMPLEX>(localC.get_matrix_style(), NA, MB);
        gather_blacs_matrix<COMPLEX>(localC, C, NA, MB, NB_A, MB_B, nrows_C, ncols_C, ctxt, mpi::ROOT_ID);
        blacs_gridexit(ctxt);
    }
    */
    //blacs_exit(&iZERO);
}

void mpi::parallel_zhemm(char side, const Matrix<COMPLEX>& A, const Matrix<COMPLEX>& B, Matrix<COMPLEX>& C,
                    const std::vector<ILP_TYPE>& desca,
                    const std::vector<ILP_TYPE>& descb, const std::vector<ILP_TYPE>& descc,
                    char op_A, char op_B) {
    ILP_TYPE NA, MA, NB, MB, NB_A, MB_A, NB_B, MB_B;

    NA = desca[2];
    MA = desca[3];
    NB = descb[2];
    MB = descb[3];

    ILP_TYPE iONE = 1;
    COMPLEX alpha(1.0, 0);
    COMPLEX betta(0, 0);
    char uplo = 'U';

    pzhemm_(&side, &uplo, &NA, &MB, &alpha, A.data(), &iONE, &iONE, desca.data(),
                                B.data(), &iONE, &iONE, descb.data(),
                                &betta, C.data(), &iONE, &iONE, descc.data());
}

void mpi::parallel_dgeadd(const Matrix<double>& A, Matrix<double>& C,
                    const std::vector<ILP_TYPE>& desca,
                    const std::vector<ILP_TYPE>& descc, double alpha, double betta,
                    char op_A) {
    ILP_TYPE iONE = 1;
    ILP_TYPE NA = desca[2];
    ILP_TYPE MA = desca[3];

    pdgeadd(&op_A, &NA, &MA, &alpha, A.data(), &iONE, &iONE, desca.data(), &betta, C.data(), &iONE, &iONE, descc.data());
}

void mpi::parallel_zgeadd(const Matrix<COMPLEX>& A, Matrix<COMPLEX>& C,
                    const std::vector<ILP_TYPE>& desca,
                    const std::vector<ILP_TYPE>& descc,
                    COMPLEX alpha, COMPLEX betta,
                    char op_A) {
    ILP_TYPE iONE = 1;
    ILP_TYPE NA = desca[2];
    ILP_TYPE MA = desca[3];

    pzgeadd(&op_A, &NA, &MA, &alpha, A.data(), &iONE, &iONE, desca.data(), &betta, C.data(), &iONE, &iONE, descc.data());
}

// only for distributed version
void mpi::parallel_dgemv(const Matrix<double>& A, const std::vector<double>& x, std::vector<double>& y,
                    const std::vector<ILP_TYPE>& desca,
                    const std::vector<ILP_TYPE>& descx, const std::vector<ILP_TYPE>& descy,
                    char op_A) {
    ILP_TYPE N_A, M_A;
    N_A = desca[2];
    M_A = desca[3];

    int iONE = 1;
    double alpha = 1.0;
    double betta = 0;

    pdgemv_(&op_A, &N_A, &M_A, &alpha, A.data(), &iONE, &iONE, desca.data(),
                                   x.data(), &iONE, &iONE, descx.data(), &iONE,
                                   &betta, y.data(), &iONE, &iONE, descy.data(), &iONE);
}

// only for distributed version
void mpi::parallel_zgemv(const Matrix<COMPLEX>& A, const std::vector<COMPLEX>& x, std::vector<COMPLEX>& y,
                    const std::vector<ILP_TYPE>& desca,
                    const std::vector<ILP_TYPE>& descx, const std::vector<ILP_TYPE>& descy,
                    char op_A) {
    ILP_TYPE N_A, M_A;
    N_A = desca[2];
    M_A = desca[3];

    int iONE = 1;
    COMPLEX alpha (1.0, 0);
    COMPLEX betta(0, 0);

    pzgemv_(&op_A, &N_A, &M_A, &alpha, A.data(), &iONE, &iONE, desca.data(),
                                   x.data(), &iONE, &iONE, descx.data(), &iONE,
                                   &betta, y.data(), &iONE, &iONE, descy.data(), &iONE);
}

#endif // ENABLE_CLUSTER

} // namespace QComputations
#endif // ENABLE_MPI


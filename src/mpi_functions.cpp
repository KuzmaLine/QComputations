#ifdef ENABLE_MPI

#include "mpi_functions.hpp"
#include "functions.hpp"
#include "hamiltonian.hpp"
#include "dynamic.hpp"

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

#endif // ENABLE_MPI
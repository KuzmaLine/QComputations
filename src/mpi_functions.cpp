#ifdef ENABLE_MPI

#include "mpi_functions.hpp"
#include "functions.hpp"
#include "hamiltonian.hpp"

void mpi::make_command(int command) {
    MPI_Bcast(&command, 1, MPI_INT, ROOT_ID, MPI_COMM_WORLD);
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

void mpi::run_mpi_slaves() {
    while(true) {
        int command;
        MPI_Bcast(&command, 1, MPI_INT, ROOT_ID, MPI_COMM_WORLD);

        if (command == COMMAND::STOP) {
            break;
        }

        if (command == COMMAND::GENERATE_H) {
            State tmp = bcast_state(State());
            //std::cout << tmp.to_string() << std::endl;
            H_TCH H(tmp);
        } else if (command = COMMAND::SCHRODINGER) {
        } else {
            std::cerr << "UNKNOWN/UNAVAILABLE COMMAND - " << command << std::endl;
        }
    }
}

void mpi::stop_mpi_slaves() {
    int command = COMMAND::STOP;
    make_command(command);
    MPI_Finalize();
}

#endif // ENABLE_MPI
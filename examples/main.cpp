#include "../src/QComputations.hpp"

int main(int argc, char** argv) {
    using namespace QComputations;
    int n = 8000;

    std::vector<size_t> grid_config = {1, 1};
    //State state("|0;00>");
    State state(grid_config);
    state.set_gamma(0.002);
    state.set_leak_for_cavity(0, 0.005);
    state.set_gain_for_cavity(0, 0.002);
    state.set_max_N(2);
    state.set_min_N(0);

#ifdef ENABLE_MPI
    int rank, world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    //std::cout << "WORLD SIZE - " << world_size << std::endl;
    if (world_size == 1) {
        std::cerr << "Should have at least 2 processes\n";

        MPI_Finalize();
        return 1;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::map<int, std::vector<mpi::MPI_Data>> data;
    data[COMMAND::GENERATE_H].resize(1);
    data[COMMAND::GENERATE_H_FUNC].resize(1);

    data[COMMAND::GENERATE_H][0].state = state;
    data[COMMAND::GENERATE_H_FUNC][0].n = n;

    if (rank != 0) {
        mpi::run_mpi_slaves(data);
        MPI_Finalize();
        return 0;
    }
#endif

    H_TCH H(state);

#ifdef ENABLE_MPI
    mpi::stop_mpi_slaves();
#endif

    return 0;
}

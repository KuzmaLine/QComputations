#include <iostream>
#include <chrono>
#include <complex>
#include "/home/kuzmaline/Quantum/diploma/src/QComputations_MPI_CLUSTER.hpp"

int main(int argc, char** argv) {
    using namespace QComputations;

    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //if (rank == 0) QConfig::instance().show();

    std::vector<size_t> grid_config = {3};
    State grid(grid_config);
    grid.set_max_N(2);
    grid.set_min_N(2);
    //std::cout << grid.to_string() << std::endl;

    int ctxt;
    mpi::init_grid(ctxt);

    BLOCKED_H_TC H(ctxt, grid);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) show_basis(H.get_basis());
    MPI_Barrier(MPI_COMM_WORLD);

    H.show();

    H.print_distributed("H_TCH");

    MPI_Finalize();
    return 0;
}
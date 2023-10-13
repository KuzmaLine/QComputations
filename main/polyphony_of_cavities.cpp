#include "/home/kuzmaline/Quantum/diploma/src/QComputations_CPU_CLUSTER.hpp"
#include <iostream>

int main(int argc, char** argv) {
    using namespace QComputations;
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::vector<size_t> grid_config = {1, 1};
    State grid(grid_config);
    grid.set_max_N(2);
    grid.set_min_N(2);
    grid.set_gamma(COMPLEX(0.2,0));

    int ctxt;
    mpi::init_grid(ctxt);
    BLOCKED_H_TCH H(ctxt, grid);

    //if (rank == 0) { show_basis(H.get_basis()); }

    //H.show();

    MPI_Finalize();
    return 0;
}
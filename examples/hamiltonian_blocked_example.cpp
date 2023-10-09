#include <iostream>
#include <chrono>
#include <complex>
#include "/home/kuzmaline/Quantum/diploma/src/QComputations_CPU_CLUSTER_NO_PLOTS.hpp"

int main(int argc, char** argv) {
    using namespace QComputations;
    //QConfig::instance().set_h(0.1); // - изменение постояннной планка на 0.1
    //std::cout << QConfig::instance().h() << std::endl;

    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) QConfig::instance().show(); // - вывод всех параметров внутри конфига

    std::vector<size_t> grid_config = {2};
    State grid(grid_config);
    grid.set_max_N(3);
    grid.set_min_N(3);

    int ctxt;
    mpi::init_grid(ctxt);

    BLOCKED_H_TCH H(ctxt, grid);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) show_basis(H.get_basis());
    MPI_Barrier(MPI_COMM_WORLD);

    //H.show();

    //H.print_distributed("H_TCH");

    MPI_Finalize();
    return 0;
}

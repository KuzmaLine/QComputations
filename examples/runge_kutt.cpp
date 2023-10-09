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

    BLOCKED_H_TC H(ctxt, grid);

    H_TC H_single(grid);

    if (rank == 0) show_basis(H.get_basis());

    H.show();

    if (rank == 0) show_basis(H_single.get_basis());

    if (rank == 0) H_single.show();

    std::vector<COMPLEX> init_state(H.size(), 0);
    init_state[State("|3;00>").get_index(H.get_basis())] = COMPLEX(1, 0);

    if (rank == 0) std::cout << init_state << std::endl;
    //H.print_distributed("H_TCH");

    MPI_Finalize();
    return 0;
}

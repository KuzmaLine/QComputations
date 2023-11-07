#include "/home/kuzmaline/Quantum/diploma/src/QComputations_CPU_CLUSTER.hpp"
#include <iostream>
#include <complex>

using COMPLEX = std::complex<double>;

int main(int argc, char** argv) {
    using namespace QComputations;
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    QConfig::instance().set_width(30);

    size_t grid_size = 1;
    size_t atoms_num = 3;
    std::vector<size_t> grid_config;

    for (size_t i = 0; i < grid_size; i++) {
        if (i == 0 or i == grid_size - 1) {
            grid_config.emplace_back(atoms_num);
        } else {
            grid_config.emplace_back(0);
        }
    }

    State grid(grid_config);
    //grid.set_qubit(0, 0, 1);
    //grid.set_qubit(0, 1, 1);
    grid.set_n(1);

    int ctxt;
    mpi::init_grid(ctxt);
    BLOCKED_H_TCH H(ctxt, grid);

    if (rank == 0) { show_basis(H.get_basis()); }

    H.show();

    std::vector<COMPLEX> init_state(H.size(), 0);
    init_state[grid.get_index(H.get_basis())] = COMPLEX(1, 0);

    auto time_vec = linspace(0, 1000, 2000);

    std::cout << "0\u2082\u2083\u29FD" << std::endl;

    auto probs = Evolution::schrodinger(init_state, H, time_vec);
    //auto probs = Evolution::quantum_master_equation(init_state, H, time_vec);

    if (rank == 0) {
        matplotlib::make_figure(1920, 1080);
    }

    matplotlib::probs_to_plot(probs, time_vec, H.get_basis());
    if (rank == 0) {
        matplotlib::grid();
        matplotlib::show();
    }

    MPI_Finalize();
    return 0;
}

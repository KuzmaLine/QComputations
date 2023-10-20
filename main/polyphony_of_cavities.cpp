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
    QConfig::instance().set_width(20);
    QConfig::instance().set_g(0.001);

    std::vector<size_t> grid_config = {1, 0, 1};
    State grid(grid_config);
    //grid.reshape(2, 2, 1);
    grid.set_max_N(3);
    grid.set_min_N(3);
    grid.set_waveguide(0.002 / sqrt(10), 1);
    grid.set_qubit(0, 0, 1);
    //grid.set_n(1, 0);
    //grid.set_qubit(1, 0, 1);

    int ctxt;
    mpi::init_grid(ctxt);
    BLOCKED_H_TCH H(ctxt, grid);

    if (rank == 0) { show_basis(H.get_basis()); }

    H.show();

    std::vector<COMPLEX> init_state(H.size(), 0);
    init_state[grid.get_index(H.get_basis())] = COMPLEX(1, 0);

    auto time_vec = linspace(0, 32000, 64000);

    auto probs = Evolution::quantum_master_equation(init_state, H, time_vec);

    if (rank == 0) {
        matplotlib::make_figure(1920, 1080);
    }

    matplotlib::probs_to_plot(probs, time_vec, H.get_basis());
    //matplotlib::probs_in_cavity_to_plot(probs, time_vec, H.get_basis(), 0);
    if (rank == 0) {
        matplotlib::grid();
        matplotlib::show(false);
    }

    if (rank == 0) {
        matplotlib::make_figure(1920, 1080);
    }

    //matplotlib::probs_to_plot(probs, time_vec, H.get_basis());
    matplotlib::probs_in_cavity_to_plot(probs, time_vec, H.get_basis(), 0);
    if (rank == 0) {
        matplotlib::grid();
        matplotlib::show(false);
    }

    if (rank == 0) {
        matplotlib::make_figure(1920, 1080);
    }

    //matplotlib::probs_to_plot(probs, time_vec, H.get_basis());
    matplotlib::probs_in_cavity_to_plot(probs, time_vec, H.get_basis(), grid_config.size() - 1);
    if (rank == 0) {
        matplotlib::grid();
        matplotlib::show();
    }

    MPI_Finalize();
    return 0;
}
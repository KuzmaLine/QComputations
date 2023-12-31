#include "QComputations_CPU_CLUSTER.hpp"
#include <iostream>
#include <regex>
#include <complex>

using COMPLEX = std::complex<double>;

int main(int argc, char** argv) {
    using namespace QComputations;
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    QConfig::instance().set_width(30);
    double h = QConfig::instance().h();
    double w = QConfig::instance().w(); // E = hw
    QConfig::instance().set_g(0.005); // сила взаимодействия с полем атома

    std::vector<size_t> grid_config = {1};

    State grid(grid_config);
    grid.set_qubit(0, 0, 1);
    // |0>ph|1>at
    //grid.set_n(1);
    grid.set_leak_for_cavity(0, 0.1);
    //grid.set_gain_for_cavity(0, 0.1); // приток фотонов

    int ctxt;
    mpi::init_grid(ctxt);
    BLOCKED_H_JC H(ctxt, grid);

    if (rank == 0) { show_basis(H.get_basis()); }

    H.show();

    std::vector<COMPLEX> init_state(H.size(), 0);
    init_state[grid.get_index(H.get_basis())] = COMPLEX(1, 0);

    auto time_vec = linspace(0, 5000, 5000);

    std::vector<double> gamma_vec = linspace(0.001, 0.1, 100);

    for (size_t i = 0; i < 10; i++) {
        gamma_vec.emplace_back(0.1 + (i + 1) * 0.01);
    }

    auto tau_vec = Evolution::scan_gamma(init_state, H, 0, time_vec, gamma_vec, 0.9);

    if (rank == 0) {
        matplotlib::make_figure(1920, 1080);
        matplotlib::plot(gamma_vec, tau_vec);
        matplotlib::xlabel("gamma");
        matplotlib::ylabel("time");
        matplotlib::grid();
        matplotlib::savefig("leak_third_example.png");
        matplotlib::show();
    }

    MPI_Finalize();
    return 0;
}
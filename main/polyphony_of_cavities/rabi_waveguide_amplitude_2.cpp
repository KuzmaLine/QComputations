#include <iostream>
#include <complex>
#include "QComputations_CPU_CLUSTER_NO_PLOTS.hpp"

int main(int argc, char** argv) {
    using namespace QComputations;
    int world_size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    //QConfig::instance().set_max_photons(2);
    std::vector<size_t> grid_config = {1};
    CHE_State state(grid_config);

    State<CHE_State> init_state(state);
    state.set_n(1, 0);
    //state.set_qudit(1, 1, 0);
    //state.set_qudit(1, 1, 1);
    //state.set_waveguide(0.002, 0);

    init_state.insert(state, 1);

    int ctxt;
    mpi::init_grid(ctxt);

    //std::cout << "HERE\n";
    BLOCKED_H_TCH H(ctxt, init_state);

    if (rank == 0) show_basis(H.get_basis());

    H.show();

    auto time_vec = linspace(0, 1000, 1000);

    //time_vec_to_file("test_time_vec", time_vec, "");

    auto probs = Evolution::quantum_master_equation(init_state.fit_to_basis(H.get_basis()), H, time_vec);

    //basis_to_file("basis_check.csv", H.get_basis());
    make_probs_files(H, probs, time_vec, H.get_basis(), "rabi_waveguide_amplitude_2");  
    //make_plot("test_tch.svg", H, probs, time_vec, H.get_basis(), "test_dir");

    MPI_Finalize();
    return 0;
}

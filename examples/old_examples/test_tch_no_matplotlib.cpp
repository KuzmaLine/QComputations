#include <iostream>
#include <complex>
#include "QComputations_CPU_CLUSTER_NO_PLOTS.hpp"

int main(int argc, char** argv) {
    using namespace QComputations;
    int world_size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    QConfig::instance().set_max_photons(2);
    std::vector<size_t> grid_config = {2};
    CHE_State state(grid_config);

    State<CHE_State> init_state(state);
    //state.set_n(1, 0);
    //state.set_qudit(1, 1, 0);
    //state.set_qudit(1, 1, 1);
    state.set_leak_for_cavity(0, 0.005);
    state.set_gain_for_cavity(0, 0.002);
    //state.set_waveguide(0.002, 0);

    init_state[0] = 1/std::sqrt(3);

    state.set_qudit(1, 1, 0);
    init_state.insert(state, -1/std::sqrt(3));

    state.set_qudit(0, 1, 0);
    state.set_qudit(1, 2, 0);

    init_state.insert(state, 1/std::sqrt(3));

    std::cout << init_state.to_string() << std::endl;

    int ctxt;
    mpi::init_grid(ctxt);

    //std::cout << "HERE\n";
    BLOCKED_H_TCH H(ctxt, init_state);

    if (rank == 0) show_basis(H.get_basis());

    H.show();

    auto time_vec = linspace(0, 2000, 2000);

    //time_vec_to_file("test_time_vec", time_vec, "");

    auto probs = Evolution::quantum_master_equation(init_state.fit_to_basis(H.get_basis()), H, time_vec);

    //basis_to_file("basis_check.csv", H.get_basis());    
    make_plot("test_tch.svg", H, probs, time_vec, H.get_basis(), "test_dir");

    MPI_Finalize();
    return 0;
}

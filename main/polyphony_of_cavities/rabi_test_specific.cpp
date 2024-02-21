#include <iostream>
#include <complex>
#include "QComputations_CPU_CLUSTER_NO_PLOTS.hpp"

int main(int argc, char** argv) {
    using namespace QComputations;
    int world_size, rank;
    size_t atoms_count = 2, steps_count = 100;
    QConfig::instance().set_max_photons(atoms_count);
    double a = 0.001, b = 0.05;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);


    auto amplitude_range = linspace(0.0001, 0.001, 100);
    int ctxt;
    mpi::init_grid(ctxt);

    for (auto amplitude: amplitude_range) {
        std::vector<size_t> grid_config = {atoms_count, atoms_count};
        auto state = CHE_State(grid_config);

        state.set_n(atoms_count, 0);
        state.set_waveguide(amplitude, 0);

        State<CHE_State> init_state_2_cavity(state);

        auto time_vec = linspace(0, 8000, 8000);
        //std::cout << "HERE\n";

        auto H = BLOCKED_H_TCH(ctxt, init_state_2_cavity);

        //if (rank == 0) show_basis(H.get_basis());

        H.show();

        auto probs = Evolution::quantum_master_equation(init_state_2_cavity.fit_to_basis(H.get_basis()), H, time_vec);

        make_probs_files(H, probs, time_vec, H.get_basis(), "test_2_atoms/q=waveguide_" + std::to_string(amplitude));
    }

    MPI_Finalize();
    return 0;
}

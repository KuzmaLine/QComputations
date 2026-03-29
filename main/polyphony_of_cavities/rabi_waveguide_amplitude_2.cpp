#include <iostream>
#include <complex>
#include "QComputations_CPU_CLUSTER_NO_PLOTS.hpp"

// amplitude = 0.0115 = резонанс

int main(int argc, char** argv) {
    using namespace QComputations;
    int world_size, rank;
    size_t atoms_count = 1, steps_count = 900;
    double a = 0.001, b = 0.03;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    //QConfig::instance().set_max_photons(2);
    std::vector<size_t> grid_config = {atoms_count};
    CHE_State state(grid_config);

    state.set_n(atoms_count, 0);
    //state.set_qudit(1, 1, 0);
    //state.set_qudit(1, 1, 1);
    //state.set_waveguide(0.002, 0);

    State<CHE_State> init_state(state);

    int ctxt;
    mpi::init_grid(ctxt);

    //std::cout << "HERE\n";
    H_TCH H(init_state);

    //if (rank == 0) show_basis(H.get_basis());

    //H.show();

    auto time_vec = linspace(0, 4000, 4000);

    //time_vec_to_file("test_time_vec", time_vec, "");

    auto probs = Evolution::quantum_master_equation(init_state.fit_to_basis(H.get_basis()), H, time_vec);

    //basis_to_file("basis_check.csv", H.get_basis());
    if (rank == 0) {
        make_probs_files(H, probs, time_vec, H.get_basis(), "rabi_waveguide_amplitude_2/original", rank);
    }
    //make_plot("test_tch.svg", H, probs, time_vec, H.get_basis(), "test_dir");

    auto amplitude_range = linspace(a, b, steps_count);
    time_vec = linspace(0, 8000, 8000);
    size_t start, count;
    make_rank_map(amplitude_range.size(), rank, world_size, start, count);


    for (size_t i = start; i < count + start; i++) {
        auto amplitude = amplitude_range[i];
        grid_config = {atoms_count, atoms_count};
        state = CHE_State(grid_config);

        state.set_n(1, 0);
        state.set_waveguide(amplitude, 0);

        State<CHE_State> init_state_2_cavity(state);

        //std::cout << "HERE\n";
        H = H_TCH(init_state_2_cavity);

        //if (rank == 0) show_basis(H.get_basis());

        //H.show();

        probs = Evolution::quantum_master_equation(init_state_2_cavity.fit_to_basis(H.get_basis()), H, time_vec);

        make_probs_files(H, probs, time_vec, H.get_basis(), "rabi_waveguide_amplitude_2/amplitude_" + std::to_string(amplitude), rank);
        
        auto p_0 = Evolution::probs_to_cavity_probs(probs, H.get_basis(), 0);
        auto p_1 = Evolution::probs_to_cavity_probs(probs, H.get_basis(), 1);

        make_probs_files(H, p_0.first, time_vec, p_0.second, "rabi_waveguide_amplitude_2/0_amplitude_" + std::to_string(amplitude), rank);
        make_probs_files(H, p_1.first, time_vec, p_1.second, "rabi_waveguide_amplitude_2/1_amplitude_" + std::to_string(amplitude), rank);
    }

    amplitude_range = linspace(QConfig::instance().g(), QConfig::instance().g() + 0.002, 200);
    make_rank_map(amplitude_range.size(), rank, world_size, start, count);

    time_vec = linspace(0, 8000, 12000);

    for (size_t i = start; i < count + start; i++) {
        auto amplitude = amplitude_range[i];
        grid_config = {atoms_count, atoms_count};
        state = CHE_State(grid_config);

        state.set_n(atoms_count, 0);
        state.set_waveguide(amplitude, 0);

        State<CHE_State> init_state_2_cavity(state);
        
        //std::cout << "HERE\n";
        H = H_TCH(init_state_2_cavity);

        //if (rank == 0) show_basis(H.get_basis());

        //H.show();

        probs = Evolution::quantum_master_equation(init_state_2_cavity.fit_to_basis(H.get_basis()), H, time_vec);

        make_probs_files(H, probs, time_vec, H.get_basis(), "rabi_waveguide_amplitude_2/q=waveguide_" + std::to_string(amplitude), rank);

        auto p_0 = Evolution::probs_to_cavity_probs(probs, H.get_basis(), 0);
        auto p_1 = Evolution::probs_to_cavity_probs(probs, H.get_basis(), 1);

        make_probs_files(H, p_0.first, time_vec, p_0.second, "rabi_waveguide_amplitude_2/0_q=waveguide_" + std::to_string(amplitude), rank);
        make_probs_files(H, p_1.first, time_vec, p_1.second, "rabi_waveguide_amplitude_2/1_q=waveguide_" + std::to_string(amplitude), rank);
    }

    MPI_Finalize();
    return 0;
}

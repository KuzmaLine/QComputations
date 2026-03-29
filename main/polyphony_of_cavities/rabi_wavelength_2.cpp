#include <iostream>
#include <complex>
#include "QComputations_CPU_CLUSTER_NO_PLOTS.hpp"

double find_amplitude(double amplitude, size_t path_length, size_t count_paths) {
    return std::pow(amplitude / count_paths, double(1) / path_length);
}

int main(int argc, char** argv) {
    using namespace QComputations;
    int world_size, rank;
    size_t atoms_count = 1;
    size_t target_cavity = 3;
    double amplitude = 0.001;
    double step = 0.25;
    double steps_count = double(2) / 0.25;
    std::string dir = "results/";
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    //QConfig::instance().set_g(0.1);
    QConfig::instance().set_max_photons(atoms_count);
    std::vector<size_t> grid_config = {atoms_count, 0, 0, atoms_count};
    CHE_State state(grid_config);
    state.reshape(2, 2, 1);
    state.set_waveguide(find_amplitude(amplitude, 2, 2) / 20, 2*M_PI);

    //state.set_n(atoms_count, 0);
    //state.set_n(atoms_count, 0);
    for (size_t i = 0; i < atoms_count; i++) {
        state.set_atom(1, i, 0);
    }
    //state.set_atom(1, 0, 0);

    //state.set_qudit(1, 1, 0);
    //state.set_qudit(1, 1, 1);
    //state.set_waveguide(0.002, 0);

    State<CHE_State> init_state(state);

    int ctxt;
    mpi::init_grid(ctxt);

    //std::cout << "HERE\n";
    H_TCH H(init_state);

    if (rank == 0) {
        show_basis(H.get_basis());
        H.show();
        std::cout << "------------------------------------\n";
    }

    //if (rank == 0) show_basis(H.get_basis());

    //H.show();

    auto time_vec = linspace(0, 16000, 16000);

    //time_vec_to_file("test_time_vec", time_vec, "");

    auto probs = Evolution::schrodinger(init_state.fit_to_basis(H.get_basis()), H, time_vec);

    auto p_0 = Evolution::probs_to_cavity_probs(probs, H.get_basis(), 0);
    auto p_1 = Evolution::probs_to_cavity_probs(probs, H.get_basis(), target_cavity);

    make_probs_files(H, p_0.first, time_vec, p_0.second, dir + "rabi_wavelength/0_original", 0);
    make_probs_files(H, p_1.first, time_vec, p_1.second, dir + "rabi_wavelength/1_original", 0);

    for (size_t i = 0; i < steps_count; i++) {
        double coef = step * (i + 1);
        state.set_waveguide(1, 3, find_amplitude(amplitude, 2, 2) / 20, coef * M_PI);

        State<CHE_State> init_state(state);
        H = H_TCH(init_state);

        if (rank == 0) {
            H.show();
            std::cout << coef << " ------------------------------------\n";
        }

        //time_vec_to_file("test_time_vec", time_vec, "");

        probs = Evolution::schrodinger(init_state.fit_to_basis(H.get_basis()), H, time_vec);

        //basis_to_file("basis_check.csv", H.get_basis());
        p_0 = Evolution::probs_to_cavity_probs(probs, H.get_basis(), 0);
        p_1 = Evolution::probs_to_cavity_probs(probs, H.get_basis(), target_cavity);

        make_probs_files(H, p_0.first, time_vec, p_0.second, dir + "rabi_wavelength/0_" + std::to_string(coef) + "PI", 0);
        make_probs_files(H, p_1.first, time_vec, p_1.second, dir + "rabi_wavelength/1_" + std::to_string(coef) + "PI", 0);
    }
    MPI_Finalize();
    return 0;
}

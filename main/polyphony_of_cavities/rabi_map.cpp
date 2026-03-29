#include <iostream>
#include <complex>
#include "QComputations_CPU_CLUSTER_NO_PLOTS.hpp"

double find_amplitude(double amplitude, size_t path_length, size_t count_paths) {
    return std::pow(amplitude / count_paths, double(1) / path_length);
}

std::string make_filename(const std::vector<size_t>& shapes) {
    return std::string(std::string("map_") + std::to_string(shapes[0]) + "_" + std::to_string(shapes[1]) + "_" + std::to_string(shapes[2]));
}

int main(int argc, char** argv) {
    using namespace QComputations;
    int world_size, rank;
    size_t atoms_count = 1;
    double amplitude = 0.001;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    QConfig::instance().set_max_photons(atoms_count);
    std::vector<size_t> grid_config = {atoms_count};
    CHE_State state(grid_config);

    for (size_t i = 0; i < atoms_count; i++) {
        state.set_atom(1, i, 0);
    }

    State<CHE_State> init_state(state);

    H_TCH H(init_state);

    auto time_vec = linspace(0, 1000, 1000);

    auto probs = Evolution::schrodinger(init_state.fit_to_basis(H.get_basis()), H, time_vec);

    if (rank == 0) {
        make_probs_files(H, probs, time_vec, H.get_basis(), "rabi_map/original", rank);
    }

// ---------------------- graphs ----------------------------

    std::vector<std::vector<size_t>> grid_configs = {{atoms_count, atoms_count},
                                                     {atoms_count, 0, atoms_count},
                                                     {atoms_count, 0, 0, atoms_count},
                                                     {atoms_count, 0, 0, atoms_count},
                                                     {atoms_count, 0, 0,
                                                      0, atoms_count, 0,
                                                      0, 0, 0},
                                                      {atoms_count, 0, 0, 0, 0, 0, 0, 0, 0,
                                                      0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                      0, 0, 0, 0, 0, 0, 0, 0, atoms_count},
                                                      {atoms_count, 0, 0, 0, 0, 0, 0, 0, 0,
                                                      0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                      0, 0, 0, 0, 0, 0, 0, 0, atoms_count},
                                                    {atoms_count, 0, 0, atoms_count}};
                                                      

    std::vector<std::vector<size_t>> shapes = {{2, 1, 1},
                                               {3, 1, 1},
                                               {2, 2, 1},
                                               {4, 1, 1},
                                               {3, 3, 1},
                                               {3, 3, 3},
                                               {3, 3, 3},
                                               {2, 2, 1}};

    std::vector<size_t> target_cavity = {1, 2, 3, 3, 4, 26, 26, 3};

    std::vector<double> amplitudes = {amplitude,
                                      find_amplitude(amplitude, 2, 1) / 20,
                                      find_amplitude(amplitude, 2, 2) / 20,
                                      find_amplitude(amplitude, 3, 1) / 35,
                                      find_amplitude(amplitude, 2, 2) / 20,
                                      find_amplitude(amplitude, 3, 6) / 20,
                                      find_amplitude(amplitude, 3, 6) / 40,
                                      find_amplitude(amplitude, 2, 2) / 15};

    std::vector<int> times = {16000, 16000, 16000, 16000, 64000, 16000, 16000, 32000};

    for (size_t i = 0; i < shapes.size(); ++i) {
        auto time_vec = linspace(0, times[i], times[i]);
        auto new_amplitude = amplitudes[i];
        CHE_State new_state(grid_configs[i]);

        size_t start_cavity = 0;

        if (i == 4) {
            start_cavity = 4;
        }
        for (size_t i = 0; i < atoms_count; i++) {
            new_state.set_atom(1, i, start_cavity);
        }

        new_state.reshape(shapes[i][0], shapes[i][1], shapes[i][2]);
        new_state.set_waveguide(new_amplitude, 0);

        if (i == 4) {
            new_state.set_waveguide(4, 7, 0, 0);
            new_state.set_waveguide(7, 4, 0, 0);
            new_state.set_waveguide(4, 5, 0, 0);
            new_state.set_waveguide(5, 4, 0, 0);
            new_state.set_waveguide(0, 8, find_amplitude(amplitude, 2, 2) / 20, 0);
        }

        std::string second;
        if (i == shapes.size() - 1) {
            new_state.set_waveguide(0, 2, 0, 0);
            new_state.set_waveguide(2, 3, 0, 0);
            new_state.set_waveguide(1, 2, find_amplitude(amplitude, 2, 2) / 15, 0);
            second = "_not_quadro";
        }

        if (i == shapes.size() - 2) {
            new_state.set_waveguide(0, 26, amplitude, 0);
            new_state.set_waveguide(26, 0, amplitude, 0);
            std::cout << new_state.get_gamma(0, 26) << std::endl;
            second = "_second";
        }

        State<CHE_State> new_init_state(new_state);

        H_TCH H(new_init_state);

        if (rank == 0) show_basis(H.get_basis());

        H.show();

        std::cout << new_amplitude << std::endl;

        auto probs = Evolution::schrodinger(new_init_state.fit_to_basis(H.get_basis()), H, time_vec);

        if (rank == 0) {
            make_probs_files(H, probs, time_vec, H.get_basis(), "rabi_map/" + make_filename(shapes[i]) + second, rank);
        }

        auto p_0 = Evolution::probs_to_cavity_probs(probs, H.get_basis(), 0);
        auto p_1 = Evolution::probs_to_cavity_probs(probs, H.get_basis(), target_cavity[i]);

        make_probs_files(H, p_0.first, time_vec, p_0.second, "rabi_map/0_" + make_filename(shapes[i]) + second, rank);
        make_probs_files(H, p_1.first, time_vec, p_1.second, "rabi_map/" + std::to_string(target_cavity[i]) + "_" + make_filename(shapes[i]) + second, rank);
    }

    MPI_Finalize();
    return 0;
}

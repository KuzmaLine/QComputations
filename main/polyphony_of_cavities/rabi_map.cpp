#include <iostream>
#include <complex>
#include "QComputations_CPU_CLUSTER_NO_PLOTS.hpp"

double find_amplitude(double amplitude, size_t path_length, size_t count_paths) {
    return std::pow(amplitude / count_paths, double(1) / path_length) / std::pow(10, double(1) / path_length - 1);
}

std::string make_filename(const std::vector<size_t>& shapes) {
    return std::string(std::string("map_") + std::to_string(shapes[0]) + "_" + std::to_string(shapes[1]) + "_" + std::to_string(shapes[2]));
}

int main(int argc, char** argv) {
    using namespace QComputations;
    int world_size, rank;
    size_t atoms_count = 2;
    double amplitude = 0.001;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    QConfig::instance().set_max_photons(atoms_count);
    std::vector<size_t> grid_config = {atoms_count};
    CHE_State state(grid_config);

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

    //if (rank == 0) show_basis(H.get_basis());

    //H.show();

    auto time_vec = linspace(0, 1000, 1000);

    //time_vec_to_file("test_time_vec", time_vec, "");

    auto probs = Evolution::quantum_master_equation(init_state.fit_to_basis(H.get_basis()), H, time_vec);

    //basis_to_file("basis_check.csv", H.get_basis());
    if (rank == 0) {
        make_probs_files(H, probs, time_vec, H.get_basis(), "rabi_map/original", rank);
    }
    //make_plot("test_tch.svg", H, probs, time_vec, H.get_basis(), "test_dir");

// ---------------------- graphs ----------------------------

    std::vector<std::vector<size_t>> grid_configs = {{atoms_count, atoms_count},
                                                     {atoms_count, 0, atoms_count},
                                                     {atoms_count, 0, 0, atoms_count},
                                                     {atoms_count, 0, 0, atoms_count},
                                                     {atoms_count, 0, 0,
                                                      0, atoms_count, 0,
                                                      0, 0, 0}};
                                                      

    std::vector<std::vector<size_t>> shapes = {{2, 1, 1},
                                               {3, 1, 1},
                                               {2, 2, 1},
                                               {4, 1, 1},
                                               {3, 3, 1}};

    std::vector<size_t> target_cavity = {1, 2, 3, 3, 4};

    std::vector<double> amplitudes = {amplitude,
                                      find_amplitude(amplitude, 2, 1),
                                      find_amplitude(amplitude, 2, 2),
                                      find_amplitude(amplitude, 3, 1) / 10,
                                      find_amplitude(amplitude, 2, 2)};

    for (size_t i = 0; i < shapes.size(); ++i) {
        auto time_vec = linspace(0, 16000, 16000);
        auto new_amplitude = amplitudes[i];
        CHE_State new_state(grid_configs[i]);

        //new_state.set_n(atoms_count, 0);
        for (size_t i = 0; i < atoms_count; i++) {
            new_state.set_atom(1, i, 0);
        }
        //new_state.set_atom(1, 0, 0);

        new_state.reshape(shapes[i][0], shapes[i][1], shapes[i][2]);
        new_state.set_waveguide(new_amplitude, 0);

        State<CHE_State> new_init_state(new_state);

        H_TCH H(new_init_state);

        if (rank == 0) show_basis(H.get_basis());

        /*
        for (size_t i = 0; i < new_state.get_groups_count(); i++) {
            for (size_t j = 0; j < new_state.get_groups_count(); j++) {
                if (i == j) std::cout << 0 << " ";
                else {
                    std::cout << std::setw(10) << new_state.get_gamma(i, j).real() << " ";
                }
            }
            std::cout << std::endl;
        }
        */

        //H.show();
        std::cout << new_amplitude << std::endl;

        auto probs = Evolution::quantum_master_equation(new_init_state.fit_to_basis(H.get_basis()), H, time_vec);

        if (rank == 0) {
            make_probs_files(H, probs, time_vec, H.get_basis(), "rabi_map/" + make_filename(shapes[i]), rank);
        }

        auto p_0 = Evolution::probs_to_cavity_probs(probs, H.get_basis(), 0);
        auto p_1 = Evolution::probs_to_cavity_probs(probs, H.get_basis(), target_cavity[i]);

        make_probs_files(H, p_0.first, time_vec, p_0.second, "rabi_map/0_" + make_filename(shapes[i]), rank);
        make_probs_files(H, p_1.first, time_vec, p_1.second, "rabi_map/" + std::to_string(target_cavity[i]) + "_" + make_filename(shapes[i]), rank);
    }

    MPI_Finalize();
    return 0;
}

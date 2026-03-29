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
    //QConfig::instance().set_g(0.005);

    /*
    size_t grid_size = 4;
    size_t atoms_num = 2;
    int second_pol = 3;
    std::vector<size_t> grid_config(grid_size, 0);
    grid_config[0] = atoms_num;
    grid_config[second_pol] = atoms_num;
    */

    std::vector<size_t> grid_config = {1, 1, 1};

    /*
    for (size_t i = 0; i < grid_size; i++) {
        if (i == 0 or i == grid_size - 1) {
            grid_config.emplace_back(atoms_num);
        } else {
            grid_config.emplace_back(0);
        }
    }
    */
    State grid(grid_config);
    //grid.reshape(2, 2, 1);
    grid.set_waveguide(0.0005, 1);
    //grid.set_waveguide(0, 1);
    grid.set_qubit(0, 0, 1);
    //grid.set_qubit(0, 1, 1);
    //grid.set_qubit(0, 1, 1);
    //grid.set_qubit(0, 2, 0);
    //grid.set_n(2);
    //grid.set_qubit(0, 3, 1);
    //grid.set_qubit(0, 1, 1);
    //grid.set_qubit(0, 2, 1);
    //grid.set_n(1, 0);
    //grid.set_qubit(1, 0, 1);

    //State grid_copy(grid);
    //grid_copy.set_qubit(0, 0, 0);
    //grid_copy.set_qubit(0, 1, 0);
    //grid_copy.set_qubit(0, 2, 0);
    //grid_copy.set_qubit(1, 0, 1);
    //grid_copy.set_qubit(grid_config.size() - 1, 1, 1);
    //grid_copy.set_qubit(grid_config.size() - 1, 2, 1);

    int ctxt;
    mpi::init_grid(ctxt);
    BLOCKED_H_TCH H(ctxt, grid);

    //H.get_blocked_matrix().write_to_csv_file("H.csv");

    if (rank == 0) { show_basis(H.get_basis()); }

    H.show();

    std::vector<COMPLEX> init_state(H.size(), 0);
    //std::vector<COMPLEX> target_state(H.size(), 0);
    init_state[grid.get_index(H.get_basis())] = COMPLEX(1, 0) / std::sqrt(2);
    grid.set_qubit(0, 0, 0);
    grid.set_qubit(2, 0, 1);
    init_state[grid.get_index(H.get_basis())] = COMPLEX(-1, 0) / std::sqrt(2);
    //size_t target_index = grid_copy.get_index(H.get_basis());
    //target_state[grid_copy.get_index(H.get_basis())] = COMPLEX(1, 0);

    auto time_vec = linspace(0, 600000, 600000);

    auto probs = Evolution::schrodinger(init_state, H, time_vec);
    
    //probs.show();
    //probs.write_to_csv_file("probs.csv");

    double max_prob = 0;
    size_t t_max = 0;
    QConfig::instance().set_eps(1e-1);
    /**    for (size_t t = 0; t < time_vec.size(); t++) {
        auto prob = probs.get(target_index, t);
        //if (rank == mpi::ROOT_ID and time_vec[t] >= 9300 and time_vec[t] <= 9450) {
        //    usleep(1000);
        //    std::cout << t << " - " << prob << std::endl;
        //}
        if (is_zero(prob - 1)) {
            if (prob > max_prob) {
                max_prob = std::max(prob, max_prob);
                t_max = t;
            }
        } else {
            if (max_prob > 0) {
                if (rank == 0) {
                    std::cout << "Prob - " << max_prob << " | " << "t = " << time_vec[t] << " | IN PI - " << time_vec[t] / M_PI << " * PI " << std::endl; 
                }
                
                max_prob = 0;
                t_max = 0;
            }
        }
    }
    */

   /*
    if (rank == 0) {
        matplotlib::make_figure(1920, 1080);
    }

    matplotlib::probs_to_plot(probs, time_vec, H.get_basis());
    //matplotlib::probs_in_cavity_to_plot(probs, time_vec, H.get_basis(), 0);
    if (rank == 0) {
        matplotlib::grid();
        matplotlib::show(false);
    }
    */
    if (rank == 0) {
        matplotlib::make_figure(1920, 1080);
    }

    //matplotlib::probs_to_plot(probs, time_vec, H.get_basis());
    matplotlib::probs_in_cavity_to_plot(probs, time_vec, H.get_basis(), 0);
    if (rank == 0) {
        matplotlib::title("CAVITY_0");
        matplotlib::grid();
        matplotlib::show(false);
    }

    if (rank == 0) {
        matplotlib::make_figure(1920, 1080);
    }

    //matplotlib::probs_to_plot(probs, time_vec, H.get_basis());
    matplotlib::probs_in_cavity_to_plot(probs, time_vec, H.get_basis(), 2);
    if (rank == 0) {
        matplotlib::title("CAVITY_2");
        matplotlib::grid();
        matplotlib::show(false);
    }

    if (rank == 0) {
        matplotlib::make_figure(1920, 1080);
    }

    matplotlib::probs_in_cavity_to_plot(probs, time_vec, H.get_basis(), 1);
    if (rank == 0) {
        matplotlib::title("CAVITY_1");
        matplotlib::grid();
        matplotlib::show();
    }

    MPI_Finalize();
    return 0;
}
#include <iostream>
#include <chrono>
#include <complex>
#include <unistd.h>
#include "/home/kuzmaline/Quantum/diploma/src/QComputations_CPU_CLUSTER_NO_PLOTS.hpp"

int main(int argc, char** argv) {
    using namespace QComputations;
    //QConfig::instance().set_h(0.1); // - изменение постояннной планка на 0.1
    //std::cout << QConfig::instance().h() << std::endl;

    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) QConfig::instance().show(); // - вывод всех параметров внутри конфига

    std::vector<size_t> grid_config = {2};
    State grid(grid_config);
    grid.set_qubit(0, 0, 1);
    grid.set_qubit(0, 1, 1);
    grid.set_n(1);
    //grid.set_min_N(1);
    grid.set_leak_for_cavity(0, 0.02);

    int ctxt;
    mpi::init_grid(ctxt);

    BLOCKED_H_TCH H(ctxt, grid);

    if (rank == 0) show_basis(H.get_basis());

    H.show();

    //if (rank == 0) show_basis(H_single.get_basis());

    //if (rank == 0) H_single.show();

    std::vector<COMPLEX> init_state(H.size(), 0);
    int state_index = grid.get_index(H.get_basis());
    //std::vector<COMPLEX> init_state(H.size(), 0);
    //init_state[1] = COMPLEX(-1/std::sqrt(2), 0);
    //init_state[2] = COMPLEX(1/std::sqrt(2), 0);

    /*
    if (state_index == -1) {
        std::cerr << "Init State error!" << std::endl;

        return 0;
    }

    */

    //init_state[state_index] = COMPLEX(1, 0);
    init_state[1] = COMPLEX(-1/sqrt(2));
    init_state[2] = COMPLEX(1/sqrt(2));
    double eps = 1e-4;
    //if (rank == 0) std::cout << init_state << std::endl;
    //H.print_distributed("H_TCH");

    auto time_vec = linspace(0, 1000, 1000);

    auto probs = Evolution::quantum_master_equation(init_state, H, time_vec);

    if (rank == 0) {
        for (int E = grid.get_grid_energy(); E >= 0; E--) {
            std::cout << std::setw(QConfig::instance().width()) << grid.get_grid_energy() - E << " ";
        }

        std::cout << std::endl;
    }

    for (size_t t = 0; t < time_vec.size(); t++) {
        for (int E = grid.get_grid_energy(); E >= 0; E--) {
            double sum = 0;

            for (size_t i = 0; i < probs.n(); i++) {
                if (get_elem_from_set(H.get_basis(), i).get_grid_energy() == E) {
                    sum += probs.get(i, t);
                }
            }

            if (rank == 0) std::cout << std::setw(QConfig::instance().width()) << sum << " ";
        }


        if (rank == 0) std::cout << std::endl;
        //usleep(100000);
    }

    if (rank == 0) show_basis(H.get_basis());

    size_t index = 0;
    for (size_t i = 0; i < time_vec.size(); i++) {
        //if (rank == 0) std::cout << "t = " << time_vec[i] << std::endl;
        double sum = 0;
        for (size_t j = 0; j < H.size(); j++) {
            auto elem = probs.get(j, i);

            sum += elem;
            //sum += probs_single[j][i];
            if (rank == 0) std::cout << std::setw(QConfig::instance().width()) << elem << " ";
            //if (rank == 0) std::cout << std::setw(QConfig::instance().width()) << probs_single[j][i] << " ";
        }

        usleep(100000);
        if (rank == 0 and std::abs(double(1) - sum) >= eps) {
            std::cout << "SUM ERROR: SUM = " << sum << std::endl;
        } else if (rank == 0) {
            std::cout << "SUM = " << sum << std::endl;
        }
    }

    //H.print_distributed("H_TCH");

    MPI_Finalize();
    return 0;
}

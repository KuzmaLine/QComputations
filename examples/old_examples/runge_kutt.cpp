#include <iostream>
#include <chrono>
#include <complex>
#include <string>
#include <iomanip>
#include <chrono>
#include <unistd.h>
#include <cstdlib>
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
    grid.set_max_N(1);
    grid.set_min_N(1);
    std::string init_state_str = "|1;00>";

    int ctxt;
    mpi::init_grid(ctxt);

    BLOCKED_H_TC H(ctxt, grid);

    H_TC H_single(grid);

    if (rank == 0) show_basis(H.get_basis());

    H.show();

    //if (rank == 0) show_basis(H_single.get_basis());

    //if (rank == 0) H_single.show();

    std::vector<COMPLEX> init_state(H.size(), 0);
    int state_index = State(init_state_str).get_index(H.get_basis());
    //std::vector<COMPLEX> init_state(H.size(), 0);
    //init_state[1] = COMPLEX(-1/std::sqrt(2), 0);
    //init_state[2] = COMPLEX(1/std::sqrt(2), 0);

    /*
    if (state_index == -1) {
        std::cerr << "Init State error!" << std::endl;

        return 0;
    }

    */

    init_state[state_index] = COMPLEX(1, 0);
    double eps = 1e-4;
    //if (rank == 0) std::cout << init_state << std::endl;
    //H.print_distributed("H_TCH");

    auto time_vec = linspace(0, 1000, 1000);

    auto begin = std::chrono::steady_clock::now();
    auto probs_single = Evolution::quantum_master_equation(init_state, H_single, time_vec);
    auto end = std::chrono::steady_clock::now();
    if (rank == 0) std::cout << "SINGLE: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
    if (rank == 0) probs_testing::check_probs(probs_single, H_single.get_basis(), time_vec, eps);

    begin = std::chrono::steady_clock::now();
    auto probs = Evolution::quantum_master_equation(init_state, H, time_vec);
    end = std::chrono::steady_clock::now();
    if (rank == 0) std::cout << "PARALLEL: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;

    //probs.show();

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

    probs_testing::check_probs(probs, H.get_basis(), time_vec, eps);

    MPI_Finalize();
    return 0;
}

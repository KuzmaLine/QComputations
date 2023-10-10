#include <iostream>
#include <chrono>
#include <complex>
#include <string>
#include <iomanip>
#include "/home/kuzmaline/Quantum/diploma/src/QComputations_CPU_CLUSTER_NO_PLOTS.hpp"

int main(int argc, char** argv) {
    using namespace QComputations;
    //QConfig::instance().set_h(0.1); // - изменение постояннной планка на 0.1
    //std::cout << QConfig::instance().h() << std::endl;

    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) QConfig::instance().show(); // - вывод всех параметров внутри конфига

    std::vector<size_t> grid_config = {3};
    State grid(grid_config);
    grid.set_max_N(2);
    grid.set_min_N(2);
    std::string init_state_str = "|0;101>";

    int ctxt;
    mpi::init_grid(ctxt);

    BLOCKED_H_TC H(ctxt, grid);

    H_TC H_single(grid);

    //if (rank == 0) show_basis(H.get_basis());

    //H.show();

    //if (rank == 0) show_basis(H_single.get_basis());

    //if (rank == 0) H_single.show();

    std::vector<COMPLEX> init_state(H.size(), 0);
    int state_index = State(init_state_str).get_index(H.get_basis());

    if (state_index == -1) {
        std::cerr << "Init State error!" << std::endl;

        return 0;
    }

    init_state[state_index] = COMPLEX(1, 0);

    double eps = 1e-9;
    //if (rank == 0) std::cout << init_state << std::endl;
    //H.print_distributed("H_TCH");

    auto time_vec = linspace(0, 1000, 2000);

    auto probs_single = Evolution::quantum_master_equation(init_state, H_single, time_vec);

    if (rank == 0) probs_testing::check_probs(probs_single, H_single.get_basis(), time_vec, eps);

    auto probs = Evolution::quantum_master_equation(init_state, H, time_vec);

    probs_testing::check_probs(probs, H.get_basis(), time_vec, eps);

    MPI_Finalize();
    return 0;
}

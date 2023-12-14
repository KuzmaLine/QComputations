#include "QComputations_CPU_CLUSTER.hpp"
#include <iostream>
#include <regex>
#include <complex>

using COMPLEX = std::complex<double>;

int main(int argc, char** argv) {
    using namespace QComputations;
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    QConfig::instance().set_width(30);
    double h = QConfig::instance().h();
    double w = QConfig::instance().w();
    QConfig::instance().set_g(1.2);

    std::vector<size_t> grid_config = {1, 1};

    State grid(grid_config);
    grid.set_n(1, 0);
    grid.set_waveguide(0, 1, 0.8, 1);
    grid.set_leak_for_cavity(1, 20);

    int ctxt;
    mpi::init_grid(ctxt);
    BLOCKED_H_TCH H(ctxt, grid);

    if (rank == 0) { show_basis(H.get_basis()); }

    H.show();

    std::vector<COMPLEX> init_state(H.size(), 0);
    init_state[grid.get_index(H.get_basis())] = COMPLEX(1, 0);

    auto time_vec = linspace(0, 16, 20000);

    matplotlib::make_figure(1920, 1080);

    //std::vector<double> terms = {0, 0.05, 0.1, 0.15, 0.25, 0.5};
    std::vector<double> terms = {0, 15, 30, 60};
    for (auto term: terms) {
        std::map<std::string, std::string> keywords;
        keywords["label"] = "gamma = " + std::to_string(term);
        grid.set_term(0, term, 0);
        grid.set_term(0, term, 1);
        H.set_grid(grid);
        auto probs = Evolution::quantum_master_equation(init_state, H, time_vec);
        auto prob_sink = blocked_matrix_get_row(probs.ctxt(), probs, H.get_basis().size() - 1).get_vector();

        if (rank == 0) {
            matplotlib::plot(time_vec, prob_sink, keywords);
        }
    }

    if (rank == 0) {
        matplotlib::xlabel("time");
        matplotlib::ylabel("p_sink");
        matplotlib::grid();
        matplotlib::savefig("DAT.png");
        matplotlib::show();
    }

    MPI_Finalize();
    return 0;
}

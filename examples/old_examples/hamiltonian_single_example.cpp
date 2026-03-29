#include <iostream>
#include <chrono>
#include <complex>
#include "/home/kuzmaline/Quantum/diploma/src/QComputations.hpp"

int main(void) {
    using namespace QComputations;
    //QConfig::instance().set_h(0.1);
    QConfig::instance().show();
    //QConfig::instance().set_g(0.001);
    size_t grid_size = 1;
    size_t atoms_num = 1;
    std::vector<size_t> grid_config;

    for (size_t i = 0; i < grid_size; i++) {
        if (i == 0 or i == grid_size - 1) {
            grid_config.emplace_back(atoms_num);
        } else {
            grid_config.emplace_back(0);
        }
    }

    State grid(grid_config);
    //grid.set_qubit(0, 0, 1);
    grid.set_n(1);

    H_TC H(grid);

    show_basis(H.get_basis());
    H.show();

    std::vector<COMPLEX> init_state(H.size(), 0);
    init_state[grid.get_index(H.get_basis())] = COMPLEX(1);

    auto time_vec = linspace(0, 20 * M_PI, 10000);

    auto probs = Evolution::schrodinger(init_state, H, time_vec);
    //auto probs = Evolution::quantum_master_equation(init_state, H, time_vec);

    //std::cout << time_vec << std::endl;
    //probs.show();

    matplotlib::make_figure(1920, 1080);
    matplotlib::probs_to_plot(probs, time_vec, H.get_basis());
    matplotlib::grid();
    matplotlib::show();

    return 0;
}
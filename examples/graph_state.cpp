#include "QComputations_CPU_CLUSTER.hpp"

int main(int argc, char** argv) {
    using namespace QComputations;
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    size_t cavity_counts = 3; // число полостей
    std::vector<size_t> grid_atoms; // задаёт количество частиц в каждой полости, у нас везде будут 0
    std::pair<double, double> gamma = std::make_pair(0.01, 1); // амплитуда перехода из полости в полости
    std::pair<double, double> pair_zero = std::make_pair(0, 0); // длина волновода
    Matrix<std::pair<double, double>> waveguides_parametrs({{pair_zero, gamma, pair_zero}, {gamma, pair_zero, gamma}, {pair_zero, gamma, pair_zero}}); // матрица параметров
    for (size_t i = 0; i < cavity_counts; i++){
        grid_atoms.emplace_back(0);
    }

    State state(grid_atoms);
    state.set_waveguide(waveguides_parametrs);
    state.set_leak_for_cavity(2, 0.001); // для 2 полости делаем утечки 0.001
    state.set_n(1, 0); // в 0 полость помещаем 1 фотон

    int ctxt;
    mpi::init_grid(ctxt); // делаем сетку процессов
    BLOCKED_H_TCH H(ctxt, state);

    if (rank == 0) show_basis(H.get_basis());

    H.show();

    std::vector<COMPLEX> init_state(H.get_basis().size(), 0);
    init_state[state.get_index(H.get_basis())] = COMPLEX(1, 0);
    auto time_vec = linspace(0, 100000, 100000);

    auto probs = Evolution::quantum_master_equation(init_state, H, time_vec);

    if (rank == 0) {
        matplotlib::make_figure(1920, 1080);
    }

    matplotlib::probs_to_plot(probs, time_vec, H.get_basis());
    if (rank == 0) {
        matplotlib::grid();
        matplotlib::show();
    }

    MPI_Finalize();
    return 0;
}
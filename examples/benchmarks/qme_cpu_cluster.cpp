#include "QComputations_CPU_CLUSTER.hpp"
#include <iostream>
#include <regex>
#include <complex>
#include <chrono>
#include <mkl.h>

constexpr bool is_python_api = false;
constexpr int max_photons = 4;

using COMPLEX = std::complex<double>;

int main(int argc, char** argv) {
    using namespace QComputations;

    MPI_Init(&argc, &argv);

    QConfig::instance().set_width(20); // Ширина ячейки элемента матрицы для stdout
    double h = QConfig::instance().h(); // Получить постоянную планка
    double w = QConfig::instance().w(); // Получить частоту
    QConfig::instance().set_g(0.005); // сила взаимодействия с полем атома
    QConfig::instance().set_max_photons(max_photons);

    std::vector<size_t> grid_config = {5};

    TCH_State state(grid_config);
    state.set_n(QConfig::instance().max_photons(), 0);
    // state.set_waveguide(0, 1, 0.01);
    state.set_leak_for_cavity(0, 0.2);

    int ctxt;
    mpi::init_grid(ctxt);
    
    BLOCKED_H_TCH H(ctxt, state);

    // show_basis(H.get_basis());

    // H.show();
    // std::cout << H.size() << std::endl;

    auto time_vec = linspace(0, 100, 100);

    if (is_main_proc())
    std::cout << "H size = " << H.size() << " TIME SIZE = " << time_vec.size() << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    auto probs = quantum_master_equation(State<Basis_State>(state), H, time_vec);

    MPI_Barrier(MPI_COMM_WORLD);

    auto end = std::chrono::high_resolution_clock::now();

    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    if (is_main_proc())
    std::cout << "Execution time: " << elapsed.count() << " ms" << std::endl;

    // make_probs_files(H, probs, time_vec, H.get_basis(), "results/general_tch_QME");

    MPI_Finalize();
    return 0;
}
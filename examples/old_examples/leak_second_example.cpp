#include "QComputations_CPU_CLUSTER.hpp"
#include <iostream>
#include <regex>
#include <complex>

using COMPLEX = std::complex<double>;

namespace QComputations {
Operator<TCH_State> H_TCH_OP() {
    using OpType = Operator<TCH_State>;

    OpType my_H;
    my_H = my_H + OpType(photons_count) + OpType(atoms_exc_count) + OpType(exc_relax_atoms) + OpType(photons_transfer);

    return my_H;
}
}

int main(int argc, char** argv) {
    using namespace QComputations;
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    QConfig::instance().set_width(30);
    double h = QConfig::instance().h();
    double w = QConfig::instance().w();

    std::vector<size_t> grid_config = {0, 0};

    TCH_State grid(grid_config);
    grid.set_n(1, 0);
    grid.set_waveguide(0, 1, 0.01, 1); // интесивность волновода gamma = |gamma|*exp(i * l)
    grid.set_leak_for_cavity(1, 0.1); // интенсинвость стока из 1 полости
    grid.set_leak_for_cavity(0, 0.1); // интенсинвость стока из 0 полости
    //grid.set_gain_for_cavity(0, 0.1); // интенсивность притока фотонов в 0 полость
    //grid.set_max_N(2);
    using OpType = Operator<TCH_State>;

    std::vector<OpType> dec;
    for (size_t i = 0; i < grid.cavities_count(); i++) {
        std::function<State<TCH_State>(const TCH_State&)> a_destroy_i = {[i](const TCH_State& che_state) {
            return set_qudit(che_state, che_state.n(i) - 1, 0, i) * std::sqrt(che_state.n(i));
        }};

        dec.emplace_back(OpType(a_destroy_i));
    }

    int ctxt;
    mpi::init_grid(ctxt);
    auto basis = State_Graph<TCH_State>(grid, H_TCH_OP(), dec).get_basis();
    BLOCKED_H_TCH H(ctxt, grid);

    if (rank == 0) { show_basis(H.get_basis()); }

    H.show();

    State<TCH_State> init_state(grid, basis);

    auto time_vec = linspace(0, 2000, 2000);

    std::vector<double> gamma_1_vec = linspace(0.001, 0.1, 20);
    std::vector<double> gamma_2_vec = linspace(0.001, 0.1, 20);

    for (size_t i = 0; i < 10; i++) {
        gamma_1_vec.emplace_back(0.1 + (i + 1) * 0.01);
        gamma_2_vec.emplace_back(0.1 + (i + 1) * 0.01);
    }

    std::vector<std::vector<double>> x;
    std::vector<std::vector<double>> y;
    std::vector<std::vector<double>> z;

    for (size_t i = 0; i < gamma_1_vec.size(); i++) {
        if (rank == 0) { std::cout << "PROGRESS: " << 100 * double(i) / gamma_1_vec.size() << "%" << std::endl;}
        x.emplace_back(gamma_2_vec.size(), gamma_1_vec[i]);
        y.emplace_back(gamma_2_vec);

        grid.set_leak_for_cavity(0, gamma_1_vec[i]);
        H.set_grid(grid);
        z.emplace_back(scan_gamma(init_state.get_vector(), H, 1, time_vec, gamma_2_vec, 0.9, basis));
    }

    if (rank == 0) {
        matplotlib::surface(x, y, z);
        matplotlib::xlabel("gamma_0");
        matplotlib::ylabel("gamma_1");
        matplotlib::zlabel("tau");
        matplotlib::grid();
        matplotlib::savefig("leak_second_example.png");
        matplotlib::show();
    }

    MPI_Finalize();
    return 0;
}
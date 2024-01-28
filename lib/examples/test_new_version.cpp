#include <iostream>
#include <complex>
#include "QComputations_CPU_CLUSTER.hpp"

namespace QComputations {

class Hydrogen_State: public Basis_State {
    public:
        explicit Hydrogen_State(): Basis_State(2) {}

        COMPLEX get_g_bond() const { return g_bond_; }
        COMPLEX get_g_dist() const { return g_dist_; }
    private:
        COMPLEX g_bond_ = 0.001;
        COMPLEX g_dist_ = 0.01;
};

using OpType = Operator<Hydrogen_State>;

State<Hydrogen_State> H_dist(const Hydrogen_State& state) {
    return sigma_x(state, 1) * state.get_g_dist();
}

State<Hydrogen_State> check_bond_1(const Hydrogen_State& state) {
    //std::cout << "check_bond_1\n";
    return check(state, 1, 0);
}

State<Hydrogen_State> check_dist_0(const Hydrogen_State& state) {
    //std::cout << "check_dist_0\n";
    return check(state, 0, 1);
}

State<Hydrogen_State> H_bond(const Hydrogen_State& state) {
    //std::cout << "H_bond\n";
    auto st = sigma_x(state, 0) * state.get_g_bond();
    return st;
}

State<Hydrogen_State> bond_energy(const Hydrogen_State& state) {
    return get_qudit(state, 0) * QConfig::instance().h() * QConfig::instance().w();
}

State<Hydrogen_State> A_in(const Hydrogen_State& state) {
    return set_qudit(state, 0, state.get_qudit(0) + 1);
}

State<Hydrogen_State> A_out(const Hydrogen_State& state) {
    return set_qudit(state, 0, state.get_qudit(0) - 1);
}

} // namespace QComputations

int main(int argc, char** argv) {
    using namespace QComputations;
    int world_size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    Hydrogen_State state;
    state.set_qudit(0, 0);
    state.set_qudit(1, 1);

    std::cout << state.to_string() << std::endl;

    OpType my_H;
    my_H = my_H + OpType(H_dist) * OpType(check_bond_1) + OpType(H_bond) * OpType(check_dist_0) + OpType(bond_energy);

    OpType my_A_in(A_in);
    OpType my_A_out(A_out);

    std::vector<std::pair<double, OpType>> dec;
    dec.emplace_back(std::make_pair(0.01, my_A_in));
    dec.emplace_back(std::make_pair(0.01, my_A_out));

    auto res = my_A_in.run(state);
    std::cout << res.to_string() << std::endl;

    int ctxt;
    mpi::init_grid(ctxt);

    BLOCKED_H_by_Operator H(ctxt, State(state), my_H, dec);

    if (rank == 0) show_basis(H.get_basis());

    H.show();

    std::cout << "Here-1\n";
    auto time_vec = linspace(0, 2000, 2000);
    std::cout << "here-2\n";

    auto probs = Evolution::quantum_master_equation(State(state).fit_to_basis(H.get_basis()), H, time_vec);

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
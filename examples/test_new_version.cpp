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
        COMPLEX g_bond_ = 0.01;
        COMPLEX g_dist_ = 0.001;
};

using OpType = Operator<Hydrogen_State>;

State<Hydrogen_State> H_dist(const Hydrogen_State& state) {
    std::cout << "H_dist\n";
    return sigma_x(state, 1) * state.get_g_dist();
}

State<Hydrogen_State> check_bond_1(const Hydrogen_State& state) {
    std::cout << "check_bond_1\n";
    return check(state, 1, 0);
}

State<Hydrogen_State> check_dist_0(const Hydrogen_State& state) {
    std::cout << "check_dist_0\n";
    return check(state, 0, 1);
}

State<Hydrogen_State> H_bond(const Hydrogen_State& state) {
    std::cout << "H_bond\n";
    auto st = sigma_x(state, 0) * state.get_g_bond();

    std::cout << st.to_string() << std::endl;

    st.insert(state);
    std::cout << st.to_string() << std::endl;
    st[st.get_index(state)] = QConfig::instance().h() * QConfig::instance().w() * state.get_qudit(0);
    std::cout << st.to_string() << std::endl;
    return st;
}

} // namespace QComputations

int main(int argc, char** argv) {
    using namespace QComputations;
    int world_size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    Hydrogen_State state;
    state.set_qudit(1, 0);
    state.set_qudit(1, 1);

    std::cout << state.to_string() << std::endl;

    OpType my_H;
    my_H = my_H + OpType(H_dist) * OpType(check_bond_1) + OpType(H_bond) * OpType(check_dist_0);

    auto res = my_H.run(state);

    std::cout << res.to_string() << std::endl;

    MPI_Finalize();
    return 0;
}
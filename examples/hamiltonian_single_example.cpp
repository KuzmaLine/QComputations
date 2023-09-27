#include <iostream>
#include <chrono>
#include <complex>
#include "/home/kuzmaline/Quantum/diploma/src/QComputations.hpp"

int main(void) {
    using namespace QComputations;
    QConfig::instance().set_h(0.1);
    QConfig::instance().show();
    State grid("|0;11>");
    grid.set_max_N(2);
    grid.set_min_N(2);

    H_TC H(grid);

    show_basis(H.get_basis());
    H.show();

    return 0;
}
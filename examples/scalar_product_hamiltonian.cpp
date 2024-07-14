/*
Демонстрация реализации собственного понятия состояния и 
операторов. Одноядерная версия.
*/
#include "QComputations_SINGLE.hpp"

#include <iostream>
#include <complex>

using COMPLEX = std::complex<double>;

using namespace QComputations;

COMPLEX h_func(const Basis_State& i, const Basis_State& j) {
    COMPLEX res(0, 0);
    for (size_t k = 0; k < i.qudits_count(); k++) {
        res += (i.get_qudit(k) ^ j.get_qudit(k));
    }

    return res;
}

int main(int argc, char** argv) {
    Basis_State st("|0;0;1>");
    H_by_Scalar_Product<Basis_State> H(st, h_func);
    show_basis(H.get_basis());
    H.show();

    return 0;
}
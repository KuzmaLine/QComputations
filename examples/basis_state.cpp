/*
Демонстрация инициализации базисного состояния.
*/
#include "QComputations_SINGLE.hpp"

#include <iostream>

int main(int argc, char** argv) {
    using namespace QComputations;

// Creating Basis_State
    Basis_State state(5, 1, {3, 2});

    state.set_qudit(1, 1, 0);
    state.set_max_val(2, 2, 0);
    state.set_qudit(2, 2, 0);
    state.set_qudit(1, 0, 1);

    std::cout << "SET_MANUALLY: " << state.to_string() << std::endl;
    // |0;1;2>|1;0>

    Basis_State new_state(8, 4, 4);
    new_state.set_state("|4;2>|1;1>|3;2>|1;2>");

    std::cout << "SET_STATE: " << new_state.to_string() << std::endl;
    // |4;2>|1;1>|3;2>|1;2>

    new_state.set_group(2, "|0;0>");

    std::cout << "SET_GROUP: " << new_state.to_string() << std::endl;
    // |4;2>|1;1>|0;0>|1;2>

    Basis_State by_vec({0,1,0,1,1}, 1, {3, 2});
    std::cout << "SET_VEC: " << by_vec.to_string() << std::endl;

// Creating State<Basis_State>

    Basis_State zero("|0;0>", 1);
    Basis_State one("|1;1>", 1);

    State<Basis_State> st(zero);
    st += one;
    st *= 2;

    std::cout << "State<Basis_State>: " << st.to_string() << std::endl;
    // 2 * |0;0> + 2 * |1;1>

    st.normalize();

    std::cout << "State<Basis_State> NORMALIZE: " << st.to_string() << std::endl;
    // (|0;0> + |1;1>)/sqrt(2)
    Basis_State zero_one("|0;1>", 1);
    // Add state in basis of State<Basis_State>, but with coefficient = 0
    st.insert(zero_one);

    std::cout << "State<Basis_State> + zero_one: " << st.to_string() << std::endl;
    // (|0;0> + |1;1>)/sqrt(2) + 0*|0;1>

    return 0;
}
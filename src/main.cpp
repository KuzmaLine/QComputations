//#define _USE_MATH_DEFINES
#include <iostream>
#include "functions.hpp"
#include "additional_operators.hpp"
#include "matrix.hpp"
#include "hamiltonian.hpp"
#include "state.hpp"
#include "test.hpp"
#include "plot.hpp"
#include "config.hpp"
//#include "graph.hpp"
#include "dynamic.hpp"

namespace plt = matplotlibcpp;

int main(void) {
    int n = 1;
    int m = 2;

    State state("|0>|10>");
    H_TC H(n, m, state);
    std::cout << state.to_string() << " n = " << n << " m = " << m <<" h = " << config::h << " w = " << config::w << " g = " << config::g << " LOSS_PHOTONS = " << config::LOSS_PHOTONS << std::endl;

    //State_Graph graph(state);
    //graph.show();

    //std::cout << std::endl;
    auto basis = H.get_basis();

    for (const auto& b: basis) {
        std::cout << std::setw(config::WIDTH) << b.to_string() << " ";
    }
    std::cout << std::endl;

    H.show(config::WIDTH);

    auto H_m = H.get_matrix();

    auto p = H.eigen();

    //functions_testing::check_eigenvectors(p.first, p.second, H_m);

    /*
    for (size_t i = 0; i < H.size(); i++) {
        for (size_t j = 0; j < H.size(); j++) {
            std::cout << scalar_product(p.second.col(i), p.second.col(j)) << std::endl;
        }
        std::cout << std::endl;
    }
    */
    //Matrix<COMPLEX> a = matrix_testing::create_hermit_rand_matrix(1024, 1024, COMPLEX(0, 0), COMPLEX(10, 10));
    //auto a_p = Hermit_Lanczos(a);
    //functions_testing::check_eigenvectors(a_p.first, a_p.second, a);

    std::vector<double> time_vec = make_timeline(0, 100 * M_PI, 5 * M_PI);
    std::vector<COMPLEX> st(H.size(), 0);
    st[1] = 1;
    auto probs = Evolution::evol(st, H, time_vec);

    size_t index = 0;
    for (const auto& b: basis) {
        std::cout << std::setw(config::WIDTH) << b.to_string() << " : ";
        for (size_t i = 0; i < time_vec.size(); i++) {
            std::cout << std::setw(config::WIDTH) << probs[index][i] << " ";
        }
        std::cout << std::endl;
        index++;
    }

    std::array<std::string, 3> ls = {"-", "--", "-."};
    std::array<std::string, 9> c = {"b", "r", "g", "tab:orange", "m", "tab:brown", "tab:violet", "tab:olive", "tab:purple"};
    std::vector<std::map<std::string, std::string>> keywords(basis.size());
    index = 0;
    for (auto& item: keywords) {
        item["ls"] = ls[(index / ls.size()) % ls.size()];
        item["c"] = c[index % c.size()]; 
        index++;
    }
    matplotlib::make_figure(640, 400);
    matplotlib::probs_to_plot(probs, time_vec, basis, keywords);
    matplotlib::grid();
    matplotlib::show();
    return 0;
}

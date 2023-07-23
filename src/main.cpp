//#define _USE_MATH_DEFINES
#include <iostream>
#include "functions.hpp"
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
    int n = 2;
    int m = 1;
    double gamma = 0.005;

    State state("|1>|1>");
    //H_TC H(n, m, state, !is_zero(gamma));
    H_JC H(n, state, !is_zero(gamma));
    std::cout << state.to_string() << " n = " << n << " m = " << m <<" h = " << config::h << " w = " << config::w << " g = " << config::g << " LOSS_PHOTONS = " << config::LOSS_PHOTONS << std::endl;

    auto basis = H.get_basis();

    for (const auto& b: basis) {
        std::cout << std::setw(config::WIDTH) << b.to_string() << " ";
    }
    std::cout << std::endl;

    H.show(config::WIDTH);

    std::vector<double> time_vec = make_timeline(0, 300 * M_PI, M_PI / 4);
    std::vector<COMPLEX> st(H.size(), 0);
    st[state.get_index(basis)] = 1;

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

    //auto probs = Evolution::schrodinger(st, H, time_vec);
    auto probs = Evolution::quantum_master_equation(st, H, time_vec, gamma, true);
    //std::vector<double> x = make_timeline(0, 100, 1);
    //functions_testing::check_runge_kutt<double, double>(x, double(0), &func, &func_correct);

    std::array<std::string, 3> ls = {"-", "--", "-."};
    std::array<std::string, 9> c = {"b", "r", "g", "tab:orange", "m", "tab:brown", "tab:violet", "tab:olive", "tab:purple"};
    std::vector<std::map<std::string, std::string>> keywords(basis.size());
    size_t index = 0;
    for (auto& item: keywords) {
        item["ls"] = ls[(index / ls.size()) % ls.size()];
        item["c"] = c[index % c.size()];
        index++;
    }


    /*
    matplotlib::make_figure(config::fig_width, config::fig_height, config::dpi);
    matplotlib::probs_to_plot(probs, time_vec, basis, keywords);
    matplotlib::grid();
    matplotlib::show();
    */

    std::vector<double> gamma_vec = make_timeline(0.003, 0.1, 0.001);
    auto tau_vec = Evolution::scan_gamma(st, H, time_vec, gamma_vec, 0.9);

    matplotlib::make_figure(config::fig_width, config::fig_height, config::dpi);
    plt::plot(gamma_vec, tau_vec);
    matplotlib::grid();
    matplotlib::show();
    /*
    matplotlib::make_figure(config::fig_width, config::fig_height, config::dpi);
    matplotlib::rho_probs_to_plot(probs, time_vec, basis, keywords);
    matplotlib::grid();
    matplotlib::show();
    */

    /*
    Matrix<double> a ({{1, 10, 0, 0, 0}, {2, 8, 2, 0, 0}, {0, 4, 23423, 1, 0}, {0, 0, 111, 8, 9}, {0, 0, 0, 5, 0}});
    std::vector<double> y = {1, 9, 2, 3, 4};

    auto answer = Pro_Race_Algorithm(a, y);
    show_vector(answer);
    functions_testing::check_pro_race(a, answer, y);
    */

    /*
    std::vector<double> x = {1, 2, 5, 10, 12, 15};
    std::vector<double> f = {-10, -7, -13, 20, 10, 5}; 
    matplotlib::make_figure(config::fig_width, config::fig_height, config::dpi);
    plt::scatter(x, f, 50);
    auto x_time = make_timeline(1, 15, 0.01);
    auto func = Cubic_Spline_Interpolate(x, f);
    std::vector<double> y_time;
    for (const auto& t: x_time) {
        y_time.emplace_back(func(t));
    }

    plt::plot(x_time, y_time);
    plt::grid();
    plt::show();
    */
    /*
    matplotlib::make_figure(config::fig_width, config::fig_height, config::dpi);
    matplotlib::rho_diag_to_plot(probs, time_vec, basis, keywords);
    matplotlib::grid();
    matplotlib::show(false);

    matplotlib::make_figure(config::fig_width, config::fig_height, config::dpi);
    matplotlib::rho_subdiag_to_plot(probs, time_vec, basis, keywords);
    matplotlib::grid();
    matplotlib::show();
    */
    return 0;
}

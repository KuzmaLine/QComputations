//#define _USE_MATH_DEFINES
#include <iostream>
#include "functions.hpp"
#include "matrix.hpp"
#include "hamiltonian.hpp"
#include "state.hpp"
#include "test.hpp"
#include "plot.hpp"
#include "config.hpp"
#include "dynamic.hpp"
#include <chrono>
#include <omp.h>

#ifdef ENABLE_MPI
#include "mpi_functions.hpp"
#endif

namespace plt = matplotlibcpp;

using COMPLEX = std::complex<double>;

COMPLEX func(size_t i, size_t j) {
    return COMPLEX(i * (j + 1));
}

int main(int argc, char** argv) {
    int n = 2048;

    std::vector<size_t> grid_config = {1, 1};
    //State state("|0;00>");
    State state(grid_config);
    state.set_gamma(0.002);
    state.set_leak_for_cavity(0, 0.005);
    //state.set_gain_for_cavity(0, 0.002);
    state.set_max_N(1);
    state.set_min_N(1);
    //state.set_leak_for_cavity(1, 0.0002);

    std::cout << omp_get_max_threads() << std::endl;

#ifdef ENABLE_MPI
    int rank, world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    //std::cout << "WORLD SIZE - " << world_size << std::endl;
    if (world_size == 1) {
        std::cerr << "Should have at least 2 processes\n";
 
       MPI_Finalize();
        return 1;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::map<int, std::vector<mpi::MPI_Data>> data;
    data[COMMAND::GENERATE_H].resize(1);
    data[COMMAND::GENERATE_H_FUNC].resize(1);

    data[COMMAND::GENERATE_H][0].state = state;
    data[COMMAND::GENERATE_H_FUNC][0].n = n;
    data[COMMAND::GENERATE_H_FUNC][0].func = func;

    if (rank != 0) {
        mpi::run_mpi_slaves(data);
        MPI_Finalize();
        return 0;
    }
#endif

    Matrix<COMPLEX> a (n, n, 1);
    Matrix<COMPLEX> b (n, n, 2);

    //a.show();
    //b.show();
    auto begin = std::chrono::steady_clock::now();
    auto c = a * b;
    auto end = std::chrono::steady_clock::now();
    std::cout << "MULTIPLY: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
    //c.show();

#ifdef ENABLE_MPI
    mpi::stop_mpi_slaves();
#endif
    return 0;

    begin = std::chrono::steady_clock::now();
    H_TCH H(state);
    end = std::chrono::steady_clock::now();

    std::cout << "H_TCH: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;

    auto basis = H.get_basis();
    //show_basis(basis);

    //H.show(config::WIDTH);
    //auto basis_correct = H_correct.get_basis();
    //show_basis(basis_correct);

    //H_correct.show(config::WIDTH);

    std::vector<double> time_vec = make_timeline(0, 200 * M_PI, M_PI / 8);
    time_vec = linspace(0, 4000, 4000);
    std::vector<COMPLEX> st(H.size(), 0);
    //st[state.get_index(basis)] = 1;

    st[State("|0,0;0,1>").get_index(basis)] = 1;
    //st[State("|0;01>").get_index(basis)] = 1/sqrt(3);
    //st[State("|0;10>").get_index(basis)] = -1/sqrt(3);
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

    begin = std::chrono::steady_clock::now();
    //auto probs = Evolution::schrodinger(st, H, time_vec);
    auto probs = Evolution::quantum_master_equation(st, H, time_vec, false);
    end = std::chrono::steady_clock::now();
    //std::cout << "SCHRODINGER: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
    std::cout << "QME: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
    //auto probs = Evolution::quantum_master_equation(st, H, time_vec, gamma, false);
    //std::vector<double> x = make_timeline(0, 100, 1);
    //functions_testing::check_runge_kutt<double, double>(x, double(0), &func, &func_correct);

    /*
    std::array<std::string, 3> ls = {"-", "--", "-."};
    std::array<std::string, 9> c = {"b", "r", "g", "tab:orange", "m", "tab:brown", "tab:violet", "tab:olive", "tab:purple"};
    std::vector<std::map<std::string, std::string>> keywords(basis.size());
    size_t index = 0;
    for (auto& item: keywords) {
        item["ls"] = ls[(index / ls.size()) % ls.size()];
        item["c"] = c[index % c.size()];
        index++;
    }

    */


    matplotlib::make_figure(config::fig_width, config::fig_height, config::dpi);
    matplotlib::probs_to_plot(probs, time_vec, basis);
    matplotlib::grid();
    matplotlib::show();

    /*
    std::vector<double> gamma_vec = make_timeline(0.008, 0.1, 0.001);
    auto tau_vec = Evolution::scan_gamma(st, H, 0, time_vec, gamma_vec, 0.9);

    auto x = linspace(0.008, 0.1, 1000);
    auto tau_spline = Cubic_Spline_Interpolate(gamma_vec, tau_vec);
    auto tau_spline_vec = f_vector(tau_spline, x);

    auto gamma_min = fmin(tau_spline, x[0], x[x.size() - 1]);
    std::cout << "MIN = " << gamma_min << std::endl;

    H.set_leak(0, gamma_min);
    auto probs = Evolution::quantum_master_equation(st, H, time_vec, gamma_min);
    matplotlib::make_figure(config::fig_width, config::fig_height, config::dpi);
    matplotlib::probs_to_plot(probs, time_vec, basis);
    matplotlib::grid();
    matplotlib::show();

    matplotlib::make_figure(config::fig_width, config::fig_height, config::dpi);
    plt::scatter(gamma_vec, tau_vec, 40);
    plt::plot(x, tau_spline_vec);
    matplotlib::grid();
    matplotlib::show();
    */
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
    matplotlib::rho_diag_to_plot(probs, time_vec, basis);
    matplotlib::grid();
    matplotlib::show(false);

    matplotlib::make_figure(config::fig_width, config::fig_height, config::dpi);
    matplotlib::rho_subdiag_to_plot(probs, time_vec, basis);
    matplotlib::grid();
    matplotlib::show();
    */

#ifdef ENABLE_MPI
    mpi::stop_mpi_slaves();
#endif
    return 0;
}

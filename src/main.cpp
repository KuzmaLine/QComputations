//#define _USE_MATH_DEFINES
#include <iostream>
#include "functions.hpp"
#include "matrix.hpp"
#include "hamiltonian.hpp"
#include "state.hpp"
//#include "test.hpp"
#include "plot.hpp"
#include "config.hpp"
#include "dynamic.hpp"
#include <chrono>
#include <omp.h>

#ifdef ENABLE_MPI
#include "mpi_functions.hpp"
#endif

#ifdef ENABLE_MATPLOTLIB
namespace plt = matplotlibcpp;
#endif

using COMPLEX = std::complex<double>;

COMPLEX func(size_t i, size_t j) {
    return COMPLEX(i * (j + 1));
}

int main(int argc, char** argv) {
    int n = 8000;

    /*
    unsigned int tmp = 0xFFFFFFFF;
    BigUInt a_num(tmp);
    BigUInt b_num(2);
    auto res = a_num + b_num;
    auto str_res = res.binary_str();

    std::cout << str_res.size() << std::endl;
    for (size_t i = 0; (i + 1) * 32 < str_res.size(); i++) {
        str_res.insert((i + 1) * 32 + i, " ");
    }
    std::cout << str_res << std::endl;
    */
    /*
    res = res - BigUInt(2);
    str_res = res.binary_str();
    std::cout << str_res.size() << std::endl;
    for (size_t i = 0; (i + 1) * 32 < str_res.size(); i++) {
        str_res.insert((i + 1) * 32 + i, " ");
    }
    std::cout << str_res << std::endl;

    res <<= 3;
    str_res = res.binary_str();
    std::cout << str_res.size() << std::endl;
    for (size_t i = 0; (i + 1) * 32 < str_res.size(); i++) {
        str_res.insert((i + 1) * 32 + i, " ");
    }
    std::cout << str_res << std::endl;

    res >>= 2;
    str_res = res.binary_str();
    std::cout << str_res.size() << std::endl;
    for (size_t i = 0; (i + 1) * 32 < str_res.size(); i++) {
        str_res.insert((i + 1) * 32 + i, " ");
    }
    std::cout << str_res << std::endl;

    std::cout << "CHECK >= " << (BigUInt(3) >= BigUInt(3)) << std::endl;
    */
    /*
    std::cout << res << std::endl;
    res <<= 1;
    std::cout << res << std::endl;
    res = res % BigUInt(2);
    str_res = res.binary_str();
    std::cout << str_res.size() << std::endl;
    for (size_t i = 0; (i + 1) * 32 < str_res.size(); i++) {
        str_res.insert((i + 1) * 32 + i, " ");
    }
    std::cout << str_res << std::endl;

    res = BigUInt(3) % BigUInt(2);
    str_res = res.binary_str();
    std::cout << str_res.size() << std::endl;
    for (size_t i = 0; (i + 1) * 32 < str_res.size(); i++) {
        str_res.insert((i + 1) * 32 + i, " ");
    }
    std::cout << str_res << std::endl;
    
    */

    State tmp_state("|12,10;10,1>");
    std::cout << tmp_state.to_uint().binary_str() << std::endl;
    State second_state("|11,13;11,1>");
    print_state_biguint(tmp_state);
    std::cout << std::endl;
    print_state_biguint(second_state);
    std::cout << std::endl;
    tmp_state.from_uint(second_state.to_uint());
    print_state_biguint(tmp_state);
    std::cout << std::endl;
    return 0;


    std::vector<size_t> grid_config = {1, 1};
    //State state("|0;00>");
    State state(grid_config);
    state.set_gamma(0.002);
    state.set_leak_for_cavity(0, 0.005);
    state.set_gain_for_cavity(0, 0.002);
    state.set_max_N(2);
    state.set_min_N(0);
    //state.set_leak_for_cavity(1, 0.0002);

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

    using type = double;
    //Matrix<double> a (n, n, 1);
    //Matrix<double> b (n, n, 1);

    //Matrix<double> a = matrix_testing::create_rand_matrix<double>(n, n, 0.0, 10.0);
    //Matrix<double> b = matrix_testing::create_rand_matrix<double>(n, n, 0.0, 10.0);
    //Matrix<COMPLEX> a({{1, 9, 1}, {2, 8, 1}, {3, 7, 1}, {4, 4, 1}, {5, 5, 1}, {6, 3, 1}});
    //Matrix<COMPLEX> b({{1, 2}, {4, 6}, {6, 4}});

    /*
    Matrix<type> a({{1, 2, -1, -1, 4},
                      {2, 0, 1, 1, -1},
                      {1, -1, -1, 1, 2},
                      {-3, 2, 2, 2, 0},
                      {4, 0, -2, 1, -1},
                      {-1, -1, 1, -3, 2}});
    Matrix<type> b({{1, -1, 0, 2},
                      {2, 2, -1, -2},
                      {1, 0, -1, 1},
                      {-3, -1, 1, -1},
                      {4, 2, -1, 1}});
    */

    /*
    Matrix<double> a({{0, 1, 2, 3, 4},
                      {1, 2, 3, 4, 5},
                      {2, 3, 4, 5, 6},
                      {3, 4, 5, 6, 7},
                      {4, 5, 6, 7, 8}});
    Matrix<double> b({{0, 1, 2, 3, 4},
                      {1, 2, 3, 4, 5},
                      {2, 3, 4, 5, 6},
                      {3, 4, 5, 6, 7},
                      {4, 5, 6, 7, 8}});
    */
    //a.show();
    //b.show();

    //a.to_fortran_style();
    //b.to_fortran_style();
    //a.show();
    //b.show();

    /*
    auto begin = std::chrono::steady_clock::now();
    auto c = a * b;
    auto end = std::chrono::steady_clock::now();
    std::cout << "MULTIPLY: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
    c.show();
    std::cout << c.is_c_style() << std::endl;

    a.set_multiply_mode(config::COMMON_MODE);
    begin = std::chrono::steady_clock::now();
    c = a * b;
    end = std::chrono::steady_clock::now();
    std::cout << "COMMON: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;

    if (a.is_c_style()) {
        Matrix<type> check(true, a.n(), b.m(), type(0));
        for (size_t i = 0; i < a.n(); i++) {
            for (size_t j = 0; j < b.m(); j++) {
                for (size_t k = 0; k < a.m(); k++) {
                    check[i][j] += a[i][k] * b[k][j];
                }
            }
        }

        check.show();
    } else {
        Matrix<type> check(false, a.n(), b.m(), type(0));
        for (size_t i = 0; i < a.n(); i++) {
            for (size_t j = 0; j < b.m(); j++) {
                for (size_t k = 0; k < a.m(); k++) {
                    //std::cout << i << " " << j << " " << k << " - " << a(i, k) << std::endl;
                    check(i, j) += a(i, k) * b(k, j);
                }
            }
        }

        check.show();
    }
    */
    //std::cout << std::endl;

/*
#ifdef ENABLE_MPI
    mpi::stop_mpi_slaves();
#endif
    return 0;
*/

    auto begin = std::chrono::steady_clock::now();
    H_TCH H(state);
    auto end = std::chrono::steady_clock::now();

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

#ifdef ENABLE_MATPLOTLIB

    //std::cout << "P HERE\n";
    matplotlib::make_figure(config::fig_width, config::fig_height, config::dpi);
    //std::cout << "P HERE\n";
    matplotlib::probs_to_plot(probs, time_vec, basis);
    matplotlib::grid();
    matplotlib::show(false);

    begin = std::chrono::steady_clock::now();
    //auto probs = Evolution::schrodinger(st, H, time_vec);
    probs = Evolution::Parallel_QME(st, H, time_vec, false);
    end = std::chrono::steady_clock::now();
    //std::cout << "SCHRODINGER: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
    std::cout << "QME: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;

    //std::cout << config::fig_width << " " << config::fig_height << " " << config::dpi << std::endl;
    matplotlib::make_figure(config::fig_width, config::fig_height, config::dpi);
    //matplotlib::make_figure();
    //std::cout << "P HERE\n";
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

#endif // ENABLE_MATPLOTLIB

#ifdef ENABLE_MPI
    mpi::stop_mpi_slaves();
#endif
    return 0;
}


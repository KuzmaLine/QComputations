#pragma once
#include <complex>

namespace config {
//  Quantum parametrs
    constexpr double h = 1;
    constexpr double w = 1;
    constexpr double g = 0.01;

// matplotlib figure() parametrs
    constexpr double fig_width = 960;
    constexpr double fig_height = 540;
    constexpr size_t dpi = 80;

// printing parametrs
    constexpr int WIDTH = 15;

// is_zero
    constexpr double eps = 10e-12;

    constexpr int COMMON_MODE = 0;
    constexpr int CANNON_MODE = 1;
    constexpr int DIM_MODE = 2;
    constexpr int P_GEMM_MODE = 3;

    constexpr int MULTIPLY_MODE = P_GEMM_MODE;
}


#pragma once
#include <complex>

namespace config {
//  Quantum parametrs
    constexpr double h = 1;
    constexpr double w = 1;
    constexpr double g = 0.01;
    constexpr bool LOSS_PHOTONS = false;

// matplotlib figure() parametrs
    constexpr double fig_width = 960;
    constexpr double fig_height = 540;
    constexpr size_t dpi = 80;

// printing parametrs
    constexpr int WIDTH = 15;
}

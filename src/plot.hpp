#pragma once
#include "dynamic.hpp"
#include "state.hpp"
#include "matplotlibcpp.hpp"
#include <string>
#include <map>
#include <algorithm>

namespace matplotlib {
    void make_figure(size_t x = 0, size_t y = 0, size_t dpi = 100);
    void probs_to_plot(const Evolution::Probs& probs,
                       const std::vector<double>& time_vec,
                       const std::set<State>& basis,
                       std::vector<std::map<std::string, std::string>> keywords = {});
    void show(bool is_block = true);
    void savefig(const std::string& filename, size_t dpi = 100);
    void grid(bool is_enable = true);
}

namespace plotly {

}

namespace matlab {

}

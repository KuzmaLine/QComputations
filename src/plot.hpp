#pragma once
#include "dynamic.hpp"
#include "state.hpp"
#include "matplotlibcpp.hpp"
#include <algorithm>

namespace matplotlib {
    void probs_to_plot(const Evolution::Probs& probs, std::vector<double> time_vec, std::set<State> basis);
}

namespace plotly {

}

namespace matlab {

}

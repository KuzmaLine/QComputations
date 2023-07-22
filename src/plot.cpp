#include "plot.hpp"
#include "functions.hpp"

namespace {
    namespace plt = matplotlibcpp;
}

void matplotlib::probs_to_plot(const Evolution::Probs& probs, std::vector<double> time_vec, std::set<State> basis) {
    size_t index = 0;
    for (const auto& state: basis) {
        std::vector<double> state_probs(time_vec.size());
        //std::copy(probs[index * probs.m()], probs[(index + 1) * probs.m()], std::back_inserter(state_probs));
        for (size_t i = 0; i < time_vec.size(); i++) {
            state_probs[i] = probs[index][i];
        }
        index++;
        //plt::named_plot(state.to_string(), state_probs, time_vec);
        plt::plot(time_vec, state_probs);
    }
}

void matplotlib::show(bool is_block) {
    plt::show(is_block);
}

void matplotlib::make_figure(size_t x, size_t y, size_t dpi) {
    if (x == 0 or y == 0) plt::figure();
    else {
        plt::figure_size(x, y, dpi);
    }
}

void matplotlib::savefig(const std::string& filename, size_t dpi) {
    plt::save(filename, dpi);
}

void matplotlib::grid(bool is_enable) {
    plt::grid(is_enable);
}

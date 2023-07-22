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

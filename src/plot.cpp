#include "plot.hpp"
#include "functions.hpp"

namespace {
    namespace plt = matplotlibcpp;
}

void matplotlib::probs_to_plot(const Evolution::Probs& probs, 
                               const std::vector<double>& time_vec,
                               const std::set<State>& basis,
                               std::vector<std::map<std::string, std::string>> keywords) {
    size_t index = 0;
    for (const auto& state: basis) {
        if (keywords.size() <= index) {
            std::map<std::string, std::string> tmp;
            keywords.emplace_back(tmp);
        }
        keywords[index]["label"] = state.to_string();
        /*
        for (const auto& p: keywords[index]) {
            std::cout << p.first << " " << p.second << std::endl;
        }
        */
        plt::plot(time_vec, probs.row(index), keywords[index]);
        index++;
        //plt::plot(time_vec, state_probs);
    }
    plt::legend();
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

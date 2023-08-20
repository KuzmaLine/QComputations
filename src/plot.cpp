#include "plot.hpp"
#include "functions.hpp"

namespace {
#ifdef ENABLE_MATPLOTLIB
    namespace plt = matplotlibcpp;
#endif
}

#ifdef ENABLE_MATPLOTLIB

void matplotlib::probs_to_plot(const Evolution::Probs& probs, 
                               const std::vector<double>& time_vec,
                               const std::set<State>& basis,
                               std::vector<std::map<std::string, std::string>> keywords) {
    //std::cout << "HERE\n";
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

void matplotlib::rho_probs_to_plot(const Evolution::Probs& probs,
                       const std::vector<double>& time_vec,
                       const std::set<State>& basis,
                       std::vector<std::map<std::string, std::string>> keywords) {
    size_t from = 0;
    size_t to = 0;
    size_t basis_size = basis.size();
    for (const auto& state_from: basis) {
        for (const auto& state_to: basis) {
            size_t index = from * basis_size + to;
            if (keywords.size() <= index) {
                while(keywords.size() <= index) {
                    std::map<std::string, std::string> tmp;
                    keywords.emplace_back(tmp);
                }
            }

            if (from == to) {
                keywords[index]["label"] = state_from.to_string();
            } else {
                keywords[index]["label"] = state_from.to_string() + " -> " + state_to.to_string();
            }
            /*
            for (const auto& p: keywords[index]) {
                std::cout << p.first << " " << p.second << std::endl;
            }
            */
            plt::plot(time_vec, probs.row(index), keywords[index]);
            //plt::plot(time_vec, state_probs);
            to++;
        }
        from++;
        to = from;
    }
    plt::legend();
}

void matplotlib::rho_diag_to_plot(const Evolution::Probs& probs,
                                  const std::vector<double>& time_vec,
                                  const std::set<State>& basis,
                                  std::vector<std::map<std::string, std::string>> keywords) {
    size_t state_index = 0;
    size_t basis_size = basis.size();
    for (const auto& state: basis) {
        size_t index = state_index * basis_size + state_index;
        if (keywords.size() <= index) {
            while(keywords.size() <= index) {
                std::map<std::string, std::string> tmp;
                keywords.emplace_back(tmp);
            }
        }
        
        keywords[index]["label"] = state.to_string();
        /*
        for (const auto& p: keywords[index]) {
            std::cout << p.first << " " << p.second << std::endl;
        }
        */
        plt::plot(time_vec, probs.row(index), keywords[index]);
        //plt::plot(time_vec, state_probs);
        state_index++;
    }

    plt::legend();
}

void matplotlib::rho_subdiag_to_plot(const Evolution::Probs& probs,
                                     const std::vector<double>& time_vec,
                                     const std::set<State>& basis,
                                     std::vector<std::map<std::string, std::string>> keywords) {
    size_t from = 0;
    size_t to = 0;
    size_t basis_size = basis.size();
    for (const auto& state_from: basis) {
        if (state_from.get_index() == 0) {
            from++;
            continue;
        }

        for (const auto& state_to: basis) {
            size_t index = from * basis_size + to;
            if (state_to.get_index() == 0 or from >= to or probs[index][0] == -1) {
                to++;
                continue;
            }
            if (keywords.size() <= index) {
                while(keywords.size() <= index) {
                    std::map<std::string, std::string> tmp;
                    keywords.emplace_back(tmp);
                }
            }

            keywords[index]["label"] = state_from.to_string() + " -> " + state_to.to_string();
            /*
            for (const auto& p: keywords[index]) {
                std::cout << p.first << " " << p.second << std::endl;
            }
            */
            plt::plot(time_vec, probs.row(index), keywords[index]);
            //plt::plot(time_vec, state_probs);
            to++;
        }
        from++;
        to = 0;
    }
    plt::legend();
}

void matplotlib::show(bool is_block) {
    plt::show(is_block);
}

void matplotlib::make_figure(size_t x, size_t y, size_t dpi) {
    if (x == 0 or y == 0) plt::figure();
    else {
        std::cout << "START_MAKE_FIGURE\n";
        plt::figure_size(x, y, dpi);
        std::cout << "END_MAKE_FIGURE\n";
    }
}

void matplotlib::savefig(const std::string& filename, size_t dpi) {
    plt::save(filename, dpi);
}

void matplotlib::grid(bool is_enable) {
    plt::grid(is_enable);
}

#endif
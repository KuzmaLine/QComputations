#include "QComputations_SINGLE.hpp"
#include <iostream>
#include <regex>
#include <complex>

using COMPLEX = std::complex<double>;

int main(int argc, char** argv) {
    using namespace QComputations;
    QConfig::instance().set_width(30);
    double h = QConfig::instance().h();
    double w = QConfig::instance().w();
    QConfig::instance().set_g(2);
    using OpType = Operator<TCH_State>;

    std::vector<size_t> grid_config = {1, 1};

    TCH_State grid(grid_config);
    grid.set_n(1, 0);
    grid.set_waveguide(0, 1, 1, 0);
    grid.set_leak_for_cavity(1, 15);

    auto time_vec = linspace(0, 10, 30000);

    matplotlib::make_figure(1920, 1080);

    //std::vector<double> terms = {0, 0.05, 0.1, 0.15, 0.25, 0.5};
    std::vector<double> terms = {0, 1, 5, 15};
    //for (auto term: terms) {

    for (size_t i = 0; i < terms.size(); i++) {
        auto term = terms[i];
        auto A_term_func = std::function<State<TCH_State>(const TCH_State&)>{[term](const TCH_State& state){
            COMPLEX res = 0;
            for (int i = 0; i < state.cavities_count(); i++) {
                res += state.get_qudit(1, i);
            }

            return State<TCH_State>(state, res*term);
        }};

        auto A_term = OpType(A_term_func);
        std::vector<std::pair<double, OpType>> dec;
        dec.emplace_back(1, A_term);

        H_TCH H(grid, dec);

        show_basis(H.get_basis());
        H.show();
        std::cout << std::endl;
        auto mass = H.get_decoherence();
        mass[0].second.show();

        std::map<std::string, std::string> keywords;
        keywords["label"] = "gamma = " + std::to_string(term);
        auto probs = quantum_master_equation(grid, H, time_vec);
        auto prob_sink = probs.row(H.size() - 1);

        matplotlib::plot(time_vec, prob_sink, keywords);
    }

    matplotlib::xlabel("time");
    matplotlib::ylabel("p_sink");
    matplotlib::grid();
    matplotlib::savefig("DAT.png");
    matplotlib::show();

    return 0;
}

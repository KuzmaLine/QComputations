#include "QComputations_SINGLE.hpp"

using namespace QComputations;

constexpr int max_photons = 1;

int main() {
    QConfig::instance().set_width(20); // Ширина ячейки элемента матрицы для stdout
    double h = QConfig::instance().h(); // Получить постоянную планка
    double w = QConfig::instance().w(); // Получить частоту
    QConfig::instance().set_g(0.005); // сила взаимодействия с полем атома
    QConfig::instance().set_max_photons(max_photons);

    std::vector<size_t> grid_config = {1, 1};

    TCH_State state(grid_config);
    state.set_n(QConfig::instance().max_photons(), 0);
    state.set_waveguide(0, 1, 0.01);
    state.set_leak_for_cavity(1, 0.2);
    
    H_TCH H(state);

    H.show();

    CSR_H_TCH h_csr(state);

    h_csr.show();

    return 0;
}
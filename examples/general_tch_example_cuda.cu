/*
Одноядерная версия

Пример демонстрирует типичную работу с уже готовыми моделями на примере 
модели Тависа-Каммингса-Хаббарда с 2 способами визуализации. Способ визуализации 
выбирается с помощью переменной is_python_api.
True - визуализация с помощью встроенного PythonAPI
False - генерируются CSV файлы в папку general_tch_example_csv, потом они обрабатываются 
с помощью скрипта $SEABORN_PLOT. Конфигуратор в $SEABORN_CONFIG.

Моделируется система с 2 полостями по 1 электрону в каждой с начальным состоянием
|1;0>|0;0> с фактором декогеренции утечки фотонов из 1 полости с интенсивность 0.2.

Базис здесь очень маленький, поэтому прогонять рекомендую на 1 ядре, иначе будет замедление.
*/

#include "QComputations_CUDA_NO_PLOTS.hpp"
#include <iostream>
#include <regex>
#include <complex>

constexpr int max_photons = 2;

using COMPLEX = std::complex<double>;

int main(int argc, char** argv) {
    using namespace QComputations;

    QConfig::instance().set_width(30); // Ширина ячейки элемента матрицы для stdout
    double h = QConfig::instance().h(); // Получить постоянную планка
    double w = QConfig::instance().w(); // Получить частоту
    QConfig::instance().set_g(0.005); // сила взаимодействия с полем атома
    QConfig::instance().set_max_photons(max_photons);

    std::vector<size_t> grid_config = {1, 1};

    TCH_State state(grid_config);
    state.set_n(max_photons, 0);
    state.set_waveguide(0, 1, 0.01);
    // state.set_leak_for_cavity(1, 0.2);

    cublasHandle_t handle;
    cublasCreate(&handle);

    CUDA_H_TCH H(handle, state);

    //show_basis(H.get_basis());

    //H.show();
    std::cout << H.size() << std::endl;

    auto time_vec = linspace(0, 1000, 1000);
    // auto init_state = State<TCH_State>(state, std::sqrt(0.5));
    auto init_state = State<TCH_State>(state);
    // state.set_atom(0, 0);
    // state.set_atom(1, 1);
    // init_state += state;
    // init_state[state] = -std::sqrt(0.5); 
    // auto probs = quantum_master_equation(init_state.fit_to_basis_state(H.get_basis()).get_vector(), H, time_vec);
    auto probs = schrodinger(State<Basis_State>(state), H, time_vec);

    make_probs_files(H.to_cpu(), probs, time_vec, H.get_basis(), "general_tch_plots/CUDA_CSV_FILES");
    //}

    cublasDestroy(handle);
    return 0;
}
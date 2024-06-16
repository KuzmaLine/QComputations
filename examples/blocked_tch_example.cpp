/*
Блочная версия

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

#include "QComputations_CPU_CLUSTER.hpp"
#include <iostream>
#include <regex>
#include <complex>

constexpr bool is_python_api = true;

using COMPLEX = std::complex<double>;

int main(int argc, char** argv) {
    using namespace QComputations;
    MPI_Init(&argc, &argv);

    QConfig::instance().set_width(30); // Ширина ячейки элемента матрицы для stdout
    double h = QConfig::instance().h(); // Получить постоянную планка
    double w = QConfig::instance().w(); // Получить частоту
    QConfig::instance().set_g(0.005); // сила взаимодействия с полем атома

    std::vector<size_t> grid_config = {1, 1};

    TCH_State state(grid_config);
    state.set_n(1, 0);
    state.set_waveguide(0, 1, 0.01);
    state.set_leak_for_cavity(1, 0.2);

    int ctxt;
    mpi::init_grid(ctxt);
    
    BLOCKED_H_TCH H(ctxt, state);

    if (is_main_proc()) { show_basis(H.get_basis()); }

    H.show();

    auto time_vec = linspace(0, 5000, 5000);

    auto probs = quantum_master_equation(State<Basis_State>(state), H, time_vec);

    if (is_python_api) {
        if (is_main_proc()) {
            matplotlib::make_figure(1900, 1000, 80);
            matplotlib::grid();
            matplotlib::title("PYTHON_API_PLOT");
            matplotlib::xlabel("time");
            matplotlib::ylabel("Probability");
        }

        matplotlib::probs_to_plot(probs, time_vec, H.get_basis());

        if (is_main_proc()) {
            matplotlib::savefig("general_tch_plots/python_api_result.png");
            matplotlib::show();
        }
    } else {
        make_probs_files(H, probs, time_vec, H.get_basis(), "general_tch_plots/CSV_FILES");
    }

    MPI_Finalize();
    return 0;
}
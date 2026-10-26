// Кросс-бэкендная эквивалентность (шаблон).
//
// Идея: одна и та же физическая задача (ДЕТЕРМИНИРОВАННЫЕ начальные условия,
// без случайных чисел) прогоняется на двух бэкендах; сравниваем НЕ сырые
// векторы (порядок базиса у бэкендов может отличаться), а инварианты:
// вероятности (to_probs) / expectation values.
//
// Включается в сборку только если найдены prebuilt-библиотеки (install.sh) —
// см. tests/CMakeLists.txt. Заполните TODO-блоки своим солювером.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <complex>
#include <vector>

using Catch::Approx;
using C = std::complex<double>;

// TODO: include нужных бэкендов (single / blocked / csr / cuda)
// #include "QComputations_SINGLE.hpp"
// #include "QComputations_BLOCKED.hpp"

namespace {

// Запуск задачи на заданном бэкенде -> вектор вероятностей (инвариант порядка)
// TODO: подставить вызов вашего динамического солювера.
std::vector<double> run_problem(/* backend, params */)
{
    // 1) зафиксировать задачу: grid_config TCH, H, t, dt — без случайных чисел;
    // 2) пропустить через динамический солювер;
    // 3) вернуть psi.to_probs() (или expectation values), а НЕ state_vec_.
    return {};
}

} // namespace

TEST_CASE("backends: эквивалентность по вероятностям", "[integration][backends]")
{
    const std::vector<double> p_a = run_problem(/* backend A, напр. single/CPU */);
    const std::vector<double> p_b = run_problem(/* backend B, напр. blocked/CUDA */);

    REQUIRE(p_a.size() == p_b.size());
    for (size_t i = 0; i < p_a.size(); ++i)
    {
        // относительная точность: 1e-9 — для double; 1e-4 — для float
        REQUIRE(p_a[i] == Approx(p_b[i]).epsilon(1e-9));
    }
}

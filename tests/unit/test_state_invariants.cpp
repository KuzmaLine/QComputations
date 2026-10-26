// Числовые инварианты суперпозиций — примеры Approx-сравнений для квантовых
// амплитуд (норма, вероятности, ортогональность, масштабирование).
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "state.hpp"

#include <complex>
#include <numeric>

using namespace QComputations;
using Catch::Approx;
using C = std::complex<double>;

TEST_CASE("State: числовые инварианты", "[unit][state][numeric]")
{
    Basis_State a(2, 1); a.set_qudit(1, 0);
    Basis_State b(2, 1); b.set_qudit(1, 1);
    Basis_State c(2, 1); // [0,0] — конфигурация, ортогональная и a, и b

    // Нормализованное состояние: |psi> = 0.6|a> + 0.8|b>,  <psi|psi> = 0.36+0.64 = 1
    State<Basis_State> psi;
    psi.insert(a, C(0.6, 0.0));
    psi.insert(b, C(0.8, 0.0));

    REQUIRE(scalar_product(psi, psi).real() == Approx(1.0).margin(1e-12));

    SECTION("сумма вероятностей = 1")
    {
        const auto probs = psi.to_probs();
        const double s = std::accumulate(probs.begin(), probs.end(), 0.0);
        REQUIRE(s == Approx(1.0).margin(1e-12));
        // порядок вероятностей соответствует отсортированному базису
        REQUIRE(probs.size() == psi.get_basis().size());
    }

    SECTION("ортогональные состояния -> скалярное произведение = 0")
    {
        State<Basis_State> phi;
        phi.insert(c, C(1.0, 0.0));
        REQUIRE(std::abs(scalar_product(psi, phi)) < 1e-12);
    }

    SECTION("масштабирование: ||c*psi||^2 = |c|^2 * ||psi||^2")
    {
        const C k(2.0, 0.0);
        const State<Basis_State> scaled = psi * k;
        REQUIRE(scalar_product(scaled, scaled).real() == Approx(4.0).margin(1e-12));
    }

    SECTION("оператор += расширяет базис согласованно")
    {
        State<Basis_State> psi2 = psi;
        State<Basis_State> extra;
        extra.insert(c, C(0.3, 0.0));
        psi2 += extra;
        // c ортогонально: норма^2 растёт на 0.3^2
        REQUIRE(scalar_product(psi2, psi2).real() == Approx(1.09).margin(1e-12));
    }
}

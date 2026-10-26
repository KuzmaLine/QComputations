// Юнит-тесты API класса State: clone / swap / erase / итераторы.
// (Конвертация из ad-hoc test_new_api.cpp в Catch2.)
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "state.hpp"

#include <complex>

using namespace QComputations;
using Catch::Approx;
using C = std::complex<double>;

TEST_CASE("State: clone/swap/erase/итераторы", "[unit][state]")
{
    // два базисных состояния: 2 кудита, max_val=1
    Basis_State a(2, 1); a.set_qudit(1, 0);
    Basis_State b(2, 1); b.set_qudit(1, 1);

    State<Basis_State> psi;
    psi.insert(a, C(1.0, 0.0));
    psi.insert(b, C(0.0, 1.0));
    const size_t n0 = psi.size();
    REQUIRE(n0 == 2);

    SECTION("clone — независимая копия")
    {
        auto p = psi.clone();
        REQUIRE(p != nullptr);
        REQUIRE(p->size() == n0);
        REQUIRE(scalar_product(*p, *p).real() == Approx(2.0).margin(1e-12));

        // независимость: обнуление амплитуд копии не трогает оригинал
        p->set_zero();
        REQUIRE(scalar_product(*p, *p).real() == Approx(0.0).margin(1e-15));
        REQUIRE(scalar_product(psi, psi).real() == Approx(2.0).margin(1e-12));
    }

    SECTION("итераторы — range-for и begin/end/cbegin/cend")
    {
        int cnt = 0;
        for (const auto& pr : psi)
        {
            REQUIRE(pr.first != nullptr);
            (void)pr.second;
            ++cnt;
        }
        REQUIRE(cnt == (int)psi.size());

        auto it = psi.begin();
        const auto en = psi.end();
        REQUIRE(it != en);
        ++it;
        REQUIRE(psi.cbegin() != psi.cend());
    }

    SECTION("swap — member и free (ADL)")
    {
        State<Basis_State> q;
        q.insert(b, C(2.0, 0.0));
        const size_t qsize = q.size();

        psi.swap(q);
        REQUIRE(psi.size() == qsize);
        REQUIRE(q.size() == n0);

        // свободный swap через ADL
        State<Basis_State> s1, s2;
        s1.insert(a, C(1.0, 0.0));
        swap(s1, s2);
        REQUIRE(s1.is_empty());
        REQUIRE(s2.size() == 1);
    }

    SECTION("erase — удаление по значению и по shared_ptr")
    {
        State<Basis_State> r;
        r.insert(a, C(1.0, 0.0));
        r.insert(b, C(2.0, 0.0));
        const size_t r0 = r.size();

        REQUIRE(r.erase(a) == 1);
        REQUIRE(r.size() == r0 - 1);
        REQUIRE(r.erase(a) == 0);          // уже удалено
        REQUIRE(!r.is_in_state(a));
        REQUIRE(r.is_in_state(b));          // b не задето

        auto sp = std::make_shared<Basis_State>(b);
        REQUIRE(r.erase(sp) == 1);
        REQUIRE(r.is_empty());
    }
}

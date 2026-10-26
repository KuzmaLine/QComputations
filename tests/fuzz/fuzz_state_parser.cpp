// Фаззинг строкового парсера Basis_State (зона «//!!!!» в state.hpp).
//
// Таргет проверяет:
//  1) строковый конструктор не падает на произвольных байтах;
//  2) round-trip: to_string -> строковый конструктор -> те же qudits;
//  3) round-trip через set_state (только для одногруппных состояний —
//     flat-разбор set_state покрывает их; многогруппные — отдельный таргет
//     для длинного прогона).
//
// Сборка с libFuzzer (clang):
//   clang++ -std=c++17 -fsanitize=fuzzer,address -fno-omit-frame-pointer \
//     -I src/QComputations tests/fuzz/fuzz_state_parser.cpp \
//     src/QComputations/state.cpp src/QComputations/functions.cpp -o qc_fuzz
//   ./qc_fuzz -runs=200000 tests/fuzz/corpus
//
// Обычная сборка (g++, для CI-смока, без libFuzzer):
//   main() прогоняет таргет на детерминированных seeds и на файле из argv[1].
#include "state.hpp"

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Контракт парсера: «строка состояния» = цифры + '| ; >'.
// Произвольные байты фильтруем до запуска парсера (это и есть входная
// область таргета, а не баг парсера).
static inline std::string filter_state_string(const uint8_t* data, std::size_t size)
{
    std::string s;
    s.reserve(size);
    for (std::size_t i = 0; i < size; ++i)
    {
        const char c = static_cast<char>(data[i]);
        if (std::isdigit(static_cast<unsigned char>(c)) || c == '|' || c == ';' || c == '>')
            s += c;
    }
    return s;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, std::size_t size)
{
    const std::string s = filter_state_string(data, size);
    try
    {
        // 1) строковый конструктор; max_val=99 — терпимо к двум цифрам
        QComputations::Basis_State st(s, 99);

        // 2) round-trip через to_string
        const std::string s2 = st.to_string();
        QComputations::Basis_State st2(s2, 99);
        if (st.qudits() != st2.qudits())
            return 1; // баг: round-trip не тождествен

        // 3) set_state round-trip — только для одногруппных состояний
        if (s2.size() > 0 && st.get_groups_count() == 1)
        {
            QComputations::Basis_State st3(st.qudits().size(), 99);
            st3.set_state(s2);
            if (st3.qudits() != st.qudits())
                return 1;
        }
    }
    catch (...)
    {
        // исключения (out_of_range и т.п.) — ожидаемое поведение, не баг
    }
    return 0;
}

#ifndef LIBFUZZER
int main(int argc, char** argv)
{
    using namespace std;
    auto run = [](const string& s) -> int {
        return LLVMFuzzerTestOneInput(reinterpret_cast<const uint8_t*>(s.data()), s.size());
    };

    // детерминированные seeds (одногруппные, корректные)
    const vector<string> seeds = {
        "",        // пустое
        ";1;0>",   // обычная группа
        "|1;0>",   // с '|'
        ";12;3>",  // многозначные кудиты
        // ";1;0>;2;3>" — многогруппное; оставлено для длинного fuzz-прогона
        // (flat-разбор set_state его не покрывает — сознательный scope таргета)
    };
    int bad = 0;
    for (const auto& s : seeds)
        bad += (run(s) != 0);

    // файл из argv (для ctest-смока)
    if (argc > 1)
    {
        ifstream f(argv[1]);
        if (!f)
        {
            cout << "FUZZ SMOKE: cannot open corpus file: " << argv[1] << "\n";
            return 1;
        }
        string content((istreambuf_iterator<char>(f)), istreambuf_iterator<char>());
        bad += (run(content) != 0);
        cout << "corpus file: " << argv[1] << " (" << content.size() << " bytes)\n";
    }

    if (bad)
    {
        cout << "FUZZ SMOKE: " << bad << " case(s) FAILED\n";
        return 1;
    }
    cout << "FUZZ SMOKE: all cases passed\n";
    return 0;
}
#endif

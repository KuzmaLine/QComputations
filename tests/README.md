# QComputations — автотесты (каркас)

Стек: **Catch2 v3** (юнит + Approx для чисел) + **CTest** (раннер, лейблы)
+ **libFuzzer** (парсер) + **ASan/UBSan** (отдельная сборка).

## Быстрый старт

```bash
cmake -S tests -B tests/build
cmake --build tests/build -j4
ctest --test-dir tests/build --output-on-failure
```

## Селекция по лейблам

```bash
ctest --test-dir tests/build -L unit          # быстро, без полной библиотеки
ctest --test-dir tests/build -L integration   # тяжёлые (нужны .so из install.sh)
ctest --test-dir tests/build -L fuzz          # fuzz-смок парсера
```

## Sanitizers

```bash
cmake -S tests -B tests/build-asan -DQC_TESTS_ASAN=ON
cmake --build tests/build-asan -j4
ctest --test-dir tests/build-asan -L unit     # ASan+UBSan на юнит-тестах
```

## Fuzzing (длинный прогон, clang)

```bash
clang++ -std=c++17 -fsanitize=fuzzer,address -fno-omit-frame-pointer \
  -I src/QComputations tests/fuzz/fuzz_state_parser.cpp \
  src/QComputations/state.cpp src/QComputations/functions.cpp -o qc_fuzz
./qc_fuzz -runs=200000 tests/fuzz/corpus
```

## Обвязка существующих ad-hoc тестов

См. блок в конце `tests/CMakeLists.txt`: ваши `main()`-программы
(`examples/Grover_example.cpp`, `fft_test.cpp`, ...) добавляются в ctest
БЕЗ переписывания — через `add_test` + `PASS_REGULAR_EXPRESSION`
(строка успеха в выводе) или просто по exit-code.

## Структура

```
tests/
  CMakeLists.txt            # каркас (всё выше)
  unit/
    test_state_api.cpp      # clone/swap/erase/итераторы (из test_new_api.cpp)
    test_state_invariants.cpp # норма/вероятности/ортогональность (Approx)
  integration/
    test_backends_equivalence.cpp  # шаблон: сравнение бэкендов по вероятностям
  fuzz/
    fuzz_state_parser.cpp   # libFuzzer-таргет парсера + CI-смок без libFuzzer
    corpus/sample_state.txt # seed-корпус
```

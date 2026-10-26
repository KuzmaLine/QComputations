# =============================================================================
# QComputations — автотесты (каркас)
#
# Сборка и запуск (из корня репо):
#   cmake -S tests -B tests/build
#   cmake --build tests/build -j4
#   ctest --test-dir tests/build --output-on-failure
#
# Селекция по меткам:
#   ctest --test-dir tests/build -L unit          # юнит-тесты (быстро)
#   ctest --test-dir tests/build -L integration   # тяжёлые интеграционные
#   ctest --test-dir tests/build -L fuzz          # fuzz-смок парсера
#
# Sanitizers (ASan+UBSan):
#   cmake -S tests -B tests/build-asan -DQC_TESTS_ASAN=ON && cmake --build tests/build-asan
#
# Fuzzing с libFuzzer (clang, длинный прогон):
#   clang++ -std=c++17 -fsanitize=fuzzer,address -fno-omit-frame-pointer \
#     -I src/QComputations tests/fuzz/fuzz_state_parser.cpp \
#     src/QComputations/state.cpp src/QComputations/functions.cpp -o qc_fuzz
#   ./qc_fuzz -runs=200000 tests/fuzz/corpus   # corpus как seed-директория
# =============================================================================
cmake_minimum_required(VERSION 3.20)
project(qc_tests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# tests/ лежит в корне репо, рядом с src/
set(QC_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src/QComputations")

# ------------------------------ Catch2 v3 -----------------------------------
# Header-only фреймворк без зависимостей; Approx — для численных сравнений.
# На CMake >= 3.24 можно добавить DOWNLOAD_EXTRACT_TIMESTAMP TRUE;
# на 3.22 (Ubuntu 22.04) этот ключ НЕ поддерживается и ломает список URL.
# PROPERTIES LABELS + ADD_TAGS_AS_LABELS доступны в Catch >= v3.8
# (v3.5 не переносил LABELS на ctest — отсюда актуальная версия ниже).
include(FetchContent)
FetchContent_Declare(Catch2
    URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.16.1.tar.gz
)
FetchContent_MakeAvailable(Catch2)

list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
include(Catch)
enable_testing()

# -------------------------- Sanitizers (опц.) -------------------------------
option(QC_TESTS_ASAN "Собрать тесты с ASan+UBSan" OFF)
if(QC_TESTS_ASAN)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()

# ------------- unit: API класса State + числовые инварианты ----------------
# Компилируются только state.cpp + functions.cpp — быстрые юнит-тесты
# без полной библиотеки (без MKL/CUDA).
add_executable(qc_test_unit
    unit/test_state_api.cpp
    unit/test_state_invariants.cpp
    ${QC_SRC_DIR}/state.cpp
    ${QC_SRC_DIR}/functions.cpp
)
target_include_directories(qc_test_unit PRIVATE ${QC_SRC_DIR})
target_link_libraries(qc_test_unit PRIVATE Catch2::Catch2WithMain)
catch_discover_tests(qc_test_unit TEST_PREFIX "unit." ADD_TAGS_AS_LABELS
                     PROPERTIES LABELS "unit")

# ------------- fuzz: парсер состояний (смок без libFuzzer) -----------------
add_executable(qc_fuzz_parser
    fuzz/fuzz_state_parser.cpp
    ${QC_SRC_DIR}/state.cpp
    ${QC_SRC_DIR}/functions.cpp
)
target_include_directories(qc_fuzz_parser PRIVATE ${QC_SRC_DIR})
add_test(NAME fuzz.parser_smoke
         COMMAND qc_fuzz_parser ${CMAKE_CURRENT_SOURCE_DIR}/fuzz/corpus/sample_state.txt)
set_tests_properties(fuzz.parser_smoke PROPERTIES LABELS "fuzz")

# ------------- integration: эквивалентность бэкендов ------------------------
# Нужны prebuilt-библиотеки (install.sh). Если их нет — пропускаем честно.
find_library(QC_SINGLE_LIB  NAMES QComputations_single  HINTS /usr/local/lib)
find_library(QC_BLOCKED_LIB NAMES QComputations_BLOCKED HINTS /usr/local/lib)
if(QC_SINGLE_LIB AND QC_BLOCKED_LIB)
    add_executable(qc_test_backends integration/test_backends_equivalence.cpp)
    target_include_directories(qc_test_backends PRIVATE ${QC_SRC_DIR})
    target_link_libraries(qc_test_backends PRIVATE Catch2::Catch2WithMain
                        ${QC_SINGLE_LIB} ${QC_BLOCKED_LIB})
    catch_discover_tests(qc_test_backends TEST_PREFIX "backend." ADD_TAGS_AS_LABELS
                         PROPERTIES LABELS "integration;cpu")
    message(STATUS "qc_tests: backend-equivalence test ENABLED")
else()
    message(STATUS "qc_tests: SKIP backend test (нет prebuilt QComputations .so; см. install.sh)")
endif()

# ------------- обвязка существующих ad-hoc тестов ---------------------------
# Паттерн БЕЗ переписывания: ваши main()-программы (examples/Grover_example.cpp,
# fft_test.cpp, ...) запускаются как ctest-тесты; успех — по строке в выводе
# (PASS_REGULAR_EXPRESSION) или по exit-code.
#
# Вариант А (рекомендуемый — линковка против .so из install.sh):
#
#   add_executable(qc_adhoc_grover ../examples/Grover_example.cpp)
#   target_include_directories(qc_adhoc_grover PRIVATE ${QC_SRC_DIR})
#   target_link_libraries(qc_adhoc_grover PRIVATE ${QC_SINGLE_LIB})
#   add_test(NAME adhoc.grover COMMAND qc_adhoc_grover)
#   set_tests_properties(adhoc.grover PROPERTIES
#       PASS_REGULAR_EXPRESSION "passed|OK|0 errors"   # под свою строку успеха
#       LABELS "integration slow")
#
# Вариант Б (компиляция исходников библиотеки в целевой файл — подставьте
# список источников своей конфигурации сборки):
#
#   add_executable(qc_adhoc_fft ../examples/tests/fft_test.cpp
#       ${QC_SRC_DIR}/hamiltonian.cpp ${QC_SRC_DIR}/dynamic.cpp
#       ${QC_SRC_DIR}/matrix.cpp ${QC_SRC_DIR}/functions.cpp
#       ${QC_SRC_DIR}/state.cpp ${QC_SRC_DIR}/quantum_operators.cpp)
#   target_include_directories(qc_adhoc_fft PRIVATE ${QC_SRC_DIR})
#   add_test(NAME adhoc.fft COMMAND qc_adhoc_fft)
#   set_tests_properties(adhoc.fft PROPERTIES LABELS "integration slow")

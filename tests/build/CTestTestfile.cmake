# CMake generated Testfile for 
# Source directory: /home/kuzmaline/Quantum/QComputations/QComputations/tests
# Build directory: /home/kuzmaline/Quantum/QComputations/QComputations/tests/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
include("/home/kuzmaline/Quantum/QComputations/QComputations/tests/build/qc_test_unit-b12d07c_include.cmake")
add_test([=[fuzz.parser_smoke]=] "/home/kuzmaline/Quantum/QComputations/QComputations/tests/build/qc_fuzz_parser" "/home/kuzmaline/Quantum/QComputations/QComputations/tests/fuzz/corpus/sample_state.txt")
set_tests_properties([=[fuzz.parser_smoke]=] PROPERTIES  LABELS "fuzz" _BACKTRACE_TRIPLES "/home/kuzmaline/Quantum/QComputations/QComputations/tests/CMakeLists.txt;76;add_test;/home/kuzmaline/Quantum/QComputations/QComputations/tests/CMakeLists.txt;0;")
subdirs("_deps/catch2-build")

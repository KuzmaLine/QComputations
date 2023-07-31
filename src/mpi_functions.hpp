#pragma once

#ifdef ENABLE_MPI

#include <mpi.h>
#include <iostream>
#include <complex>
#include "state.hpp"
#include <map>
#include <functional>

namespace {
    using COMPLEX = std::complex<double>;
}

// COMMAND LIST
namespace COMMAND {
    constexpr int COMMANDS_COUNT = 5;

    constexpr int STOP = 0;
    constexpr int GENERATE_H = 1;
    constexpr int GENERATE_H_FUNC = 2;
    constexpr int SCHRODINGER = 3;
    constexpr int QME = 4;
}

namespace mpi {

    struct MPI_Data {
        size_t n;
        std::function<COMPLEX(size_t, size_t)> func;
        State state;
        std::vector<double> timeline;
    };

    constexpr int ROOT_ID = 0;

    // Send command to other process
    void make_command(int command);

    std::vector<COMPLEX> bcast_vector_complex(const std::vector<COMPLEX>& v = {});
    std::vector<double> bcast_vector_double(const std::vector<double>& v = {});
    State bcast_state(const State& state = State());

    // Stay to wait all MPI process until root process give commands. MPI_Init included
    void run_mpi_slaves(const std::map<int, std::vector<MPI_Data>>& data); 

    // Stop MPI. MPI_Finalize() included
    void stop_mpi_slaves();
}

#endif // ENABLE_MPI
#pragma once

#ifdef ENABLE_MPI

#include <mpi.h>
#include <iostream>
#include <complex>
#include "state.hpp"

namespace {
    using COMPLEX = std::complex<double>;
}

// COMMAND LIST
namespace COMMAND {
    constexpr int STOP = 0;
    constexpr int GENERATE_H = 1;
    constexpr int SCHRODINGER = 2;
    constexpr int QME = 3;
}

namespace mpi {
    constexpr int ROOT_ID = 0;

    // Send command to other process
    void make_command(int command);

    State bcast_state(const State& state);

    // Stay to wait all MPI process until root process give commands. MPI_Init included
    void run_mpi_slaves(); 

    // Stop MPI. MPI_Finalize() included
    void stop_mpi_slaves();
}

#endif // ENABLE_MPI
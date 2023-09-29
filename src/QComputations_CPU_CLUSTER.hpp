//#define _USE_MATH_DEFINES
#ifndef ENABLE_MATPLOTLIB
#define ENABLE_MATPLOTLIB
#endif

#ifndef ENABLE_MPI
#define ENABLE_MPI
#endif

#ifndef ENABLE_CLUSTER
#define ENABLE_CLUSTER
#endif

#pragma once
#include "functions.hpp"
#include "matrix.hpp"
#include "csr_matrix.hpp"
#include "hamiltonian.hpp"
#include "state.hpp"
#include "config.hpp"
#include "dynamic.hpp"
#include "mpi_functions.hpp"
#include "plot.hpp"
#include "test.hpp"
#include "blocked_matrix.hpp"
#include <mpi.h>
#include "hamiltonian_blocked.hpp"
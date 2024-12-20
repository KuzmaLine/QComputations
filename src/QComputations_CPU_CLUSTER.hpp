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

#ifndef ENABLE_ONEAPI
#define ENABLE_ONEAPI
#endif

#pragma once
#include <mpi.h>

#include "QComputations/blocked_matrix.hpp"
#include "QComputations/config.hpp"
#include "QComputations/csr_matrix.hpp"
#include "QComputations/dynamic.hpp"
#include "QComputations/functions.hpp"
#include "QComputations/graph.hpp"
#include "QComputations/hamiltonian.hpp"
#include "QComputations/hamiltonian_blocked.hpp"
#include "QComputations/matrix.hpp"
#include "QComputations/mpi_functions.hpp"
#include "QComputations/plot.hpp"
#include "QComputations/quantum_operators.hpp"
#include "QComputations/state.hpp"
#include "QComputations/test.hpp"

//#define _USE_MATH_DEFINES
#include "functions.hpp"
#include "matrix.hpp"
#include "csr_matrix.hpp"
#include "hamiltonian.hpp"
#include "state.hpp"
//#include "test.hpp"
#include "config.hpp"
#include "dynamic.hpp"

#ifdef ENABLE_MPI
#include "mpi_functions.hpp"
#endif

#ifdef ENABLE_MATPLOTLIB
#include "plot.hpp"
namespace plt = matplotlibcpp;
#endif

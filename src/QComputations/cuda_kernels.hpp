#ifdef __CUDACC__

#include <iostream>
#include "matrix.hpp"
#include <cublas_v2.h>
#include <typeinfo>
#include <cstdio>

namespace QComputations {

namespace CUDA {

__global__ void compute_psi_and_probs_kernel(
    double* probs,
    const cuDoubleComplex* eigenvectors,
    const cuDoubleComplex* lambda,
    const double* eigenvalues,
    const double* time_vec,
    double hbar,
    ILP_TYPE n,
    ILP_TYPE num_times
);

}

}

#endif
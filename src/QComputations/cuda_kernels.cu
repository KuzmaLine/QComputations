#ifdef __CUDACC__

#include "cuda_kernels.hpp"

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
) {
    ILP_TYPE state_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t x_grid_size = blockDim.x * gridDim.x;
    ILP_TYPE time_idx = blockIdx.y * blockDim.y + threadIdx.y;
    size_t y_grid_size = blockDim.y * gridDim.y;
    
    for (size_t st_idx = state_idx; st_idx < n; st_idx += x_grid_size) { 
        for (size_t j = time_idx; j < num_times; j += y_grid_size) {
            double t = time_vec[j];
            cuDoubleComplex psi_t = make_cuDoubleComplex(0.0, 0.0);
            
            // Calcs psi(t) = sum_i lambda[i] * exp(-i*E_i*t/hbar) * phi_i
            for (ILP_TYPE i = 0; i < n; i++) {
                // lambda[i] = <phi_i|psi0>
                cuDoubleComplex lambda_i = lambda[i];
                
                // exp(-i*E_i*t/hbar) = cos(-E_i*t/hbar) + i*sin(-E_i*t/hbar)
                double arg = -eigenvalues[i] * t / hbar;
                cuDoubleComplex exp_factor = make_cuDoubleComplex(cos(arg), sin(arg));
                
                // lambda_i * exp_factor
                cuDoubleComplex lambda_exp = make_cuDoubleComplex(
                    lambda_i.x * exp_factor.x - lambda_i.y * exp_factor.y,
                    lambda_i.x * exp_factor.y + lambda_i.y * exp_factor.x
                );

                cuDoubleComplex phi_i_state = eigenvectors[st_idx + i * n];
                
                // (lambda_i * exp_factor) * phi_i_state
                psi_t = make_cuDoubleComplex(
                    psi_t.x + (lambda_exp.x * phi_i_state.x - lambda_exp.y * phi_i_state.y),
                    psi_t.y + (lambda_exp.x * phi_i_state.y + lambda_exp.y * phi_i_state.x)
                );
            }

            probs[st_idx * num_times + j] = psi_t.x * psi_t.x + psi_t.y * psi_t.y;
        }
    }
}

}

}

#endif
#ifdef __CUDACC__
#ifdef ENABLE_MPI

#include "blocked_cuda_matrix.hpp"

namespace QComputations {

// Явные инстанциации для используемых типов
template class BLOCKED_CUDA_Matrix<double>;
template class BLOCKED_CUDA_Matrix<COMPLEX>;

} // namespace QComputations

#endif // ENABLE_MPI
#endif // __CUDACC__
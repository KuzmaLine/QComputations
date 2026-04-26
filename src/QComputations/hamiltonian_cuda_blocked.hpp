#ifdef ENABLE_MPI
#ifdef __CUDACC__
#pragma once
#include "graph.hpp"
#include "mpi_functions.hpp"
#include "blocked_matrix.hpp"
#include <complex>
#include <functional>
#include <string>
#include <set>
#include "state.hpp"
#include "config.hpp"
#include "quantum_operators.hpp"
#include "blocked_cuda_matrix.hpp"

namespace QComputations {

namespace {
    using COMPLEX = std::complex<double>;
}

class BLOCKED_CUDA_Hamiltonian {
    public:
        explicit BLOCKED_CUDA_Hamiltonian() = default;
        size_t n() const { return H_.n(); }
        size_t size() const { return H_.n(); }
        size_t n_loc() const { return H_.local_rows(); }
        size_t m_loc() const { return H_.local_cols(); }
        cublasMpHandle_t handle() const { return H_.handle(); }
        cublasMpGrid_t grid() const { return H_.grid(); }
        std::vector<std::shared_ptr<Basis_State>> get_basis() const { return basis_; }
        std::vector<std::pair<double, BLOCKED_CUDA_Matrix<COMPLEX>>>& get_decoherence() { return decoherence_;}
        const std::vector<std::pair<double, BLOCKED_CUDA_Matrix<COMPLEX>>>& get_decoherence() const { return decoherence_; }
        void get_buffersize_gemm(size_t& dev_size, size_t& host_size) const { H_.get_buffersize_gemm(H_, H_, dev_size, host_size); }
        void get_buffersize_geadd(size_t& dev_size, size_t& host_size) const { H_.get_buffersize_geadd(H_, dev_size, host_size); }
        ncclComm_t nccl_comm() const { return H_.nccl_comm(); }
        // void write_to_csv_file(const std::string& filename) const { H_.write_to_csv_file(filename); }

        // void virtual eigen() {
        //     if (!is_calculated_eigen_) {
        //         auto p = Hermit_Lanczos(H_);
        //         eigenvalues_ = p.first;
        //         eigenvectors_ = p.second;
        //         is_calculated_eigen_ = true;
        //     }
        // }

        // std::vector<double> virtual eigenvalues() {
        //     this->eigen();
        //     return eigenvalues_;
        // }

        // BLOCKED_Matrix<COMPLEX> virtual eigenvectors() {
        //     this->eigen();
        //     return eigenvectors_;
        // }

        //COMPLEX operator() (size_t i, size_t j) const { return H_(i, j); }
        void show(size_t width = QConfig::instance().width()) const { H_.show(width); }
        void print_distributed() const { H_.show_local(); }
        // Matrix<COMPLEX> get_local_matrix() const { return H_.get_local_matrix(); }
        BLOCKED_CUDA_Matrix<COMPLEX>& get_blocked_matrix() { return H_; }
        const BLOCKED_CUDA_Matrix<COMPLEX>& get_blocked_matrix() const { return H_; }
    protected:
        // bool is_calculated_eigen_ = false;
        std::vector<std::shared_ptr<Basis_State>> basis_;
        BLOCKED_CUDA_Matrix<COMPLEX> H_;
        // BLOCKED_CUDA_Matrix<COMPLEX> eigenvectors_;
        // std::vector<double> eigenvalues_;
        std::vector<std::pair<double, BLOCKED_CUDA_Matrix<COMPLEX>>> decoherence_;
        // TCH_State grid_;
};

/*

class BLOCKED_H_TC : public BLOCKED_Hamiltonian {
    public:
        explicit BLOCKED_H_TC(ILP_TYPE ctxt, const State& state);
};

class BLOCKED_H_JC : public BLOCKED_Hamiltonian {
    public:
        explicit BLOCKED_H_JC(ILP_TYPE ctxt, const State& state);
};

class BLOCKED_H_by_func: public BLOCKED_Hamiltonian {
    public:
        explicit BLOCKED_H_by_func(ILP_TYPE ctxt, size_t n, std::function<COMPLEX(size_t, size_t)> func);
    private:
        std::function<COMPLEX(size_t, size_t)> func_;
};
*/

template<typename StateType>
class BLOCKED_CUDA_H_by_Operator: public BLOCKED_CUDA_Hamiltonian {
    public:
        explicit BLOCKED_CUDA_H_by_Operator(MPI_Comm comm, ncclComm_t nccl_comm, cublasMpHandle_t handle, cublasMpGrid_t grid,
                                       const State<StateType>& init_state, const Operator<StateType>& H_op,
                                       const std::vector<std::pair<double, Operator<StateType>>>& decoherence = {});
};

template<typename StateType>
BLOCKED_CUDA_H_by_Operator<StateType>::BLOCKED_CUDA_H_by_Operator(MPI_Comm comm, ncclComm_t nccl_comm, cublasMpHandle_t handle, cublasMpGrid_t grid,
                                                        const State<StateType>& init_state, const Operator<StateType>& H_op,
                                                        const std::vector<std::pair<double, Operator<StateType>>>& decoherence) {
    std::vector<Operator<StateType>> dec_tmp;
    for (const auto& p: decoherence) {
        dec_tmp.push_back(p.second);
    }

    auto basis = State<StateType>(State_Graph<StateType>(init_state, H_op, dec_tmp).get_basis());
    basis.sort();
    auto sorted_basis = basis.get_basis();
    basis_ = convert_to<StateType>(sorted_basis);

    size_t size = basis_.size();


    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    int64_t NB = 64, MB = 64;

    // ILP_TYPE proc_rows, proc_cols, myrow, mycol, NB, MB;
    // mpi::blacs_gridinfo(ctxt, proc_rows, proc_cols, myrow, mycol);

    // NB = size / proc_rows;

    // MB = size / proc_cols;

    // if (NB == 0) {
    //     NB = 1;
    // }

    // if (MB == 0) {
    //     MB = 1;
    // }

    // NB = std::min(NB, MB);
    // MB = NB;


    BLOCKED_CUDA_Matrix<COMPLEX, cuDoubleComplex> H_gpu(comm, nccl_comm, handle, grid,
                                                         size, size, NB, MB);

    int64_t lrows = H_gpu.local_rows();
    int64_t lcols = H_gpu.local_cols();
    std::vector<COMPLEX> local_ham(lrows * lcols, COMPLEX(0.0, 0.0));

    State<StateType> basis_map(basis);
    basis_map.sort();

    for (int64_t j = 0; j < lcols; ++j) {
        int64_t global_col = H_gpu.get_global_col(j);
        auto state_from = sorted_basis[global_col];
        auto res_state = H_op.run(State<StateType>(state_from));
        // std::cout << j << " " << state_from->to_string() << " " << global_col << std::endl;
        // std::cout << res_state.to_string() << std::endl;

        res_state.set_sorted(true);
        for (const auto& p : res_state.state_map()) {
            int64_t global_row = basis_map.get_index(p.first);
            // std::cout <<p.first->to_string() << " " << p.second << " " << global_row << " " << H_gpu.get_local_row(global_row) << std::endl;
            if (H_gpu.is_my_elem_row(global_row)) {
                int64_t local_row = H_gpu.get_local_row(global_row);
                local_ham[local_row + j * lrows] = res_state[p.second];
            }
        }
    }

    cudaMemcpy(H_gpu.local_data(), local_ham.data(),
               lrows * lcols * sizeof(cuDoubleComplex),
               cudaMemcpyHostToDevice);
    H_ = std::move(H_gpu);

    for (const auto& p : decoherence) {
        BLOCKED_CUDA_Matrix<COMPLEX, cuDoubleComplex> A_gpu(comm, nccl_comm, handle, grid,
                                                             size, size, NB, MB);
        std::vector<COMPLEX> local_A(lrows * lcols, COMPLEX(0.0, 0.0));
        for (int64_t j = 0; j < lcols; ++j) {
            int64_t global_col = A_gpu.get_global_col(j);
            auto state_from = sorted_basis[global_col];
            auto res_state = p.second.run(State<StateType>(state_from));

            for (const auto& q : res_state.state_map()) {
                int64_t global_row = basis_map.get_index(q.first);
                if (A_gpu.is_my_elem_row(global_row)) {
                    int64_t local_row = A_gpu.get_local_row(global_row);
                    local_A[local_row + j * lcols] = res_state[q.second];
                }
            }
        }
        cudaMemcpy(A_gpu.local_data(), local_A.data(),
                   lrows * lcols * sizeof(cuDoubleComplex),
                   cudaMemcpyHostToDevice);
        decoherence_.emplace_back(p.first, std::move(A_gpu));
    }
}

// template<typename StateType>
// class BLOCKED_H_by_Scalar_Product: public BLOCKED_Hamiltonian {
//     public:
//         explicit BLOCKED_H_by_Scalar_Product(MPI_Comm comm, ncclComm_t nccl_comm, cublasMpHandle_t handle, cublasMpGrid_t grid,
//                                      const State<StateType>& init_state, 
//                                      const std::function<COMPLEX(const StateType& i, const StateType& j)>& func,
//                                      BasisType<StateType> basis = {});
// };

// template<typename StateType>
// BLOCKED_H_by_Scalar_Product<StateType>::BLOCKED_H_by_Scalar_Product(MPI_Comm comm, ncclComm_t nccl_comm, cublasMpHandle_t handle, cublasMpGrid_t grid,
//                                                                     const State<StateType>& init_state,
//                                                                     const std::function<COMPLEX(const StateType& i, const StateType& j)>& func,
//                                                                     BasisType<StateType> basis) {
//     if (basis.empty()) {
//         auto zero_state = init_state(0);
//         zero_state->set_zero();
//         basis.insert(std::shared_ptr<StateType>(new StateType(*zero_state)));

//         StateType cur_state = *zero_state;

//         for (ILP_TYPE i = 0; i < cur_state.qudits_count(); i++) {
//             if (cur_state.get_max_val(i) > cur_state.get_qudit(i)) {
//                 cur_state.set_qudit(cur_state.get_qudit(i) + 1, i);
//                 basis.insert(std::shared_ptr<StateType>(new StateType(cur_state)));

//                 for (ILP_TYPE j = i - 1; j >= 0; j--) {
//                     cur_state.set_qudit(0, j);
//                 }

//                 i = 0;
//             }
//         }
//     }

//     basis_ = convert_to<StateType>(basis);

//     size_t size = basis_.size();

//     ILP_TYPE proc_rows, proc_cols, myrow, mycol, NB, MB;
//     mpi::blacs_gridinfo(ctxt, proc_rows, proc_cols, myrow, mycol);

//     NB = size / proc_rows;

//     MB = size / proc_cols;

//     if (NB == 0) {
//         NB = 1;
//     }

//     if (MB == 0) {
//         MB = 1;
//     }

//     NB = std::min(NB, MB);
//     MB = NB;

//     auto matrix_func = std::function<COMPLEX(size_t i, size_t j)>{
//         [&func, &basis](size_t i, size_t j) {
//             return func(*get_state_from_basis(basis, i), *get_state_from_basis(basis, j));
//         }
//     };

//     H_ = BLOCKED_CUDA_Matrix<COMPLEX>(ctxt, HE, size, size, matrix_func, NB, MB);
// }

class BLOCKED_CUDA_H_TCH : public BLOCKED_CUDA_H_by_Operator<TCH_State> {
    public:
        explicit BLOCKED_CUDA_H_TCH(MPI_Comm comm, ncclComm_t nccl_comm, cublasMpHandle_t handle, cublasMpGrid_t grid, const State<TCH_State>& state);
};

// /*
// class BLOCKED_H_TCH_EXC: BLOCKED_Hamiltonian {
//     public:
//         explicit BLOCKED_H_TCH_EXC(ILP_TYPE ctxt, const EXC_State& state);
// };
// */

} // namespace QComputations

#endif
#endif

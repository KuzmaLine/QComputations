#ifdef __CUDACC__
#ifdef ENABLE_MPI

#pragma once

#include <mpi.h>
#include <nccl.h>
#include <cublasmp.h>        // официальный заголовок cuBLASMp
#include <cuda_runtime.h>
#include <cublas_v2.h>       // для обычных операций cublas (axpy, scal, copy)

#include <vector>
#include <functional>
#include <complex>
#include <iostream>
#include <iomanip>

#include "matrix.hpp"
#include "config.hpp"

// ---------------------------------------------------------------------------
// Макросы проверки ошибок
// ---------------------------------------------------------------------------
#define NCCL_CHECK(call) \
    do { \
        ncclResult_t res = call; \
        if (res != ncclSuccess) { \
            fprintf(stderr, "NCCL error in %s:%d: %s\n", __FILE__, __LINE__, \
                    ncclGetErrorString(res)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUBLASMP_CHECK(call)                                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        cublasMpStatus_t status = call;                                                                                \
        if (status != CUBLASMP_STATUS_SUCCESS)                                                                         \
        {                                                                                                              \
            fprintf(stderr, "cuBLASMp error at %s:%d : %d\n", __FILE__, __LINE__, status);                             \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

#define CUDA_CHECK(call) \
    do { \
        cudaError_t res = call; \
        if (res != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(res)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t res = call; \
        if (res != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error in %s:%d: %d\n", __FILE__, __LINE__, res); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

namespace QComputations {

using COMPLEX = std::complex<double>;

template <typename T, typename GPU_T = typename std::conditional_t<
    std::is_same<T, COMPLEX>::value, cuDoubleComplex,
    typename std::conditional_t<std::is_same<T, double>::value, double, void>>
>
class BLOCKED_CUDA_Matrix {
public:
    // ---------- конструкторы ----------
    BLOCKED_CUDA_Matrix() = default;

    // Пустая матрица заданных глобальных размеров
    BLOCKED_CUDA_Matrix(MPI_Comm comm, ncclComm_t nccl_comm,
                        cublasMpHandle_t handle, cublasMpGrid_t grid,
                        int64_t n, int64_t m, int64_t nb = 0, int64_t mb = 0,
                        int64_t rsrc = 0, int64_t csrc = 0);

    // Заполнение через функтор (CPU‑лямбда)
    BLOCKED_CUDA_Matrix(MPI_Comm comm, ncclComm_t nccl_comm,
                        cublasMpHandle_t handle, cublasMpGrid_t grid,
                        int64_t n, int64_t m, int64_t nb, int64_t mb,
                        std::function<T(int64_t, int64_t)> func,
                        int64_t rsrc = 0, int64_t csrc = 0);

    ~BLOCKED_CUDA_Matrix();

    // Перемещение
    BLOCKED_CUDA_Matrix(BLOCKED_CUDA_Matrix&& other) noexcept;
    BLOCKED_CUDA_Matrix& operator=(BLOCKED_CUDA_Matrix&& other) noexcept;

    // Копирование запрещено
    BLOCKED_CUDA_Matrix(const BLOCKED_CUDA_Matrix&) = delete;
    BLOCKED_CUDA_Matrix& operator=(const BLOCKED_CUDA_Matrix&) = delete;

    // ---------- линейная алгебра ----------
    BLOCKED_CUDA_Matrix operator*(const BLOCKED_CUDA_Matrix& other) const;
    BLOCKED_CUDA_Matrix operator+(const BLOCKED_CUDA_Matrix& other) const;
    BLOCKED_CUDA_Matrix operator-(const BLOCKED_CUDA_Matrix& other) const;
    BLOCKED_CUDA_Matrix operator*(T scalar) const;
    void operator*=(T scalar);
    void operator+=(const BLOCKED_CUDA_Matrix& other);
    void operator-=(const BLOCKED_CUDA_Matrix& other);

    // ---------- геттеры ----------
    int64_t n() const { return n_; }
    int64_t m() const { return m_; }
    int64_t nb() const { return nb_; }
    int64_t mb() const { return mb_; }
    int64_t local_rows() const { return local_rows_; }
    int64_t local_cols() const { return local_cols_; }
    GPU_T* local_data() { return local_dev_ptr_; }
    const GPU_T* local_data() const { return local_dev_ptr_; }
    cublasMpMatrixDescriptor_t matrix_desc() const { return mat_desc_; }
    cublasMpHandle_t handle() const { return handle_; }
    cublasMpGrid_t grid() const { return grid_; }
    ncclComm_t nccl_comm() const { return nccl_comm_; }
    MPI_Comm comm() const { return comm_; }

    // ---------- индексация ----------
    int64_t get_global_row(int64_t local_row) const;
    int64_t get_global_col(int64_t local_col) const;
    int64_t get_local_row(int64_t global_row) const;
    int64_t get_local_col(int64_t global_col) const;
    bool is_my_elem_row(int64_t global_row) const;
    bool is_my_elem_col(int64_t global_col) const;

    // ---------- сборка и печать ----------
    Matrix<T> gather_to_cpu(int root = 0) const;
    void show_local(size_t width = QConfig::instance().width()) const;
    void show(size_t width = QConfig::instance().width()) const;

    void get_buffersize_gemm(const BLOCKED_CUDA_Matrix<T, GPU_T>& B, const BLOCKED_CUDA_Matrix<T, GPU_T>& C, size_t& dev_size, size_t& host_size) const;
    void get_buffersize_geadd(const BLOCKED_CUDA_Matrix<T, GPU_T>& C, size_t& dev_size, size_t& host_size) const;


private:
    MPI_Comm comm_;
    ncclComm_t nccl_comm_ = nullptr;
    cublasMpHandle_t handle_ = nullptr;   // совместим с cublasHandle_t
    cublasMpGrid_t grid_ = nullptr;
    cublasMpMatrixDescriptor_t mat_desc_ = nullptr;
    GPU_T* local_dev_ptr_ = nullptr;

    int64_t n_ = 0, m_ = 0;
    int64_t nb_ = 0, mb_ = 0;
    int64_t local_rows_ = 0, local_cols_ = 0;

    int my_rank_ = 0;
    int my_grid_row_ = 0, my_grid_col_ = 0;
    int grid_rows_ = 0, grid_cols_ = 0;

    // Внутренние методы
    void init(int64_t rsrc, int64_t csrc);
    void release_resources();
    void copy_to_device(const std::vector<T>& host_data);
    void compute_grid_dims();

    // Локальные операции на GPU (не требуют коммуникаций)
    void local_axpy(T alpha, const BLOCKED_CUDA_Matrix& X);
    void local_scal(T alpha);
    void local_copy(const BLOCKED_CUDA_Matrix& src);
    inline void compute_optimal_blocks() {
        nb_ = (n_ + grid_rows_ - 1) / grid_rows_;   // ceil division
        mb_ = (m_ + grid_cols_ - 1) / grid_cols_;
        // Для унификации можно сделать их одинаковыми, взяв максимум
        int64_t min_block = std::min(nb_, mb_);
        nb_ = min_block;
        mb_ = min_block;
    }
};

template <typename T, typename GPU_T>
static cublasComputeType_t getMpComputeType() {
    if constexpr (std::is_same<T, double>::value) return CUBLAS_COMPUTE_64F;
    else if constexpr (std::is_same<T, COMPLEX>::value) return CUBLAS_COMPUTE_64F;
    else return CUBLAS_COMPUTE_32F;
}

// ---------------------------------------------------------------------------
// Реализация шаблонных методов (должна быть в заголовочном файле)
// ---------------------------------------------------------------------------

template <typename T, typename GPU_T>
void BLOCKED_CUDA_Matrix<T, GPU_T>::compute_grid_dims() {
    int world_size;
    MPI_Comm_size(comm_, &world_size);
    grid_rows_ = (int)std::sqrt(world_size);
    while (world_size % grid_rows_ != 0) --grid_rows_;
    grid_cols_ = world_size / grid_rows_;
}

template<typename T, typename GPU_T>
void BLOCKED_CUDA_Matrix<T, GPU_T>::get_buffersize_geadd(const BLOCKED_CUDA_Matrix<T, GPU_T>& C, size_t& devSize, size_t& hostSize) const {
    GPU_T a{}, b{};                        // значения не важны
    const void* alpha = static_cast<const void*>(&a);
    const void* A     = static_cast<const void*>(this->local_data());
    const void* beta  = static_cast<const void*>(&b);
    void*       C_ptr = const_cast<GPU_T*>(C.local_data());   // приведение const -> void*

    CUBLASMP_CHECK((cublasMpGeadd_bufferSize(
        this->handle(), CUBLAS_OP_N, this->n(), this->m(),
        alpha, A, 1, 1, this->matrix_desc(),
        beta,  C_ptr, 1, 1, C.matrix_desc(),
        &devSize, &hostSize)));
}

template<typename T, typename GPU_T>
void BLOCKED_CUDA_Matrix<T, GPU_T>::get_buffersize_gemm(const BLOCKED_CUDA_Matrix<T, GPU_T>& B, const BLOCKED_CUDA_Matrix<T, GPU_T>& C, size_t& devSize, size_t& hostSize) const {
    GPU_T a{}, b{};
    const void* alpha = static_cast<const void*>(&a);
    const void* A     = static_cast<const void*>(this->local_data());
    const void* B_ptr = static_cast<const void*>(B.local_data());
    const void* beta  = static_cast<const void*>(&b);
    void*       C_ptr = const_cast<GPU_T*>(C.local_data());

    cublasComputeType_t comp = getMpComputeType<T, GPU_T>();

    CUBLASMP_CHECK((cublasMpGemm_bufferSize(
        this->handle(), CUBLAS_OP_N, CUBLAS_OP_N,
        this->n(), B.m(), this->m(),
        alpha, A, 1, 1, this->matrix_desc(),
        B_ptr, 1, 1, B.matrix_desc(),
        beta,  C_ptr, 1, 1, C.matrix_desc(),
        comp, &devSize, &hostSize)));
}

template <typename T, typename GPU_T>
void BLOCKED_CUDA_Matrix<T, GPU_T>::init(int64_t rsrc, int64_t csrc) {
    if (grid_rows_ == 0) compute_grid_dims();

    my_grid_row_ = my_rank_ / grid_cols_;
    my_grid_col_ = my_rank_ % grid_cols_;

    if (nb_ == 0) compute_optimal_blocks();

    local_rows_ = cublasMpNumroc(n_, nb_, my_grid_row_, rsrc, grid_rows_);
    local_cols_ = cublasMpNumroc(m_, mb_, my_grid_col_, csrc, grid_cols_);

    size_t elem_size = (std::is_same<T, COMPLEX>::value ? sizeof(cuDoubleComplex) : sizeof(double));
    CUDA_CHECK(cudaMalloc(&local_dev_ptr_, local_rows_ * local_cols_ * elem_size));

    cudaDataType_t cuda_type = (std::is_same<T, COMPLEX>::value ? CUDA_C_64F : CUDA_R_64F);

    // Создание дескриптора (аргументы: n, m, nb, mb, rsrc, csrc, lld, type, grid, desc)
    CUBLASMP_CHECK(cublasMpMatrixDescriptorCreate(
        n_, m_, nb_, mb_, rsrc, csrc,
        local_rows_, cuda_type, grid_, &mat_desc_));
}

template <typename T, typename GPU_T>
void BLOCKED_CUDA_Matrix<T, GPU_T>::copy_to_device(const std::vector<T>& host_data) {
    size_t elem_size = (std::is_same<T, COMPLEX>::value ? sizeof(cuDoubleComplex) : sizeof(double));
    CUDA_CHECK(cudaMemcpy(local_dev_ptr_, host_data.data(),
                          local_rows_ * local_cols_ * elem_size,
                          cudaMemcpyHostToDevice));
}

template <typename T, typename GPU_T>
void BLOCKED_CUDA_Matrix<T, GPU_T>::release_resources() {
    if (mat_desc_) {
        cublasMpMatrixDescriptorDestroy(mat_desc_);
        mat_desc_ = nullptr;
    }
    if (local_dev_ptr_) {
        cudaFree(local_dev_ptr_);
        local_dev_ptr_ = nullptr;
    }
}

// Конструкторы
template <typename T, typename GPU_T>
BLOCKED_CUDA_Matrix<T, GPU_T>::BLOCKED_CUDA_Matrix(
    MPI_Comm comm, ncclComm_t nccl_comm,
    cublasMpHandle_t handle, cublasMpGrid_t grid,
    int64_t n, int64_t m, int64_t nb, int64_t mb,
    int64_t rsrc, int64_t csrc)
    : comm_(comm), nccl_comm_(nccl_comm), handle_(handle), grid_(grid),
      n_(n), m_(m), nb_(nb), mb_(mb)
{
    MPI_Comm_rank(comm_, &my_rank_);
    init(rsrc, csrc);
}

template <typename T, typename GPU_T>
BLOCKED_CUDA_Matrix<T, GPU_T>::BLOCKED_CUDA_Matrix(
    MPI_Comm comm, ncclComm_t nccl_comm,
    cublasMpHandle_t handle, cublasMpGrid_t grid,
    int64_t n, int64_t m, int64_t nb, int64_t mb,
    std::function<T(int64_t, int64_t)> func,
    int64_t rsrc, int64_t csrc)
    : comm_(comm), nccl_comm_(nccl_comm), handle_(handle), grid_(grid),
      n_(n), m_(m), nb_(nb), mb_(mb)
{
    MPI_Comm_rank(comm_, &my_rank_);
    init(rsrc, csrc);

    std::vector<T> host_local(local_rows_ * local_cols_);
    for (int64_t j = 0; j < local_cols_; ++j) {
        int64_t global_col = get_global_col(j);
        for (int64_t i = 0; i < local_rows_; ++i) {
            int64_t global_row = get_global_row(i);
            host_local[i + j * local_rows_] = func(global_row, global_col);
        }
    }
    copy_to_device(host_local);
}

template <typename T, typename GPU_T>
BLOCKED_CUDA_Matrix<T, GPU_T>::~BLOCKED_CUDA_Matrix() {
    release_resources();
}

// Перемещение
template <typename T, typename GPU_T>
BLOCKED_CUDA_Matrix<T, GPU_T>::BLOCKED_CUDA_Matrix(BLOCKED_CUDA_Matrix&& other) noexcept {
    *this = std::move(other);
}
template <typename T, typename GPU_T>
BLOCKED_CUDA_Matrix<T, GPU_T>& BLOCKED_CUDA_Matrix<T, GPU_T>::operator=(BLOCKED_CUDA_Matrix&& other) noexcept {
    if (this != &other) {
        release_resources();
        comm_ = other.comm_;
        nccl_comm_ = other.nccl_comm_;
        handle_ = other.handle_;
        grid_ = other.grid_;
        mat_desc_ = other.mat_desc_;
        local_dev_ptr_ = other.local_dev_ptr_;
        n_ = other.n_; m_ = other.m_; nb_ = other.nb_; mb_ = other.mb_;
        local_rows_ = other.local_rows_; local_cols_ = other.local_cols_;
        my_rank_ = other.my_rank_;
        my_grid_row_ = other.my_grid_row_; my_grid_col_ = other.my_grid_col_;
        grid_rows_ = other.grid_rows_; grid_cols_ = other.grid_cols_;

        other.mat_desc_ = nullptr;
        other.local_dev_ptr_ = nullptr;
    }
    return *this;
}

// ---------------------------- индексация ----------------------------
template <typename T, typename GPU_T>
int64_t BLOCKED_CUDA_Matrix<T, GPU_T>::get_global_row(int64_t local_row) const {
    return my_grid_row_ * nb_ + local_row;
}
template <typename T, typename GPU_T>
int64_t BLOCKED_CUDA_Matrix<T, GPU_T>::get_global_col(int64_t local_col) const {
    return my_grid_col_ * mb_ + local_col;
}
template <typename T, typename GPU_T>
int64_t BLOCKED_CUDA_Matrix<T, GPU_T>::get_local_row(int64_t global_row) const {
    if ((global_row / nb_) == my_grid_row_ && global_row < n_)
        return global_row % nb_;
    return -1;
}
template <typename T, typename GPU_T>
int64_t BLOCKED_CUDA_Matrix<T, GPU_T>::get_local_col(int64_t global_col) const {
    if ((global_col / mb_) == my_grid_col_ && global_col < m_)
        return global_col % mb_;
    return -1;
}
template <typename T, typename GPU_T>
bool BLOCKED_CUDA_Matrix<T, GPU_T>::is_my_elem_row(int64_t global_row) const {
    return (global_row / nb_) == my_grid_row_ && global_row < n_;
}
template <typename T, typename GPU_T>
bool BLOCKED_CUDA_Matrix<T, GPU_T>::is_my_elem_col(int64_t global_col) const {
    return (global_col / mb_) == my_grid_col_ && global_col < m_;
}

// ---------------------------- умножение матриц (распределённое) ----------------------------

template <typename T, typename GPU_T>
BLOCKED_CUDA_Matrix<T, GPU_T> BLOCKED_CUDA_Matrix<T, GPU_T>::operator*(const BLOCKED_CUDA_Matrix& other) const {
    assert(m_ == other.n_);
    BLOCKED_CUDA_Matrix res(comm_, nccl_comm_, handle_, grid_,
                             n_, other.m_, nb_, other.mb_);

    // Подготовим alpha и beta нужного типа (GPU_T)
    GPU_T alpha, beta;
    if constexpr (std::is_same<T, double>::value) {
        alpha = 1.0;
        beta  = 0.0;
    } else {
        // Для комплексных используем cuDoubleComplex
        alpha = make_cuDoubleComplex(1.0, 0.0);
        beta  = make_cuDoubleComplex(0.0, 0.0);
    }

    cublasComputeType_t comp = getMpComputeType<T, GPU_T>();
    size_t devSize = 0, hostSize = 0;

    CUBLASMP_CHECK((cublasMpGemm_bufferSize(
        handle_, CUBLAS_OP_N, CUBLAS_OP_N, n_, other.m_, m_,
        &alpha, local_dev_ptr_, 1, 1, mat_desc_,
        other.local_dev_ptr_, 1, 1, other.mat_desc_,
        &beta, res.local_dev_ptr_, 1, 1, res.mat_desc_,
        comp, &devSize, &hostSize)));

    void* d_work = nullptr;
    void* h_work = nullptr;
    if (devSize) CUDA_CHECK(cudaMalloc(&d_work, devSize));
    if (hostSize) h_work = malloc(hostSize);

    CUBLASMP_CHECK((cublasMpGemm(
        handle_, CUBLAS_OP_N, CUBLAS_OP_N, n_, other.m_, m_,
        &alpha, local_dev_ptr_, 1, 1, mat_desc_,
        other.local_dev_ptr_, 1, 1, other.mat_desc_,
        &beta, res.local_dev_ptr_, 1, 1, res.mat_desc_,
        comp, d_work, devSize, h_work, hostSize)));

    if (d_work) cudaFree(d_work);
    if (h_work) free(h_work);
    return res;
}

// ---------------------------- локальные операции через cuBLAS ----------------------------
template <typename T, typename GPU_T>
void BLOCKED_CUDA_Matrix<T, GPU_T>::local_copy(const BLOCKED_CUDA_Matrix& src) {
    assert(local_rows_ == src.local_rows_ && local_cols_ == src.local_cols_);
    size_t elem_size = (std::is_same<T, COMPLEX>::value ? sizeof(cuDoubleComplex) : sizeof(double));
    CUDA_CHECK(cudaMemcpy(local_dev_ptr_, src.local_dev_ptr_,
                          local_rows_ * local_cols_ * elem_size,
                          cudaMemcpyDeviceToDevice));
}

template <typename T, typename GPU_T>
void BLOCKED_CUDA_Matrix<T, GPU_T>::local_axpy(T alpha, const BLOCKED_CUDA_Matrix& X) {
    assert(local_rows_ == X.local_rows_ && local_cols_ == X.local_cols_);
    int64_t N = local_rows_ * local_cols_;
    if constexpr (std::is_same<T, double>::value) {
        CUBLAS_CHECK(cublasDaxpy((cublasHandle_t)handle_, N, &alpha, X.local_dev_ptr_, 1, local_dev_ptr_, 1));
    } else if constexpr (std::is_same<T, COMPLEX>::value) {
        cuDoubleComplex alpha_cu = make_cuDoubleComplex(alpha.real(), alpha.imag());
        CUBLAS_CHECK(cublasZaxpy((cublasHandle_t)handle_, N, &alpha_cu, reinterpret_cast<const cuDoubleComplex*>(X.local_dev_ptr_), 1, local_dev_ptr_, 1));
    }
}

template <typename T, typename GPU_T>
void BLOCKED_CUDA_Matrix<T, GPU_T>::local_scal(T alpha) {
    int64_t N = local_rows_ * local_cols_;
    if constexpr (std::is_same<T, double>::value) {
        CUBLAS_CHECK(cublasDscal((cublasHandle_t)handle_, N, &alpha, local_dev_ptr_, 1));
    } else if constexpr (std::is_same<T, COMPLEX>::value) {
        cuDoubleComplex alpha_cu = make_cuDoubleComplex(alpha.real(), alpha.imag());
        CUBLAS_CHECK(cublasZscal((cublasHandle_t)handle_, N, &alpha_cu, local_dev_ptr_, 1));
    }
}

// ---------------------------- операторы +, -, *= скаляр ----------------------------
template <typename T, typename GPU_T>
BLOCKED_CUDA_Matrix<T, GPU_T> BLOCKED_CUDA_Matrix<T, GPU_T>::operator+(const BLOCKED_CUDA_Matrix& other) const {
    assert(n_ == other.n_ && m_ == other.m_);
    BLOCKED_CUDA_Matrix res(comm_, nccl_comm_, handle_, grid_, n_, m_, nb_, mb_);
    res.local_copy(*this);
    res.local_axpy(1.0, other);
    return res;
}

template <typename T, typename GPU_T>
BLOCKED_CUDA_Matrix<T, GPU_T> BLOCKED_CUDA_Matrix<T, GPU_T>::operator-(const BLOCKED_CUDA_Matrix& other) const {
    assert(n_ == other.n_ && m_ == other.m_);
    BLOCKED_CUDA_Matrix res(comm_, nccl_comm_, handle_, grid_, n_, m_, nb_, mb_);
    res.local_copy(*this);
    T minus_one = -1.0;
    if constexpr (std::is_same<T, COMPLEX>::value) minus_one = COMPLEX(-1.0, 0.0);
    res.local_axpy(minus_one, other);
    return res;
}

template <typename T, typename GPU_T>
BLOCKED_CUDA_Matrix<T, GPU_T> BLOCKED_CUDA_Matrix<T, GPU_T>::operator*(T scalar) const {
    BLOCKED_CUDA_Matrix res(comm_, nccl_comm_, handle_, grid_, n_, m_, nb_, mb_);
    res.local_copy(*this);
    res.local_scal(scalar);
    return res;
}

template <typename T, typename GPU_T>
void BLOCKED_CUDA_Matrix<T, GPU_T>::operator+=(const BLOCKED_CUDA_Matrix& other) {
    local_axpy(1.0, other);
}

template <typename T, typename GPU_T>
void BLOCKED_CUDA_Matrix<T, GPU_T>::operator-=(const BLOCKED_CUDA_Matrix& other) {
    T minus_one = -1.0;
    if constexpr (std::is_same<T, COMPLEX>::value) minus_one = COMPLEX(-1.0, 0.0);
    local_axpy(minus_one, other);
}

template <typename T, typename GPU_T>
void BLOCKED_CUDA_Matrix<T, GPU_T>::operator*=(T scalar) {
    local_scal(scalar);
}

// ---------------------------- сборка на CPU ----------------------------
template <typename T, typename GPU_T>
Matrix<T> BLOCKED_CUDA_Matrix<T, GPU_T>::gather_to_cpu(int root) const {
    int64_t local_size = local_rows_ * local_cols_;
    int rank;
    MPI_Comm_rank(comm_, &rank);

    struct Coord { int row; int col; };
    Coord my_coord = {my_grid_row_, my_grid_col_};
    std::vector<Coord> all_coords;
    if (rank == root) all_coords.resize(grid_rows_ * grid_cols_);
    MPI_Gather(&my_coord, 2, MPI_INT, all_coords.data(), 2, MPI_INT, root, comm_);

    T* gathered_dev = nullptr;
    if (rank == root) {
        size_t elem_size = (std::is_same<T, COMPLEX>::value ? sizeof(cuDoubleComplex) : sizeof(double));
        int64_t total_size = local_size * grid_rows_ * grid_cols_;
        CUDA_CHECK(cudaMalloc(&gathered_dev, total_size * elem_size));
    }

    ncclDataType_t nccl_type = (std::is_same<T, COMPLEX>::value ? ncclFloat64 : ncclDouble);
    NCCL_CHECK(ncclAllGather(local_dev_ptr_, gathered_dev, local_size,
                             nccl_type, nccl_comm_, 0));

    Matrix<T> full_matrix(FORTRAN_STYLE, n_, m_);
    if (rank == root) {
        int64_t total_size = local_size * grid_rows_ * grid_cols_;
        std::vector<T> host_gathered(total_size);
        size_t elem_size = (std::is_same<T, COMPLEX>::value ? sizeof(cuDoubleComplex) : sizeof(double));
        CUDA_CHECK(cudaMemcpy(host_gathered.data(), gathered_dev,
                              total_size * elem_size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(gathered_dev));

        size_t offset = 0;
        for (int proc_rank = 0; proc_rank < grid_rows_ * grid_cols_; ++proc_rank) {
            int proc_row = all_coords[proc_rank].row;
            int proc_col = all_coords[proc_rank].col;
            int64_t block_rows = std::min(nb_, n_ - proc_row * nb_);
            int64_t block_cols = std::min(mb_, m_ - proc_col * mb_);
            if (block_rows <= 0 || block_cols <= 0) continue;

            for (int64_t col = 0; col < block_cols; ++col) {
                for (int64_t row = 0; row < block_rows; ++row) {
                    int64_t global_row = proc_row * nb_ + row;
                    int64_t global_col = proc_col * mb_ + col;
                    full_matrix(global_row, global_col) = host_gathered[offset + row + col * block_rows];
                }
            }
            offset += block_rows * block_cols;
        }
    }

    MPI_Barrier(comm_);
    return full_matrix;
}

// ---------------------------- печать ----------------------------
template <typename T, typename GPU_T>
void BLOCKED_CUDA_Matrix<T, GPU_T>::show_local(size_t width) const {
    size_t elem_size = (std::is_same<T, COMPLEX>::value ? sizeof(cuDoubleComplex) : sizeof(double));
    std::vector<T> host_local(local_rows_ * local_cols_);
    CUDA_CHECK(cudaMemcpy(host_local.data(), local_dev_ptr_,
                          local_rows_ * local_cols_ * elem_size,
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < grid_rows_ * grid_cols_; ++i) {
        MPI_Barrier(comm_);
        if (my_rank_ == i) {
            std::cout << "Process rank " << my_rank_
                      << " (grid: " << my_grid_row_ << ", " << my_grid_col_ << ")"
                      << " local matrix [" << local_rows_ << " x " << local_cols_ << "]:\n";
            for (int64_t r = 0; r < local_rows_; ++r) {
                for (int64_t c = 0; c < local_cols_; ++c) {
                    std::cout << std::setw(width) << host_local[r + c * local_rows_] << " ";
                }
                std::cout << "\n";
            }
            std::cout << std::endl;
        }
        MPI_Barrier(comm_);
    }
}

template <typename T, typename GPU_T>
void BLOCKED_CUDA_Matrix<T, GPU_T>::show(size_t width) const {
    Matrix<T> full = gather_to_cpu(0);
    if (my_rank_ == 0) {
        full.show(width);
    }
}


template <typename T, typename GPU_T>
void optimized_multiply(const BLOCKED_CUDA_Matrix<T, GPU_T>& A,
                        const BLOCKED_CUDA_Matrix<T, GPU_T>& B,
                        BLOCKED_CUDA_Matrix<T, GPU_T>& C,
                        T alpha = T(1), T beta = T(0),
                        cublasOperation_t transA = CUBLAS_OP_N,
                        cublasOperation_t transB = CUBLAS_OP_N,
                        void* d_work = nullptr, size_t devSize = 0, void* h_work = nullptr, size_t hostSize = 0)
{
    static_assert(std::is_same<T, double>::value || std::is_same<T, COMPLEX>::value,
                  "optimized_multiply only for double or COMPLEX");
    GPU_T a, b;
    if constexpr (std::is_same<T, double>::value) {
        a = alpha;
        b = beta;
    } else {
        a = make_cuDoubleComplex(alpha.real(), alpha.imag());
        b = make_cuDoubleComplex(beta.real(), beta.imag());
    }

    bool is_our_work = false;
    cublasComputeType_t comp = getMpComputeType<T, GPU_T>();
    if (!d_work) {
        is_our_work = true;
        CUBLASMP_CHECK((cublasMpGemm_bufferSize(
            A.handle(), transA, transB, A.n(), B.m(), A.m(),
            &a, A.local_data(), 1, 1, A.matrix_desc(),
            B.local_data(), 1, 1, B.matrix_desc(),
            &b, C.local_data(), 1, 1, C.matrix_desc(),
            comp, &devSize, &hostSize)));

        if (devSize) CUDA_CHECK(cudaMalloc(&d_work, devSize));
        if (hostSize) h_work = malloc(hostSize);
    }

    CUBLASMP_CHECK((cublasMpGemm(
        A.handle(), transA, transB, A.n(), B.m(), A.m(),
        &a, A.local_data(), 1, 1, A.matrix_desc(),
        B.local_data(), 1, 1, B.matrix_desc(),
        &b, C.local_data(), 1, 1, C.matrix_desc(),
        comp, d_work, devSize, h_work, hostSize)));

    if (is_our_work) {
        if (d_work) cudaFree(d_work);
        if (h_work) free(h_work);
    }
}

template <typename T, typename GPU_T>
void optimized_add(const BLOCKED_CUDA_Matrix<T, GPU_T>& A,
                        BLOCKED_CUDA_Matrix<T, GPU_T>& C,
                        T alpha = T(1), T beta = T(0),
                        cublasOperation_t transA = CUBLAS_OP_N,
                        void* d_work = nullptr, size_t devSize = 0, void* h_work = nullptr, size_t hostSize = 0)
{
    static_assert(std::is_same<T, double>::value || std::is_same<T, COMPLEX>::value,
                  "optimized_multiply only for double or COMPLEX");
    GPU_T a, b;
    if constexpr (std::is_same<T, double>::value) {
        a = alpha;
        b = beta;
    } else {
        a = make_cuDoubleComplex(alpha.real(), alpha.imag());
        b = make_cuDoubleComplex(beta.real(), beta.imag());
    }

    bool is_our_work = false;
    if (!d_work) {
        is_our_work = true;
        CUBLASMP_CHECK((cublasMpGeadd_bufferSize(
            A.handle(), transA, A.n(), A.m(),
            &a, A.local_data(), 1, 1, A.matrix_desc(),
            &b, C.local_data(), 1, 1, C.matrix_desc(),
            &devSize, &hostSize)));

        if (devSize) CUDA_CHECK(cudaMalloc(&d_work, devSize));
        if (hostSize) h_work = malloc(hostSize);
    }

    CUBLASMP_CHECK((cublasMpGeadd(
        A.handle(), transA, A.n(), A.m(),
        &a, A.local_data(), 1, 1, A.matrix_desc(),
        &b, C.local_data(), 1, 1, C.matrix_desc(),
        d_work, devSize, h_work, hostSize)));

    if (is_our_work) {
        if (d_work) cudaFree(d_work);
        if (h_work) free(h_work);
    }
}

template <typename T, typename GPU_T>
std::vector<BLOCKED_CUDA_Matrix<T, GPU_T>> CUDA_MPI_Runge_Kutt_2(
    const std::vector<double>& x,
    BLOCKED_CUDA_Matrix<T, GPU_T> y0,   // по значению, чтобы использовать move
    std::function<void(double, const BLOCKED_CUDA_Matrix<T, GPU_T>&, BLOCKED_CUDA_Matrix<T, GPU_T>&)> f)
{
    size_t len = x.size();
    if (len == 0) return {};

    size_t dim = y0.n();   // предполагаем квадратную матрицу
    auto comm = y0.comm();
    auto nccl_comm = y0.nccl_comm();
    auto handle = y0.handle();
    auto grid = y0.grid();
    int64_t nb = y0.nb();
    int64_t mb = y0.mb();

    std::vector<BLOCKED_CUDA_Matrix<T, GPU_T>> y;
    y.reserve(len);
    y.emplace_back(std::move(y0));   // начальное условие

    // Вспомогательные матрицы для стадий Рунге-Кутты
    BLOCKED_CUDA_Matrix<T, GPU_T> k1(comm, nccl_comm, handle, grid, dim, dim, nb, mb);
    BLOCKED_CUDA_Matrix<T, GPU_T> k2(comm, nccl_comm, handle, grid, dim, dim, nb, mb);
    BLOCKED_CUDA_Matrix<T, GPU_T> y_temp(comm, nccl_comm, handle, grid, dim, dim, nb, mb);

    // Функция копирования данных с устройства на устройство
    auto copy_matrix = [](const BLOCKED_CUDA_Matrix<T, GPU_T>& src, BLOCKED_CUDA_Matrix<T, GPU_T>& dst) {
        assert(src.local_rows() == dst.local_rows() && src.local_cols() == dst.local_cols());
        cudaMemcpy(dst.local_data(), src.local_data(),
                   src.local_rows() * src.local_cols() * sizeof(GPU_T),
                   cudaMemcpyDeviceToDevice);
    };

    for (size_t i = 0; i < len - 1; ++i) {
        double h = x[i + 1] - x[i];

        // k1 = f(x_i, y_i)
        f(x[i], y[i], k1);

        // y_temp = y_i + h * k1
        copy_matrix(y[i], y_temp);
        optimized_add(k1, y_temp,
                      static_cast<T>(h),
                      static_cast<T>(1.0));

        // k2 = f(x_i + h, y_temp)
        f(x[i] + h, y_temp, k2);

        // y_{i+1} = y_i + (h/2)*(k1 + k2)
        BLOCKED_CUDA_Matrix<T, GPU_T> y_next(comm, nccl_comm, handle, grid, dim, dim, nb, mb);
        copy_matrix(y[i], y_next);
        optimized_add(k1, y_next,
                      static_cast<T>(h / 2.0),
                      static_cast<T>(1.0));
        optimized_add(k2, y_next,
                      static_cast<T>(h / 2.0),
                      static_cast<T>(1.0));

        y.emplace_back(std::move(y_next));
    }
    return y;
}

// template <typename T, typename GPU_T>
// void optimized_add(const BLOCKED_CUDA_Matrix<T, GPU_T>& A,
//                    BLOCKED_CUDA_Matrix<T, GPU_T>& C,
//                    T alpha, T beta,
//                    cublasOperation_t transA = CUBLAS_OP_N)
// {
//     static_assert(std::is_same<T, double>::value || std::is_same<T, COMPLEX>::value,
//                   "optimized_add only for double or COMPLEX");
//     if constexpr (std::is_same<T, double>::value) {
//         CUBLASMP_CHECK((cublasMpGeadd(A.handle(),
//                                       transA,
//                                       A.n(), A.m(),
//                                       &alpha, A.local_data(), 0, 0, A.matrix_desc(),
//                                       &beta,  C.local_data(), 0, 0, C.matrix_desc(),
//                                       nullptr, 0, nullptr, 0)));
//     } else {
//         cuDoubleComplex a = make_cuDoubleComplex(alpha.real(), alpha.imag());
//         cuDoubleComplex b = make_cuDoubleComplex(beta.real(), beta.imag());
//         CUBLASMP_CHECK((cublasMpGeadd(A.handle(),
//                                       transA,
//                                       A.n(), A.m(),
//                                       &a, A.local_data(), 0, 0, A.matrix_desc(),
//                                       &b,  C.local_data(), 0, 0, C.matrix_desc(),
//                                       nullptr, 0, nullptr, 0)));
//     }
// }

} // namespace QComputations

#endif // ENABLE_MPI
#endif // __CUDACC__
// example_blocked_cuda.cpp
#include <mpi.h>
#include <nccl.h>
#include <cublasmp.h>
#include <cuda_runtime.h>
#include <iostream>
#include "QComputations_BLOCKED_CUDA.hpp"   // ваш класс BLOCKED_CUDA_Matrix

using namespace QComputations;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 1. Инициализация NCCL коммуникатора
    ncclUniqueId ncclId;
    if (rank == 0) ncclGetUniqueId(&ncclId);
    MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclComm_t ncclComm;
    ncclCommInitRank(&ncclComm, size, ncclId, rank);

    // 2. Инициализация cuBLASMp
    cublasMpHandle_t handle;
    cublasMpCreate(&handle, nullptr);   // stream = NULL

    // 3. Создание 2D-решётки процессов (автоматический подбор)
    int nprow = static_cast<int>(std::sqrt(size));
    while (size % nprow != 0) nprow--;
    int npcol = size / nprow;
    cublasMpGrid_t grid;
    cublasMpGridCreate(nprow, npcol, CUBLASMP_GRID_LAYOUT_ROW_MAJOR, ncclComm, &grid);

    // 4. Параметры матрицы (небольшие для наглядности)
    int64_t N = 5, M = 5;   // глобальные размеры
    int64_t NB = 2, MB = 2; // размеры блока

    // 5. Функтор для заполнения (пример)
    auto func = [](int64_t i, int64_t j) -> COMPLEX {
        return COMPLEX(i + 1.0, j * 0.5);   // Re = номер строки+1, Im = 0.5*номер столбца
    };

    // 6. Создание распределённой матрицы на GPU
    BLOCKED_CUDA_Matrix<COMPLEX, cuDoubleComplex> A(
        MPI_COMM_WORLD, ncclComm, handle, grid,
        N, M, NB, MB, func
    );

    A.show_local();

    BLOCKED_CUDA_Matrix<COMPLEX> B(MPI_COMM_WORLD, ncclComm, handle, grid,
        N, M, NB, MB, func);

    auto res = A * B;

    res.show_local();

    // 8. Очистка ресурсов
    cublasMpGridDestroy(grid);
    cublasMpDestroy(handle);
    ncclCommDestroy(ncclComm);

    MPI_Finalize();
    return 0;
}
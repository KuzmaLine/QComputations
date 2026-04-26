#include "QComputations_BLOCKED_CUDA.hpp"
#include <iostream>
#include <regex>
#include <complex>
#include <chrono>

constexpr bool is_python_api = false;
constexpr int max_photons = 2;

using COMPLEX = std::complex<double>;

int main(int argc, char** argv) {
    using namespace QComputations;

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

    QConfig::instance().set_width(20); // Ширина ячейки элемента матрицы для stdout
    double h = QConfig::instance().h(); // Получить постоянную планка
    double w = QConfig::instance().w(); // Получить частоту
    QConfig::instance().set_g(0.005); // сила взаимодействия с полем атома
    QConfig::instance().set_max_photons(max_photons);

    std::vector<size_t> grid_config = {21, 30};

    TCH_State state(grid_config);
    state.set_n(QConfig::instance().max_photons(), 0);
    state.set_waveguide(0, 1, 0.01);
    state.set_leak_for_cavity(1, 0.2);


    
    BLOCKED_CUDA_H_TCH H(MPI_COMM_WORLD, ncclComm, handle, grid, state);

    // show_basis(H.get_basis());

    // H.print_distributed();

    // H.show();
    // std::cout << H.size() << std::endl;

    auto time_vec = linspace(0, 50, 50);

    std::cout << "H size = " << H.size() << " TIME SIZE = " << time_vec.size() << std::endl;

    cudaEvent_t start, stop;

    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    quantum_master_equation(State<Basis_State>(state), H, time_vec);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed_ms;

    cudaEventElapsedTime(&elapsed_ms, start, stop);

    std::cout << elapsed_ms << std::endl;

    // make_probs_files(H, probs, time_vec, H.get_basis(), "results/general_tch_QME");

    cublasMpGridDestroy(grid);
    cublasMpDestroy(handle);
    ncclCommDestroy(ncclComm);

    MPI_Finalize();

    return 0;
}
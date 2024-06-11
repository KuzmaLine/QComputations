#include <iostream>
#include <chrono>
#include <complex>
#include "/home/kuzmaline/Quantum/diploma/src/QComputations_CPU_CLUSTER_NO_PLOTS.hpp"

int main(int argc, char** argv) {
    using namespace QComputations;
    //QConfig::instance().set_h(0.1); // - изменение постояннной планка на 0.1
    //std::cout << QConfig::instance().h() << std::endl;

    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    State_Graph

    MPI_Finalize();
}
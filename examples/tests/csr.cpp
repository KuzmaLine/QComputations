#include "QComputations_SINGLE.hpp"

using namespace QComputations;

COMPLEX func(ILP_TYPE i, ILP_TYPE j) {
    return 2 * i + j;
}

int main() {
    ILP_TYPE n = 4;
    ILP_TYPE m = 4;

    Matrix<COMPLEX> A(C_STYLE, n, m, func);
    CSR_Matrix<COMPLEX> CSR_A(n, m, func);

    A.show();

    std::cout << std::endl;

    CSR_A.show();

    std::cout << std::endl;

    auto B = A * A;

    B.show();

    std::cout << std::endl;

    Matrix<COMPLEX> C(C_STYLE, B.n(), B.m());

    optimized_multiply(A, A, C, COMPLEX(1, 0), COMPLEX(0, 0), 'N');

    C.show();

    std::cout << std::endl;

    auto CSR_C = sparse_spmm(CSR_A, CSR_A);

    CSR_C.show();

    return 0;
}
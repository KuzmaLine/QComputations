#include "QComputations_SINGLE.hpp"

using namespace QComputations;

COMPLEX func(ILP_TYPE i, ILP_TYPE j) {
    return 2 * i + j;
}

// Only hermitian
COMPLEX func_2(ILP_TYPE i, ILP_TYPE j) {
    return COMPLEX(i, 0) + COMPLEX(j, 0) + (i >= j ? COMPLEX(0, 1) : COMPLEX(0, -1)) - (i == j ? COMPLEX(0, 1) : 0);
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

    optimized_multiply(CSR_A, A, C, COMPLEX(1, 0), COMPLEX(0, 0), 'N');

    std::cout << std::endl;

    C.show();

    Matrix<COMPLEX> left_C(C_STYLE, n, m);

    optimized_multiply(A, CSR_A, left_C, COMPLEX(1, 0), COMPLEX(0, 0), 'N');

    std::cout << std::endl;

    left_C.show();

    // ---------------------- 
    Matrix<COMPLEX> B_syrd(C_STYLE, n, m, func_2);
    Matrix<COMPLEX> new_C(left_C);

    std::cout << std::endl;

    B_syrd.show();

    optimized_multiply(A, B_syrd, C, COMPLEX(1, 0), COMPLEX(0, 0), 'N', 'N');
    optimized_multiply(C, A, new_C, COMPLEX(1, 0), COMPLEX(1, 0), 'N', 'C');

    std::cout << std::endl;

    new_C.show();

    std::cout << std::endl;

    sparse_syrd(CSR_A, B_syrd, left_C, COMPLEX(1, 0), COMPLEX(1, 0));

    left_C.show();

    return 0;
}
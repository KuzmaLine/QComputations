#include "additional_operators.hpp"

extern "C" {
void zdotc(COMPLEX*, int*, const COMPLEX*, int*, const COMPLEX*, int*);
}

namespace QComputations {

COMPLEX operator|(const std::vector<COMPLEX>& a, const std::vector<COMPLEX>& b) {
    int size = a.size();
    int iONE = 1;
    COMPLEX res;
    zdotc(&res, &size, a.data(), &iONE, b.data(), &iONE);

    return res;
}

Matrix<COMPLEX> operator*(const Matrix<COMPLEX>& A, const Matrix<double>& B) {
    assert(A.m() == B.n());
    Matrix<COMPLEX> res(A.matrix_style(), A.n(), B.m(), COMPLEX(0));

    for (size_t i = 0; i < A.n(); i++) {
        for (size_t j = 0; j < B.m(); j++) {
            for (size_t k = 0; k < A.m(); k++) {
                res[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return res;
}

}  // namespace QComputations
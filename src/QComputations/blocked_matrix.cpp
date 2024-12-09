#ifdef ENABLE_MPI
#ifdef ENABLE_CLUSTER
#define MKL_Complex16 std::complex<double>

#include "blocked_matrix.hpp"

#include <mkl_pblas.h>
#include <mkl_scalapack.h>

#include <fstream>

namespace QComputations {

extern "C" {
//void pztranu(ILP_TYPE*, ILP_TYPE*, const COMPLEX*, const COMPLEX*, ILP_TYPE*, ILP_TYPE*,
//             ILP_TYPE*, COMPLEX*, COMPLEX*, ILP_TYPE*, ILP_TYPE*, ILP_TYPE*);
//void pdtran(ILP_TYPE*, ILP_TYPE*, const double*, const double*, ILP_TYPE*, ILP_TYPE*,
//            ILP_TYPE*, double*, double*, ILP_TYPE*, ILP_TYPE*, ILP_TYPE*);
}

namespace {
namespace of {
int ns = 0,  // Номер строки.
    bg = 1,  // Офсет начала строки.
    md = 2,  // Локальный офсет места, куда писать матрицу.
    en = 3;  // Офсет конца строки.
}
}  // namespace

template <>
double BLOCKED_Matrix<double>::get(size_t i, size_t j) const {
    return mpi::pdelget(local_matrix_, i, j, this->desc());
}

template <>
COMPLEX BLOCKED_Matrix<COMPLEX>::get(size_t i, size_t j) const {
    return mpi::pzelget(local_matrix_, i, j, this->desc());
}

template <>
void BLOCKED_Matrix<double>::set(size_t i, size_t j, double num) {
    mpi::pdelset(local_matrix_, i, j, num, this->desc());
}

template <>
void BLOCKED_Matrix<COMPLEX>::set(size_t i, size_t j, COMPLEX num) {
    mpi::pzelset(local_matrix_, i, j, num, this->desc());
}

template <>
BLOCKED_Matrix<double> BLOCKED_Matrix<double>::operator-(const BLOCKED_Matrix<double>& B) const {
    assert(this->matrix_type_ == GE and B.matrix_type_ == GE);
    assert(this->n_ == B.n_ and this->m_ == B.m_);

    BLOCKED_Matrix<double> C(B);

    mpi::parallel_dgeadd(this->get_local_matrix(), C.get_local_matrix(), this->desc(), C.desc(), double(1.0),
                         double(-1.0));

    return C;
}

template <>
BLOCKED_Matrix<COMPLEX> BLOCKED_Matrix<COMPLEX>::operator-(const BLOCKED_Matrix<COMPLEX>& B) const {
    assert(this->matrix_type_ == GE and B.matrix_type_ == GE);
    assert(this->n_ == B.n_ and this->m_ == B.m_);

    BLOCKED_Matrix<COMPLEX> C(B);

    mpi::parallel_zgeadd(this->get_local_matrix(), C.get_local_matrix(), this->desc(), C.desc(), COMPLEX(1, 0),
                         COMPLEX(-1, 0));

    return C;
}

template <>
BLOCKED_Matrix<double> BLOCKED_Matrix<double>::operator+(const BLOCKED_Matrix<double>& B) const {
    assert(this->matrix_type_ == GE and B.matrix_type_ == GE);
    assert(this->n_ == B.n_ and this->m_ == B.m_);

    BLOCKED_Matrix<double> C(B);

    mpi::parallel_dgeadd(this->get_local_matrix(), C.get_local_matrix(), this->desc(), C.desc());

    return C;
}

template <>
BLOCKED_Matrix<COMPLEX> BLOCKED_Matrix<COMPLEX>::operator+(const BLOCKED_Matrix<COMPLEX>& B) const {
    assert(this->matrix_type_ == GE and B.matrix_type_ == GE);
    assert(this->n_ == B.n_ and this->m_ == B.m_);

    BLOCKED_Matrix<COMPLEX> C(B);

    mpi::parallel_zgeadd(this->get_local_matrix(), C.get_local_matrix(), this->desc(), C.desc());

    return C;
}

template <>
BLOCKED_Matrix<double> BLOCKED_Matrix<double>::operator*(const BLOCKED_Matrix<double>& B) const {
    BLOCKED_Matrix<double> C(*this, B);

    if (this->matrix_type_ == GE and B.matrix_type_ == GE) {
        mpi::parallel_dgemm(this->get_local_matrix(), B.get_local_matrix(), C.get_local_matrix(), this->desc(),
                            B.desc(), C.desc());
    } else {
        std::cerr << "Matrix types error!" << std::endl;
    }
    return C;
}

template <>
BLOCKED_Matrix<COMPLEX> BLOCKED_Matrix<COMPLEX>::operator*(const BLOCKED_Matrix<COMPLEX>& B) const {
    BLOCKED_Matrix<COMPLEX> C(*this, B);

    if (this->matrix_type_ == GE and B.matrix_type_ == GE) {
        mpi::parallel_zgemm(this->get_local_matrix(), B.get_local_matrix(), C.get_local_matrix(), this->desc(),
                            B.desc(), C.desc());
    } else if (this->matrix_type_ == HE and B.matrix_type_ == GE) {
        mpi::parallel_zhemm('L', this->get_local_matrix(), B.get_local_matrix(), C.get_local_matrix(), this->desc(),
                            B.desc(), C.desc());
    } else if (this->matrix_type_ == GE and B.matrix_type_ == HE) {
        mpi::parallel_zhemm('R', B.get_local_matrix(), this->get_local_matrix(), C.get_local_matrix(), B.desc(),
                            this->desc(), C.desc());
    } else {
        std::cerr << "Matrix types error!" << std::endl;
    }

    return C;
}

template <>
BLOCKED_Matrix<double> BLOCKED_Matrix<double>::hermit() const {
    if (matrix_type_ == GE) {
        BLOCKED_Matrix<double> A(ctxt_, GE, m_, n_);
        double alpha = 1;
        double betta = 0;
        ILP_TYPE iONE = 1;
        ILP_TYPE m = m_, n = n_;

        pdtran(&m, &n, &alpha, this->data(), &iONE, &iONE, (this->desc()).data(), &betta, A.data(), &iONE, &iONE,
               A.desc().data());

        return A;
    }

    return *this;
}

template <>
BLOCKED_Matrix<COMPLEX> BLOCKED_Matrix<COMPLEX>::hermit() const {
    if (matrix_type_ == GE) {
        BLOCKED_Matrix<COMPLEX> A(ctxt_, GE, m_, n_);
        COMPLEX alpha(1, 0);
        COMPLEX betta(0, 0);
        ILP_TYPE iONE = 1;
        ILP_TYPE m = m_, n = n_;

        pztranc(&m, &n, &alpha, this->data(), &iONE, &iONE, (this->desc()).data(), &betta, A.data(), &iONE, &iONE,
                A.desc().data());

        return A;
    }

    return *this;
}

template <>
void BLOCKED_Matrix<double>::write_to_csv_file(const std::string& filename) const {
    const char* charname = filename.c_str();
    size_t max_number_size = QConfig::instance().csv_max_number_size();
    size_t num_accuracy = QConfig::instance().csv_num_accuracy();

    int rank, size, root_id = 0;
    int file_exists = 1, end_not_empty = 0;
    long long A_n = this->n(), A_m = this->m();

    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_File file;

    // Существует ли файл.
    if (rank == root_id) {
        std::ifstream file_ending(filename, std::ifstream::ate);
        if (file_ending)
            file_ending.close();
        else
            file_exists = 0;
    }

    MPI_Bcast(&file_exists, 1, MPI_INT, root_id, MPI_COMM_WORLD);

    const int delimiter_size = 1;
    const int one_elem_size = max_number_size + 1;
    const char char_delimiter = ',';
    std::string str_delimiter = ",";

    if (file_exists == 1) {
        MPI_File_open(MPI_COMM_WORLD, charname, MPI_MODE_APPEND | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
    } else {
        MPI_File_open(MPI_COMM_WORLD, charname, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
    }

    MPI_Offset start_offset;
    MPI_File_get_position(file, &start_offset);
    MPI_Offset offset = 0;

    for (size_t i = 0; i < this->local_n(); i++) {
        for (size_t j = 0; j < this->local_m(); j++) {
            auto cur_index = this->get_global_row(i) * A_m + this->get_global_col(j);
            offset = start_offset + (one_elem_size + delimiter_size) * cur_index;
            MPI_File_seek(file, offset, MPI_SEEK_SET);
            if ((cur_index + 1) % A_m != 0) {
                auto num_str = to_string_double_with_precision(this->data()[j * this->local_n() + i], num_accuracy,
                                                               max_number_size) +
                               char_delimiter;
                MPI_File_write(file, num_str.c_str(), num_str.length(), MPI_CHAR, &status);
            } else {
                auto num_str = to_string_double_with_precision(this->data()[j * this->local_n() + i], num_accuracy,
                                                               max_number_size);
                num_str += "\n";
                MPI_File_write(file, num_str.c_str(), num_str.length(), MPI_CHAR, &status);
            }
        }
    }
    MPI_File_close(&file);
}

template <>
void BLOCKED_Matrix<COMPLEX>::write_to_csv_file(const std::string& filename) const {
    const char* charname = filename.c_str();
    size_t max_number_size = QConfig::instance().csv_max_number_size();
    size_t num_accuracy = QConfig::instance().csv_num_accuracy();

    int rank, size, root_id = 0;
    int file_exists = 1, end_not_empty = 0;
    long long A_n = this->n(), A_m = this->m();

    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_File file;

    // Существует ли файл.
    if (rank == root_id) {
        std::ifstream file_ending(filename, std::ifstream::ate);
        if (file_ending)
            file_ending.close();
        else
            file_exists = 0;
    }

    MPI_Bcast(&file_exists, 1, MPI_INT, root_id, MPI_COMM_WORLD);

    const int delimiter_size = 1;
    const int one_elem_size = 2 * max_number_size + 3;
    const char char_delimiter = ',';
    std::string str_delimiter = ",";

    if (file_exists == 1) {
        MPI_File_open(MPI_COMM_WORLD, charname, MPI_MODE_APPEND | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
    } else {
        MPI_File_open(MPI_COMM_WORLD, charname, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
    }

    MPI_Offset start_offset;
    MPI_File_get_position(file, &start_offset);
    MPI_Offset offset = 0;

    for (size_t i = 0; i < this->local_n(); i++) {
        for (size_t j = 0; j < this->local_m(); j++) {
            auto cur_index = this->get_global_row(i) * A_m + this->get_global_col(j);
            offset = start_offset + (one_elem_size + delimiter_size) * cur_index;
            MPI_File_seek(file, offset, MPI_SEEK_SET);
            if ((cur_index + 1) % A_m != 0) {
                auto num_str = to_string_complex_with_precision(this->data()[j * this->local_n() + i], num_accuracy,
                                                                max_number_size) +
                               char_delimiter;
                MPI_File_write(file, num_str.c_str(), num_str.length(), MPI_CHAR, &status);
            } else {
                auto num_str = to_string_complex_with_precision(this->data()[j * this->local_n() + i], num_accuracy,
                                                                max_number_size);
                num_str += "\n";
                MPI_File_write(file, num_str.c_str(), num_str.length(), MPI_CHAR, &status);
            }
        }
    }
    MPI_File_close(&file);
}

// --------------------------------------------- FUNCTIONS -------------------------------------------

std::pair<std::vector<double>, BLOCKED_Matrix<COMPLEX>> Hermit_Lanczos(const BLOCKED_Matrix<COMPLEX>& M) {
    char jobz = 'V';
    char range = 'A';
    char uplo = 'U';

    BLOCKED_Matrix<COMPLEX> A(M);

    ILP_TYPE n = A.n();
    ILP_TYPE iONE = 1;
    ILP_TYPE iMINUS = -1;
    ILP_TYPE iZERO = 0;
    ILP_TYPE info;

    BLOCKED_Matrix<COMPLEX> Z(A.ctxt(), GE, A.n(), A.m(), A.NB(), A.MB());
    std::vector<double> w(A.n());
    std::vector<COMPLEX> work(1);
    std::vector<double> rwork(1);
    std::vector<ILP_TYPE> iwork(1);
    pzheevd(&jobz, &uplo, &n, A.data(), &iONE, &iONE, A.desc().data(), w.data(), Z.data(), &iONE, &iONE,
            Z.desc().data(), work.data(), &iMINUS, rwork.data(), &iMINUS, iwork.data(), &iMINUS, &info);
    ILP_TYPE lwork = work[0].real();
    ILP_TYPE lrwork = rwork[0];
    ILP_TYPE liwork = iwork[0];

    work.resize(lwork);
    rwork.resize(lrwork);
    iwork.resize(liwork);
    pzheevd(&jobz, &uplo, &n, A.data(), &iONE, &iONE, A.desc().data(), w.data(), Z.data(), &iONE, &iONE,
            Z.desc().data(), work.data(), &lwork, rwork.data(), &lrwork, iwork.data(), &liwork, &info);

    return std::make_pair(w, Z);
}

template <>
void optimized_add(const BLOCKED_Matrix<double>& A, BLOCKED_Matrix<double>& C, double alpha, double betta,
                   char trans_A) {
    assert(A.matrix_type() == GE and C.matrix_type() == GE);
    assert(A.n() == C.n() and A.m() == C.m());

    mpi::parallel_dgeadd(A.get_local_matrix(), C.get_local_matrix(), A.desc(), C.desc(), alpha, betta, trans_A);
}

template <>
void optimized_add(const BLOCKED_Matrix<COMPLEX>& A, BLOCKED_Matrix<COMPLEX>& C, COMPLEX alpha, COMPLEX betta,
                   char trans_A) {
    assert(A.matrix_type() == GE and C.matrix_type() == GE);
    assert(A.n() == C.n() and A.m() == C.m());

    mpi::parallel_zgeadd(A.get_local_matrix(), C.get_local_matrix(), A.desc(), C.desc(), alpha, betta, trans_A);
}

template <>
void optimized_multiply(const BLOCKED_Matrix<double>& A, const BLOCKED_Matrix<double>& B, BLOCKED_Matrix<double>& C,
                        double alpha, double betta, char trans_A, char trans_B) {
    assert(C.m() == B.m() and C.n() == A.n());

    if (A.matrix_type() == GE and B.matrix_type() == GE) {
        mpi::parallel_dgemm(A.get_local_matrix(), B.get_local_matrix(), C.get_local_matrix(), A.desc(), B.desc(),
                            C.desc(), alpha, betta, trans_A, trans_B);
    } else {
        std::cerr << "Matrix types error DOUBLE!" << std::endl;
    }
}

template <>
void optimized_multiply(const BLOCKED_Matrix<COMPLEX>& A, const BLOCKED_Matrix<COMPLEX>& B, BLOCKED_Matrix<COMPLEX>& C,
                        COMPLEX alpha, COMPLEX betta, char trans_A, char trans_B) {
    assert(C.m() == B.m() and C.n() == A.n());
    if (A.matrix_type() == GE and B.matrix_type() == GE) {
        mpi::parallel_zgemm(A.get_local_matrix(), B.get_local_matrix(), C.get_local_matrix(), A.desc(), B.desc(),
                            C.desc(), alpha, betta, trans_A, trans_B);
    } else if (A.matrix_type() == HE and B.matrix_type() == GE) {
        mpi::parallel_zhemm('L', A.get_local_matrix(), B.get_local_matrix(), C.get_local_matrix(), A.desc(), B.desc(),
                            C.desc(), alpha, betta);
    } else if (A.matrix_type() == GE and B.matrix_type() == HE) {
        mpi::parallel_zhemm('R', B.get_local_matrix(), A.get_local_matrix(), C.get_local_matrix(), B.desc(), A.desc(),
                            C.desc(), alpha, betta);
    } else {
        std::cerr << "Matrix types error COMPLEX! " << A.matrix_type() << " " << B.matrix_type() << std::endl;
    }
}

std::vector<BLOCKED_Matrix<COMPLEX>> MPI_Runge_Kutt_4(
    const std::vector<double>& x, const BLOCKED_Matrix<COMPLEX>& y0,
    std::function<void(double, const BLOCKED_Matrix<COMPLEX>&, BLOCKED_Matrix<COMPLEX>&)> f) {
    size_t len = x.size();
    size_t dim = y0.n();
    std::vector<BLOCKED_Matrix<COMPLEX>> y(len, BLOCKED_Matrix<COMPLEX>(y0.ctxt(), GE, dim, dim));
    y[0] = y0;

    BLOCKED_Matrix<COMPLEX> k1(y0.ctxt(), GE, dim, dim);
    BLOCKED_Matrix<COMPLEX> k2(y0.ctxt(), GE, dim, dim);
    BLOCKED_Matrix<COMPLEX> k3(y0.ctxt(), GE, dim, dim);

    for (size_t i = 0; i < len - 1; i++) {
        double h = x[i + 1] - x[i];

        f(x[i], y[i], k1);
        optimized_add(y[i], k1, COMPLEX(1, 0), COMPLEX(h / 2.0, 0));
        f(x[i] + h / 2.0, k1, k2);
        optimized_add(y[i], k2, COMPLEX(1, 0), COMPLEX(h / 2.0, 0));
        f(x[i] + h / 2.0, k2, k3);
        optimized_add(y[i], k3, COMPLEX(1, 0), COMPLEX(h, 0));
        f(x[i] + h, k3, y[i + 1]);
        optimized_add(y[i], k1, COMPLEX(double(-2) / h, 0), COMPLEX(double(2) / h, 0));
        optimized_add(y[i], k2, COMPLEX(double(-2) / h, 0), COMPLEX(double(2) / h, 0));
        optimized_add(y[i], k3, COMPLEX(double(-1) / h, 0), COMPLEX(double(1) / h, 0));

        optimized_add(k3, y[i + 1], COMPLEX((h / 3.0), 0), COMPLEX((h / 6.0), 0));
        optimized_add(k2, y[i + 1], COMPLEX((h / 3.0), 0), COMPLEX(1, 0));
        optimized_add(k1, y[i + 1], COMPLEX((h / 6.0), 0), COMPLEX(1, 0));
        optimized_add(y[i], y[i + 1], COMPLEX(1, 0), COMPLEX(1, 0));
    }

    return y;
}

std::vector<BLOCKED_Matrix<COMPLEX>> MPI_Runge_Kutt_2(
    const std::vector<double>& x, const BLOCKED_Matrix<COMPLEX>& y0,
    std::function<void(double, const BLOCKED_Matrix<COMPLEX>&, BLOCKED_Matrix<COMPLEX>&)> f) {
    size_t len = x.size();
    size_t dim = y0.n();
    std::vector<BLOCKED_Matrix<COMPLEX>> y(len, BLOCKED_Matrix<COMPLEX>(y0.ctxt(), GE, dim, dim));
    y[0] = y0;

    BLOCKED_Matrix<COMPLEX> k1(y0.ctxt(), GE, dim, dim);

    for (size_t i = 0; i < len - 1; i++) {
        double h = x[i + 1] - x[i];

        f(x[i], y[i], k1);
        optimized_add(y[i], k1, COMPLEX(1, 0), COMPLEX(h, 0));
        f(x[i] + h, k1, y[i + 1]);
        optimized_add(y[i], k1, COMPLEX(double(-1) / h, 0), COMPLEX(double(1) / h, 0));
        optimized_add(k1, y[i + 1], COMPLEX(h / 2.0, 0), COMPLEX(h / 2.0, 0));
        optimized_add(y[i], y[i + 1], COMPLEX(1, 0), COMPLEX(1, 0));
    }

    return y;
}

}  // namespace QComputations

#endif
#endif
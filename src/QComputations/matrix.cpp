#include "matrix.hpp"
#include <iostream>
#include <mkl_cblas.h>
#include <mkl_blas.h>
#include "mpi_functions.hpp"
#include <chrono>
#include <fstream>
#include "functions.hpp"

namespace QComputations {

namespace {
    using COMPLEX = std::complex<double>;
}

template<>
Matrix<double> Matrix<double>::operator* (const Matrix<double>& A) const {
    assert(m_ == A.n_);
    Matrix<double> res(this->get_matrix_style(), n_, A.m_);

    double alpha = 1.0;
    double betta = 0.0;

    auto type = CblasRowMajor;
    if (!(this->is_c_style())) type = CblasColMajor;
    cblas_dgemm(type, CblasNoTrans, CblasNoTrans,
                n_, A.m_, m_, alpha, mass_.data(),
                this->LD(), A.mass_.data(), A.LD(), betta, res.mass_.data(), res.LD());
    return res;
}

template<>
Matrix<int> Matrix<int>::operator* (const Matrix<int>& A) const {
    Matrix<double> tmp_A(A);
    Matrix<double> tmp_this(*this);

    Matrix<double> tmp_res = tmp_this * tmp_A;

    return Matrix<int>(tmp_res);
}

template<>
Matrix<COMPLEX> Matrix<COMPLEX>::operator* (const Matrix<COMPLEX>& A) const {
    assert(m_ == A.n_);
    Matrix<COMPLEX> res(this->get_matrix_style(), n_, A.m_);

    COMPLEX alpha(1, 0);
    COMPLEX betta(0, 0);

    auto type = CblasRowMajor;
    if (!(this->is_c_style())) type = CblasColMajor;
    cblas_zgemm(type, CblasNoTrans, CblasNoTrans,
                n_, A.m_, m_, &alpha, mass_.data(),
                this->LD(), A.mass_.data(), A.LD(), &betta,
                res.mass_.data(), res.LD());
    return res;
}

template<>
void Matrix<double>::write_to_csv_file(const std::string& filename) const {
#ifndef ENABLE_CLUSTER
    std::ofstream file(filename);

    size_t max_number_size = QConfig::instance().csv_max_number_size();
    size_t num_accuracy = QConfig::instance().csv_num_accuracy();

    const int delimiter_size = 1;
    const int one_elem_size = max_number_size + 1;
    const char char_delimiter = ',';

    for (size_t i = 0; i < this->n(); i++) {
        for (size_t j = 0; j < this->m(); j++) {
            auto cur_index = i * this->m() + j;

            if ((j + 1) != this->m()) {
                file << to_string_double_with_precision(this->elem(i, j),
                                                         num_accuracy, max_number_size) << char_delimiter;
            } else {
                file << to_string_double_with_precision(this->elem(i, j),
                                                         num_accuracy, max_number_size) << "\n";
            }
        }
    }

    file.close();

#else

    const char *charname = filename.c_str();
    size_t max_number_size = QConfig::instance().csv_max_number_size();
    size_t num_accuracy = QConfig::instance().csv_num_accuracy();

    int rank, size, root_id = 0;
    int file_exists = 1, end_not_empty = 0;
    long long A_n = this->n(), A_m = this->m();

    MPI_Status status;
    MPI_File file;

    // Существует ли файл.
    std::ifstream file_ending(filename, std::ifstream::ate);
    if (file_ending)
        file_ending.close();
    else
        file_exists = 0;

    const int delimiter_size = 1;
    const int one_elem_size = max_number_size + 1;
    const char char_delimiter = ',';
    std::string str_delimiter = ",";

    if (file_exists == 1) {
      MPI_File_open(MPI_COMM_WORLD, charname, MPI_MODE_APPEND | MPI_MODE_WRONLY,
                    MPI_INFO_NULL, &file);
    } else {
      MPI_File_open(MPI_COMM_WORLD, charname, MPI_MODE_CREATE | MPI_MODE_WRONLY,
                    MPI_INFO_NULL, &file);
    }

    MPI_Offset start_offset;
    MPI_File_get_position(file, &start_offset);
    MPI_Offset offset = 0;

    MPI_File_seek(file, start_offset, MPI_SEEK_SET);

    for (size_t i = 0; i < this->n(); i++) {
      for (size_t j = 0; j < this->m(); j++) {
        auto cur_index = this->get_index(i, j);
        if ((cur_index + 1) % this->LD() != 0) {
          auto num_str =
              to_string_double_with_precision(this->elem(i, j),
                                              num_accuracy, max_number_size) +
              char_delimiter;
          MPI_File_write(file, num_str.c_str(), num_str.length(), MPI_CHAR,
                        &status);
        } else {
          auto num_str = to_string_double_with_precision(
              this->elem(i, j), num_accuracy, max_number_size);
          num_str += "\n";
          MPI_File_write(file, num_str.c_str(), num_str.length(), MPI_CHAR,
                        &status);
        }
      }
    }
    MPI_File_close(&file);

#endif
}

template<>
void Matrix<COMPLEX>::write_to_csv_file(const std::string& filename) const {
#ifndef ENABLE_CLUSTER

    std::ofstream file(filename);

    size_t max_number_size = QConfig::instance().csv_max_number_size();
    size_t num_accuracy = QConfig::instance().csv_num_accuracy();

    const int delimiter_size = 1;
    const int one_elem_size = max_number_size + 1;
    const char char_delimiter = ',';

    for (size_t i = 0; i < this->n(); i++) {
        for (size_t j = 0; j < this->m(); j++) {
            auto cur_index = this->get_index(i, j);

            if ((j + 1) != this->m()) {
                file << to_string_complex_with_precision(this->elem(i, j),
                                                         num_accuracy, max_number_size) << char_delimiter;
            } else {
                file << to_string_complex_with_precision(this->elem(i, j),
                                                         num_accuracy, max_number_size) << "\n";
            }
        }
    }

    file.close();

#else
    const char *charname = filename.c_str();
    size_t max_number_size = QConfig::instance().csv_max_number_size();
    size_t num_accuracy = QConfig::instance().csv_num_accuracy();

    int rank, size, root_id = 0;
    int file_exists = 1, end_not_empty = 0;
    long long A_n = this->n(), A_m = this->m();

    MPI_Status status;
    MPI_File file;

    // Существует ли файл.
    std::ifstream file_ending(filename, std::ifstream::ate);
    if (file_ending)
    file_ending.close();
    else
    file_exists = 0;

    const int delimiter_size = 1;
    const int one_elem_size = 2 * max_number_size + 3;
    const char char_delimiter = ',';
    std::string str_delimiter = ",";

    if (file_exists == 1) {
      MPI_File_open(MPI_COMM_WORLD, charname, MPI_MODE_APPEND | MPI_MODE_WRONLY,
                    MPI_INFO_NULL, &file);
    } else {
      MPI_File_open(MPI_COMM_WORLD, charname, MPI_MODE_CREATE | MPI_MODE_WRONLY,
                    MPI_INFO_NULL, &file);
    }

    MPI_Offset start_offset;
    MPI_File_get_position(file, &start_offset);
    MPI_Offset offset = 0;

    MPI_File_seek(file, start_offset, MPI_SEEK_SET);
    for (size_t i = 0; i < this->n(); i++) {
      for (size_t j = 0; j < this->m(); j++) {
        auto cur_index = this->get_index(i, j);

        if ((cur_index + 1) % this->LD() != 0) {
          auto num_str =
              to_string_complex_with_precision(this->elem(i, j),
                                              num_accuracy, max_number_size) +
              char_delimiter;
          MPI_File_write(file, num_str.c_str(), num_str.length(), MPI_CHAR,
                        &status);
        } else {
          auto num_str = to_string_complex_with_precision(
              this->elem(i, j), num_accuracy, max_number_size);
          num_str += "\n";
          MPI_File_write(file, num_str.c_str(), num_str.length(), MPI_CHAR,
                        &status);
        }
      }
    }
    MPI_File_close(&file);

#endif
}

namespace {
    lapack_complex_double make_complex(const double a, const double b) {
        lapack_complex_double c;

        c.real = a;
        c.imag = b;

        return c;
    }
}

template<>
lapack_complex_double* Matrix<COMPLEX>::to_upper_lapack() const {
    lapack_complex_double* a;
    size_t index = 0;
    a = new lapack_complex_double [n_ * m_];
    for (size_t i = 0; i < n_; i++) {
        for (size_t j = i; j < m_; j++) {
            a[get_index(i, j)] = make_complex(mass_[get_index(i, j)].real(), mass_[get_index(i, j)].imag());
        }
    }

    return a;
}

template<>
lapack_complex_double* Matrix<COMPLEX>::to_lapack() const {
    lapack_complex_double* a;

    a = new lapack_complex_double [n_ * m_];
    for (size_t i = 0; i < n_; i++) {
        for (size_t j = 0; j < m_; j++) {
            a[get_index(i, j)] = make_complex(mass_[get_index(i, j)].real(), mass_[get_index(i, j)].imag());
        }
    }

    return a;
}

} // namespace QComputations
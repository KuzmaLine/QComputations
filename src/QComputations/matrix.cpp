#define MKL_Complex16 std::complex<double>

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

/*
extern "C" {
    void zgeadd_(const COMPLEX*, ILP_TYPE*, char*, const COMPLEX*, ILP_TYPE*, char*, COMPLEX*, ILP_TYPE*, ILP_TYPE*, ILP_TYPE*);
    void dgeadd_(const double*, ILP_TYPE*, char*, const double*, ILP_TYPE*, char*, double*, ILP_TYPE*, ILP_TYPE*, ILP_TYPE*);
    //dgeadd(this->data(), this->LD(), 'N', A.data(), A.LD(), 'N', res.data(), res.LD(), this->n_, this->m_);
}
*/
//#ifndef ENABLE_MPI
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

/*
template<>
Matrix<double> Matrix<double>::operator+(const Matrix<double>& A) const {
    assert(this->n_ == A.n_ and this->m_ == A.m_);
    Matrix<double> res(this->get_matrix_style(), A.n_, A.m_);

    char N = 'N';
    ILP_TYPE n = this->n();
    ILP_TYPE m = this->m();
    ILP_TYPE TLD = this->LD();
    ILP_TYPE ALD = A.LD();
    ILP_TYPE resLD = res.LD();
    dgeadd_(this->data(), &TLD, &N, A.data(), &ALD, &N, res.data(), &resLD, &n, &m);
    return res;
}

template<>
Matrix<COMPLEX> Matrix<COMPLEX>::operator+(const Matrix<COMPLEX>& A) const {
    assert(this->n_ == A.n_ and this->m_ == A.m_);
    Matrix<COMPLEX> res(this->get_matrix_style(), A.n_, A.m_);

    char N = 'N';
    ILP_TYPE n = this->n();
    ILP_TYPE m = this->m();
    ILP_TYPE TLD = this->LD();
    ILP_TYPE ALD = A.LD();
    ILP_TYPE resLD = res.LD();
    zgeadd_(this->data(), &TLD, &N, A.data(), &ALD, &N, res.data(), &resLD, &n, &m);
    return res;
}
*/

template<>
Matrix<double>& Matrix<double>::operator*=(double num) {
    ILP_TYPE iONE = 1;

    ILP_TYPE size = this->n() * this->m();
    dscal(&size, &num, this->data(), &iONE);

    return *this;
}

template<>
Matrix<COMPLEX>& Matrix<COMPLEX>::operator*=(COMPLEX num) {
    ILP_TYPE iONE = 1;

    ILP_TYPE size = this->n() * this->m();
    zscal(&size, &num, this->data(), &iONE);

    return *this;
}

template<>
Matrix<double>& Matrix<double>::operator/=(double num) {
    ILP_TYPE iONE = 1;
    double new_num = double(1)/num;

    ILP_TYPE size = this->n() * this->m();
    dscal(&size, &new_num, this->data(), &iONE);

    return *this;
}

template<>
Matrix<COMPLEX>& Matrix<COMPLEX>::operator/=(COMPLEX num) {
    ILP_TYPE iONE = 1;
    COMPLEX new_num = COMPLEX(1, 0) / num;

    ILP_TYPE size = this->n() * this->m();
    zscal(&size, &new_num, this->data(), &iONE);

    return *this;
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

    //std::cout << "HERE\n";

    // Существует ли файл.
    std::ifstream file_ending(filename, std::ifstream::ate);
    if (file_ending)
        file_ending.close();
    else
        file_exists = 0;

    //std::cout << "HERE1\n";
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
        //std::cout << i << " " << j << ", " << this->n() << " " << this->m() << std::endl;
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

//#else
/*
template<>
Matrix<double> Matrix<double>::operator* (const Matrix<double>& A) const {
    assert(m_ == A.n_);
    Matrix<double> res(this->get_matrix_style(), n_, A.m_);

    if (config::MULTIPLY_MODE == config::CANNON_MODE) {
        size_t n = n_, k = m_, m = A.m_;
        int rank, world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        mpi::make_command(COMMAND::CANNON_MULTIPLY);

        int grid_size = std::sqrt(world_size);
        int block_size = n / grid_size;

        if (n == m and m == k) {
            int bcast_data[4];
            bcast_data[0] = grid_size;
            bcast_data[1] = block_size;
            bcast_data[2] = n;
            bcast_data[3] = mpi::MPI_Datatype_ID::DOUBLE;

            MPI_Bcast(&bcast_data, 4, MPI_INT, mpi::ROOT_ID, MPI_COMM_WORLD);
            auto begin = std::chrono::steady_clock::now();
            mpi::Cannon_Multiply<double>(*this, A, res, grid_size, block_size, n);
            auto end = std::chrono::steady_clock::now();
            std::cout << "CANNON: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
            return res;
        }
        auto begin = std::chrono::steady_clock::now();

        auto end = std::chrono::steady_clock::now();
        std::cout << "COMMAND: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
        Matrix<double> L(*this);
        end = std::chrono::steady_clock::now();
        std::cout << "COPY L: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;

        Matrix<double>R(A);

        end = std::chrono::steady_clock::now();
        std::cout << "COPY R: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;

        size_t remove_rows = 0, remove_cols = 0, reduce_times = 0;

        if (m != n) {
            if (n < m) {
                L.add_rows(m - n);
                res.add_rows(m - n);
                remove_rows += m - n;
                n += m - n;
            } else {
                R.add_cols(n - m);
                res.add_cols(n - m);
                m += n - m;
                remove_cols += n - m;
            }
        }

        end = std::chrono::steady_clock::now();
        std::cout << "M N: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;

        if (n != k) {
            if (n < k) {
                L.add_rows(k - n);
                R.add_cols(k - n);
                res.expand(k - n);
                n += k - n;
                m += k - n;
                reduce_times += k - n;
            } else {
                L.add_cols(n - k);
                R.add_rows(n - k);
                k += n - k;
            }
        }

        end = std::chrono::steady_clock::now();
        std::cout << "N K: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;

        size_t additional_dims = n % grid_size == 0 ? 0 : grid_size - n % grid_size;

        if (additional_dims != 0) {
            L.expand(additional_dims);
            R.expand(additional_dims);
            res.expand(additional_dims);
            reduce_times += additional_dims;
        }

        block_size = L.size() / grid_size;
        int bcast_data[4];
        bcast_data[0] = grid_size;
        bcast_data[1] = block_size;
        bcast_data[2] = L.n();
        bcast_data[3] = mpi::MPI_Datatype_ID::DOUBLE;

        end = std::chrono::steady_clock::now();
        std::cout << "BEFORE BCAST: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
        MPI_Bcast(&bcast_data, 4, MPI_INT, mpi::ROOT_ID, MPI_COMM_WORLD);

        end = std::chrono::steady_clock::now();
        std::cout << "BEFORE: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;

        begin = std::chrono::steady_clock::now();
        mpi::Cannon_Multiply(L, R, res, grid_size, block_size, L.n());
        end = std::chrono::steady_clock::now();
        std::cout << "Cannon: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;


        begin = std::chrono::steady_clock::now();
        //double alpha = 1.0;
        //double betta = 0.0;
        //cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        //            n_, A.m_, m_, &alpha, mass_.data(), n_,
        //            A.mass_.data(), A.n_, &betta, res.mass_.data(), n_);

        //if (n < k) {
        //    res.reduce(k - n);
        //}

        if (reduce_times != 0) {
            res.reduce(reduce_times);
        }

        if (remove_rows != 0) {
            res.remove_rows(remove_rows);
        }

        if (remove_cols != 0) {
            res.remove_cols(remove_cols);
        }

        
        //if (m != n) {
        //    if (n < m) {
        //        res.remove_rows(m - n);
        //    } else {
        //        res.remove_cols(n - m);
        //    }
        //}

        end = std::chrono::steady_clock::now();
        std::cout << "AFTER: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
    } else if (config::MULTIPLY_MODE == config::COMMON_MODE) {
        double alpha = 1;
        double betta = 0;

        auto type = CblasRowMajor;
        if (!(res.is_c_style())) type = CblasColMajor;
        cblas_dgemm(type, CblasNoTrans, CblasNoTrans,
                    n_, A.m_, m_, alpha, mass_.data(),
                    this->LD(), A.mass_.data(), A.LD(), betta,
                    res.mass_.data(), res.LD());

    } else if (config::MULTIPLY_MODE == config::DIM_MODE) {
        size_t n = n_, k = m_, m = A.m_;
        int rank, world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        mpi::make_command(COMMAND::DIM_MULTIPLY);

        int bcast_data[3];
        bcast_data[0] = A.n();
        bcast_data[1] = A.m();
        bcast_data[2] = mpi::MPI_Datatype_ID::DOUBLE;
        MPI_Bcast(&bcast_data, 3, MPI_INT, mpi::ROOT_ID, MPI_COMM_WORLD);

        Matrix<double> R(A);
        MPI_Bcast(R.data(), k * n, MPI_DOUBLE_COMPLEX, mpi::ROOT_ID, MPI_COMM_WORLD);
        //mpi::Dim_Multiply(*this, A, res);
    }
#ifdef ENABLE_CLUSTER 
    else if (config::MULTIPLY_MODE == config::P_GEMM_MODE) {
        size_t n = n_, k = m_, m = A.m_;
        mpi::make_command(COMMAND::P_GEMM_MULTIPLY);

        int datatype = mpi::MPI_Datatype_ID::DOUBLE;
        MPI_Bcast(&datatype, 1, MPI_INT, mpi::ROOT_ID, MPI_COMM_WORLD);
        //mpi::parallel_dgemm(*this, A, res);
    }
#endif
    return res;
}


template<>
Matrix<COMPLEX> Matrix<COMPLEX>::operator* (const Matrix<COMPLEX>& A) const {
    assert(m_ == A.n_);
    Matrix<COMPLEX> res(this->get_matrix_style(), n_, A.m_);

    if (config::MULTIPLY_MODE == config::CANNON_MODE) {
        size_t n = n_, k = m_, m = A.m_;
        int rank, world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        mpi::make_command(COMMAND::CANNON_MULTIPLY);

        int grid_size = std::sqrt(world_size);
        int block_size = n / grid_size;

        if (n == m and m == k) {
            int bcast_data[4];
            bcast_data[0] = grid_size;
            bcast_data[1] = block_size;
            bcast_data[2] = n;
            bcast_data[3] = mpi::MPI_Datatype_ID::DOUBLE_COMPLEX;

            MPI_Bcast(&bcast_data, 4, MPI_INT, mpi::ROOT_ID, MPI_COMM_WORLD);
            auto begin = std::chrono::steady_clock::now();
            mpi::Cannon_Multiply<COMPLEX>(*this, A, res, grid_size, block_size, n);
            auto end = std::chrono::steady_clock::now();
            std::cout << "CANNON: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
            return res;
        }
        auto begin = std::chrono::steady_clock::now();

        auto end = std::chrono::steady_clock::now();
        std::cout << "COMMAND: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
        Matrix<COMPLEX> L(*this);
        end = std::chrono::steady_clock::now();
        std::cout << "COPY L: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;

        Matrix<COMPLEX>R(A);

        end = std::chrono::steady_clock::now();
        std::cout << "COPY R: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;

        size_t remove_rows = 0, remove_cols = 0, reduce_times = 0;

        if (m != n) {
            if (n < m) {
                L.add_rows(m - n);
                res.add_rows(m - n);
                remove_rows += m - n;
                n += m - n;
            } else {
                R.add_cols(n - m);
                res.add_cols(n - m);
                m += n - m;
                remove_cols += n - m;
            }
        }

        end = std::chrono::steady_clock::now();
        std::cout << "M N: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;

        if (n != k) {
            if (n < k) {
                L.add_rows(k - n);
                R.add_cols(k - n);
                res.expand(k - n);
                n += k - n;
                m += k - n;
                reduce_times += k - n;
            } else {
                L.add_cols(n - k);
                R.add_rows(n - k);
                k += n - k;
            }
        }

        end = std::chrono::steady_clock::now();
        std::cout << "N K: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;

        size_t additional_dims = n % grid_size == 0 ? 0 : grid_size - n % grid_size;

        if (additional_dims != 0) {
            L.expand(additional_dims);
            R.expand(additional_dims);
            res.expand(additional_dims);
            reduce_times += additional_dims;
        }

        block_size = L.size() / grid_size;
        int bcast_data[4];
        bcast_data[0] = grid_size;
        bcast_data[1] = block_size;
        bcast_data[2] = L.n();
        bcast_data[3] = mpi::MPI_Datatype_ID::DOUBLE_COMPLEX;

        end = std::chrono::steady_clock::now();
        std::cout << "BEFORE BCAST: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
        MPI_Bcast(&bcast_data, 4, MPI_INT, mpi::ROOT_ID, MPI_COMM_WORLD);

        end = std::chrono::steady_clock::now();
        std::cout << "BEFORE: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;

        begin = std::chrono::steady_clock::now();
        mpi::Cannon_Multiply(L, R, res, grid_size, block_size, L.n());
        end = std::chrono::steady_clock::now();
        std::cout << "Cannon: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;


        begin = std::chrono::steady_clock::now();
        //double alpha = 1.0;
        //double betta = 0.0;
        //cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        //            n_, A.m_, m_, &alpha, mass_.data(), n_,
        //            A.mass_.data(), A.n_, &betta, res.mass_.data(), n_);

        //if (n < k) {
        //    res.reduce(k - n);
        //}

        if (reduce_times != 0) {
            res.reduce(reduce_times);
        }

        if (remove_rows != 0) {
            res.remove_rows(remove_rows);
        }

        if (remove_cols != 0) {
            res.remove_cols(remove_cols);
        }


        //if (m != n) {
        //    if (n < m) {
        //        res.remove_rows(m - n);
        //    } else {
        //        res.remove_cols(n - m);
        //    }
        //}

        end = std::chrono::steady_clock::now();
        std::cout << "AFTER: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
    } else if (config::MULTIPLY_MODE == config::COMMON_MODE) {
        COMPLEX alpha(1, 0);
        COMPLEX betta(0, 0);

        auto type = CblasRowMajor;
        if (!(res.is_c_style())) type = CblasColMajor;
        cblas_zgemm(type, CblasNoTrans, CblasNoTrans,
                    n_, A.m_, m_, &alpha, mass_.data(),
                    this->LD(), A.mass_.data(), A.LD(), &betta,
                    res.mass_.data(), res.LD());

    } else if (config::MULTIPLY_MODE == config::DIM_MODE) {
        size_t n = n_, k = m_, m = A.m_;
        int rank, world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        mpi::make_command(COMMAND::DIM_MULTIPLY);

        int bcast_data[3];
        bcast_data[0] = A.n();
        bcast_data[1] = A.m();
        bcast_data[2] = mpi::MPI_Datatype_ID::DOUBLE_COMPLEX;
        MPI_Bcast(&bcast_data, 3, MPI_INT, mpi::ROOT_ID, MPI_COMM_WORLD);

        Matrix<COMPLEX> R(A);
        MPI_Bcast(R.data(), k * n, MPI_DOUBLE_COMPLEX, mpi::ROOT_ID, MPI_COMM_WORLD);
        mpi::Dim_Multiply(*this, A, res);
    }
#ifdef ENABLE_CLUSTER 
    else if (config::MULTIPLY_MODE == config::P_GEMM_MODE) {
        size_t n = n_, k = m_, m = A.m_;
        mpi::make_command(COMMAND::P_GEMM_MULTIPLY);

        int datatype = mpi::MPI_Datatype_ID::DOUBLE_COMPLEX;
        MPI_Bcast(&datatype, 1, MPI_INT, mpi::ROOT_ID, MPI_COMM_WORLD);
        //mpi::parallel_zgemm(*this, A, res);
    }
#endif
    return res;
}

#endif
*/

/*
template<>
Matrix<COMPLEX> Matrix<COMPLEX>::operator* <COMPLEX>(const Matrix<COMPLEX>& A) const {
    assert(m_ == A.n_);
    Matrix<COMPLEX> res(n_, A.m_, 0);

    for (size_t i = 0; i < n_; i++) {
        for (size_t j = 0; j < A.m_; j++) {
            for (size_t k = 0; k < m_; k++) {
                res.mass_[res.get_index(i, j)] += std::conj(mass_[this->get_index(i, k)]) * A.mass_[A.get_index(k, j)];
            }
        }
    }

    return res;
}

template<>
std::vector<COMPLEX> Matrix<COMPLEX>::operator* <COMPLEX>(const std::vector<COMPLEX>& v) const {
    assert(m_ == v.size());
    std::vector<COMPLEX> res(v.size(), COMPLEX(0, 0));

    size_t n = v.size();

    for (size_t i = 0; i < n; i++) {
        for (size_t k = 0; k < n; k++) {
            res[i] += std::conj(mass_[this->get_index(i, k)]) * v[k];
        }
    }

    return res;
}
*/

/* ---------------- FUNCTIONS ------------------------------ */

namespace {
    CBLAS_TRANSPOSE char_to_trans(char ch_trans) {
        switch(ch_trans) {
            case 'N': return CblasNoTrans;
            case 'T': return CblasTrans;
            case 'C': return CblasConjTrans;
            default: assert(false); // Некорректная операция транспозиции при перемножении
        }
    }
}

template<>
void optimized_multiply(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C,
                        double alpha, double betta, char trans_A, char trans_B) {
    assert(A.m() == B.n());
    assert(A.n() == C.n() and B.m() == C.m());

    auto type = CblasRowMajor;
    if (!(A.is_c_style())) type = CblasColMajor;
    cblas_dgemm(type, char_to_trans(trans_A), char_to_trans(trans_B),
                A.n(), B.m(), A.m(), alpha, A.data(),
                A.LD(), B.data(), B.LD(), betta,
                C.data(), C.LD());
}

template<>
void optimized_multiply(const Matrix<COMPLEX>& A, const Matrix<COMPLEX>& B, Matrix<COMPLEX>& C,
                        COMPLEX alpha, COMPLEX betta, char trans_A, char trans_B) {
    assert(A.m() == B.n());
    assert(A.n() == C.n() and B.m() == C.m());

    auto type = CblasRowMajor;
    if (!(A.is_c_style())) type = CblasColMajor;
    cblas_zgemm(type, char_to_trans(trans_A), char_to_trans(trans_B),
                A.n(), B.m(), A.m(), &alpha, A.data(),
                A.LD(), B.data(), B.LD(), &betta,
                C.data(), C.LD());
}

std::vector<Matrix<COMPLEX>> OPT_Runge_Kutt_4(const std::vector<double>& x,
                                        const Matrix<COMPLEX>& y0,
                                        std::function<void(double, const Matrix<COMPLEX>&, Matrix<COMPLEX>&)> f) {
    size_t len = x.size();
    size_t dim = y0.n();
    std::vector<Matrix<COMPLEX>> y(len, Matrix<COMPLEX>(y0.matrix_style(), dim, dim));
    y[0] = y0;

    Matrix<COMPLEX> k1(y0.matrix_style(), dim, dim);
    Matrix<COMPLEX> k2(y0.matrix_style(), dim, dim);
    Matrix<COMPLEX> k3(y0.matrix_style(), dim, dim);

    for (size_t i = 0; i < len - 1; i++) {
        //if (i % (len / 100) == 0) std::cout << i << " " << len << std::endl;
        //std::cout << i << " " << y[i] << " ";
        double h = x[i + 1] - x[i];

        f(x[i], y[i], k1);
        k1 *= (h / 2.0);
        k1 += y[i];
        f(x[i] + h / 2.0, k1, k2);
        k2 *= (h / 2.0);
        k2 += y[i];
        f(x[i] + h / 2.0, k2, k3);
        k3 *= h;
        k3 += y[i];
        f(x[i] + h, k3, y[i + 1]);
        k1 -= y[i];
        k2 -= y[i];
        k3 -= y[i];
        
        y[i + 1] *= (h / 6.0);
        y[i + 1] += y[i];

        k1 /= 3.0;
        y[i + 1] += k1;
        k2 *= (double(2) / double(3));
        y[i + 1] += k2;
        k3 /= 3.0;
        y[i + 1] += k3;
        //y[i + 1] = y[i]  + (k1 + (k2 + k3) * 2 + k4) * (h / 6.0);
        //std::cout << h << " " << y[i + 1] << " " << 2 * x[i + 1] << std::endl;
    }

    return y;
}

std::vector<Matrix<COMPLEX>> OPT_Runge_Kutt_2(const std::vector<double>& x,
                                        const Matrix<COMPLEX>& y0,
                                        std::function<void(double, const Matrix<COMPLEX>&, Matrix<COMPLEX>&)> f) {
    size_t len = x.size();
    size_t dim = y0.n();
    std::vector<Matrix<COMPLEX>> y(len, Matrix<COMPLEX>(y0.matrix_style(), dim, dim));
    y[0] = y0;

    Matrix<COMPLEX> k1(y0.matrix_style(), dim, dim);

    for (size_t i = 0; i < len - 1; i++) {
        double h = x[i + 1] - x[i];

        f(x[i], y[i], k1);
        k1 *= h;
        k1 += y[i];
        f(x[i] + h, k1, y[i + 1]);
        k1 -= y[i];
        k1 /= COMPLEX(2.0, 0);
        y[i + 1] *= COMPLEX(h / 2.0, 0);
        y[i + 1] += k1;
        y[i + 1] += y[i];
        //y[i + 1] = y[i]  + (k1 + k2) * (h / 2.0);
        //std::cout << h << " " << y[i + 1] << " " << 2 * x[i + 1] << std::endl;
    }

    return y;
}


/* --------------------- DON'T TOUCH ------------------------*/

namespace {
    lapack_complex_double make_complex(const double a, const double b) {
        lapack_complex_double c(a, b);

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
            //a[index++] = make_complex(mass_[get_index(i, j)].real(), mass_[get_index(i, j)].imag());
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
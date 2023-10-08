// !!!!!!!!!!!!!!!!!!!!!! TEST EXAMPLE !!!!!!!!!!!!!!!!!!!

#include "/home/kuzmaline/Quantum/diploma/src/QComputations_CPU_CLUSTER_NO_PLOTS.hpp"
#include <complex>
#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>
#include <sstream>

using COMPLEX = std::complex<double>;

std::string to_string_complex_with_precision(const COMPLEX& a_value, const int n, int max_number_size)
{
    std::ostringstream out;
    out.precision(n);
    out << ((a_value.real() >= 0) ? "+" : "-") << std::setfill('0') << std::setw(max_number_size) << std::fixed << std::abs(a_value.real());
    out << ((a_value.imag() >= 0) ? "+" : "-") << std::setfill('0') << std::setw(max_number_size) << std::fixed << std::abs(a_value.imag());
    out << "j";
    return std::move(out).str();
}

void cwfpcsv(QComputations::BLOCKED_Matrix<COMPLEX> A, const std::string &filename, int num_accuracy = 21, int max_number_size = 50) {
  const char *charname = filename.c_str();

  remove(charname);

  MPI_Offset offset;
  MPI_File file;
  MPI_Status status;
  MPI_File_open(MPI_COMM_WORLD, charname, MPI_MODE_CREATE | MPI_MODE_WRONLY,
                MPI_INFO_NULL, &file);

  const int delimiter_size = 1;
  const int one_elem_size = 2 * max_number_size + 3;
  const char char_delimiter = ',';

  for (size_t i = 0; i < A.local_n(); i++) {
    for (size_t j = 0; j < A.local_m(); j++) {
      auto cur_index = A.get_global_row(i) * A.m() + A.get_global_col(j);
      offset = (one_elem_size + delimiter_size) * sizeof(char) * cur_index;
      MPI_File_seek(file, offset, MPI_SEEK_SET);
      if ((cur_index + 1) % A.m() != 0) {
        auto num_str = to_string_complex_with_precision(A.data()[j * A.local_n() + i], num_accuracy, max_number_size) + char_delimiter;
        //if (cur_index != 0) std::cout << num_str << " " << num_str.length() << " " << offset / cur_index << std::endl;
        MPI_File_write(file, num_str.c_str(), num_str.length(),
                      MPI_CHAR, &status);
      } else if (cur_index + 1 == A.m() * A.n()) {
        auto num_str = to_string_complex_with_precision(A.data()[j * A.local_n() + i], num_accuracy, max_number_size) + "\n";
        MPI_File_write(file, num_str.c_str(), num_str.length(),
                      MPI_CHAR, &status);
      } else {
        auto num_str = to_string_complex_with_precision(A.data()[j * A.local_n() + i], num_accuracy, max_number_size) + "\n";
        MPI_File_write(file, num_str.c_str(), num_str.length(),
                      MPI_CHAR, &status);
      }
    }
  }

  MPI_File_close(&file);
}

int main(int argc, char **argv) {
  using namespace QComputations;
  MPI::Init(argc, argv);
  const std::string filename = "matrix.csv";
  const int n = 4;
  const int m = 6;
  const int k = 4;
  int ctxt;
  mpi::init_grid(ctxt);

  std::function<COMPLEX(size_t i, size_t j)> func_2 = {
      [](size_t i, size_t j) { return 10 * int(i) - int(j); }};

  BLOCKED_Matrix<COMPLEX> K(ctxt, GE, m, k, func_2);

  K.show(mpi::ROOT_ID);

  cwfpcsv(K, filename);

  MPI::Finalize();

  return 0;
}

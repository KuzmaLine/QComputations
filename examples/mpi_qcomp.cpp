#include "/home/kuzmaline/Quantum/diploma/src/QComputations_CPU_CLUSTER_NO_PLOTS.hpp"
#include <complex>
#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>

using namespace QComputations;
using COMPLEX = std::complex<double>;

// Array alike [i1, j1, i2, j1, i3, j1, i4, j1, i1, j2, i2, j2, i3, j2...]
int *get_global_indeces(BLOCKED_Matrix<COMPLEX> A) {
  const int num_elems = A.get_local_matrix().get_mass().size();
  const int cols = A.get_local_matrix().n();
  const int rows = A.get_local_matrix().m();
  int *indeces_mas = (int *)malloc(num_elems * sizeof(int));

  int straight_index = 0;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      indeces_mas[straight_index++] =
          A.n() * A.get_global_row(j) + A.get_global_col(i);
    }
  }

  return indeces_mas;
}

void cwfpcsv(BLOCKED_Matrix<COMPLEX> A, const std::string &filename,
             const int *desc) {
  const char *charname = filename.c_str();
  const int rank = MPI::COMM_WORLD.Get_rank();
  const int nprocs = MPI::COMM_WORLD.Get_size();

  const int num_indeces = A.get_local_matrix().get_mass().size();
  const int global_cols = A.n();

  int *indeces = get_global_indeces(A);

  auto localArray = A.get_local_matrix();
  // printf("%d - %d, %d\n", localArray.n(), localArray.m(),
  //     (A.get_global_row(0) + A.n() * A.get_global_col(0)));

  /*/////////////////////
  int myrank = 0;
  while (myrank < nprocs) {
    if (myrank == rank) {
      for (int i = 0; i < num_indeces; ++i)
        printf("[%d] ", indeces[i]);
      printf("\n\n");
    }
    myrank++;
    MPI_Barrier(MPI::COMM_WORLD);
  }
  /////////////////////*/

  remove(charname);
  
  MPI_Offset offset;
  MPI_File file;
  MPI_Status status;
  MPI_File_open(MPI_COMM_WORLD, charname, MPI_MODE_CREATE | MPI_MODE_WRONLY,
                MPI_INFO_NULL, &file);

  const int real_num_size = 21, delimiter_size = 1;
  const int one_elem_size = real_num_size + delimiter_size;

  char *arr_str = (char *)malloc((one_elem_size * 2 + 1) * sizeof(char));
  int cur_index;

  for (int i = 0; i < num_indeces; ++i) {
    cur_index = indeces[i];
    offset = (2 * one_elem_size + 1) * sizeof(char) * cur_index;
    MPI_File_seek(file, offset, MPI_SEEK_SET);
    if ((cur_index + 1) % global_cols != 0) {
      sprintf(arr_str, "%021.15e+%021.15ej,", localArray.data()[i].real(),
              localArray.data()[i].imag());
      MPI_File_write(file, arr_str, one_elem_size + one_elem_size + 1, MPI_CHAR,
                     &status);
    } else {
      sprintf(arr_str, "%021.15e+%021.15ej\n", localArray.data()[i].real(),
              localArray.data()[i].imag());
      MPI_File_write(file, arr_str, one_elem_size + real_num_size + 1 + 1, MPI_CHAR,
                     &status);
    }
  }

  MPI_File_close(&file);
  free(indeces);
  free(arr_str);
}

int main(int argc, char **argv) {
  MPI::Init(argc, argv);
  // const int rank = MPI::COMM_WORLD.Get_rank();
  const std::string filename = "matrix.csv";
  const int n = 4;
  const int m = 6;
  const int k = 4;
  int ctxt;
  mpi::init_grid(ctxt);

  std::function<COMPLEX(size_t i, size_t j)> func = {
      [](size_t i, size_t j) { return i + j + i % 3 + j % 2 + 1; }};
  std::function<COMPLEX(size_t i, size_t j)> func_2 = {
      [](size_t i, size_t j) { return int(i) - int(j); }};

  BLOCKED_Matrix<COMPLEX> M(ctxt, GE, n, m, func);
  BLOCKED_Matrix<COMPLEX> K(ctxt, GE, m, k, func_2);

  // std::cout << rank << " - " << M.local_n() << " " << M.local_m() << " "
  // << M.NB() << " " << M.MB() << std::endl; // M.print_distributed(ctxt, "M");
  M.show(mpi::ROOT_ID);
  //K.show(mpi::ROOT_ID);

  cwfpcsv(M, filename, &ctxt);

  MPI::Finalize();

  return 0;
}

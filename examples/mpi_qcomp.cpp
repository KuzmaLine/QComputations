#include "/home/kuzmaline/Quantum/diploma/src/QComputations_CPU_CLUSTER_NO_PLOTS.hpp"
#include <algorithm>
#include <cassert>
#include <complex>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <sstream>
#include <string>
#include <vector>

using COMPLEX = std::complex<double>;

// Сдвиги внутри матрицы офсетов.
  namespace of {
    int ns = 0, // Номер строки.
    bg = 1, // Офсет начала строки.
    md = 2, // Офсет места, куда писать матрицу.
    en = 3; // Офсет конца строки.
  }

std::string to_string_complex_with_precision(const COMPLEX &a_value,
                                             const int n, int max_number_size) {
  std::ostringstream out;
  out.precision(n);
  out << ((a_value.real() >= 0) ? "+" : "-") << std::setfill('0')
      << std::setw(max_number_size) << std::fixed << std::abs(a_value.real());
  out << ((a_value.imag() >= 0) ? "+" : "-") << std::setfill('0')
      << std::setw(max_number_size) << std::fixed << std::abs(a_value.imag());
  out << "j";
  return std::move(out).str();
}

std::string vector_to_string(std::vector<std::string> inp) {
  std::ostringstream out;
  for (auto i : inp) {
    if (i.empty())
      break;
    out << i << '\n';
  }
  return std::move(out).str();
}

void cwfippcsv(QComputations::BLOCKED_Matrix<COMPLEX> A,
               const std::string &filename, int row_place, int col_place,
               int num_accuracy = 21, int max_number_size = 50) {
  const char *charname = filename.c_str();

  int rank, size, root_id = 0;
  int file_exists = 1, end_not_empty = 0;
  long long num_lines_to_mod = 0, A_n = A.n(), A_m = A.m();

  MPI_Status status;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Offset global_offset =
                 0, // Офсет от начала до 1 строки, в которую писать.
      num_lines = 0, // Число всех строк файла.
      num_lines_left = 0, // Число строк после начала записи матрицы.
      num_elems_to_mod = 0, // Число элементов в строках, в которые пишем.
      mas_offsets[4 * A_n], // Номер строки и нужные офсеты внутри строк.
      line_begin_size[A_n] = {0}, // Офсеты от начала строк до записи.
      line_others_offset[A_n] = {0}; // Сохранение офсетов начала всех.
  // строк, нужно при записи матрицы, т.к. отступы в строках разные.
  MPI_File file;

  std::string vector_string;

  if (rank == root_id) {
    std::ifstream file_ending(filename, std::ifstream::ate);
    if (file_ending) {
      std::ifstream test_file(filename);
      num_lines = std::count(std::istreambuf_iterator<char>(test_file),
                             std::istreambuf_iterator<char>(), '\n');
      num_lines_left = num_lines - row_place;
      printf("%lld lines\n\n", num_lines);

      file_ending.close();
      test_file.close();
      if (num_lines < row_place) {
        printf("Bad number of row during writing to file!\n\n");
        assert(false);
        return;
      }
    } else {
      file_exists = 0;
    }
  }
  MPI_Bcast(&file_exists, 1, MPI_INT, root_id, MPI_COMM_WORLD);
  MPI_Bcast(&num_lines, 1, MPI_INT, root_id, MPI_COMM_WORLD);

  const int delimiter_size = 1;
  const int one_elem_size = 2 * max_number_size + 3;
  const char char_delimiter = ',';
  std::string str_delimiter = ",";

  long long lines_left_save = 0, num_lines_save = 0;
  if (rank == root_id) {
    num_lines_save = A_n;
    lines_left_save = num_lines_left - A_n;
    if (lines_left_save < 0)
      lines_left_save = 0;
  }
  // Если создать эти массивы в ифе, компилятор будет ругаться, поэтому процессы
  // создают массивы с 0 элементов. А vector.reserve не работает и роняет
  // прогу.
  std::vector<std::string> chunk(
      lines_left_save); // Массив строк после записанной матрицы.
  std::vector<std::string> lines_read(
      num_lines_save); // Массив строк, в которые пишем матрицу.

  if (file_exists == 1) {
    if (rank == root_id) {
      num_lines = A_n;
      std::string next_line;
      std::ifstream target_file(filename);

      for (int i = 0; i < row_place; ++i)
        if (!std::getline(target_file, next_line)) {
          printf("Bad number of row during writing to file!\n\n");
          assert(false);
          return;
        }
      global_offset = target_file.tellg();

      // Ищем офсеты начала и конца строк.
      // IF GETLINE WILL PERFORM AFTER I < NUM_LINES, IT WILL LOOSE ONE LINE!
      for (int i = 0; (i < num_lines) && (std::getline(target_file, next_line));
           ++i) {
        lines_read[i] = next_line;
        if (i == 0) {
          mas_offsets[i * 4 + of::bg] = 0;
          mas_offsets[i * 4 + of::en] = next_line.length() + 1;
        } else {
          mas_offsets[i * 4 + of::bg] = mas_offsets[(i - 1) * 4 + of::en];
          mas_offsets[i * 4 + of::en] =
              mas_offsets[(i - 1) * 4 + of::en] + next_line.length() + 1;
        }

        // Ищем офсет элемента, после которого писать матрицу в данной строке.
        int delim_number = col_place;
        long long index = 0, next_index = 0;
        if (delim_number != 0)
          while ((next_index = next_line.find(str_delimiter, index)) !=
                 std::string::npos) {
            index = next_index + str_delimiter.length();
            if (--delim_number == 0)
              break;
          }
        if (next_index != std::string::npos)
          end_not_empty = 1;
        else
          index = next_line.length();
        num_elems_to_mod = target_file.tellg() - global_offset;
        mas_offsets[i * 4 + of::md] = index;
        mas_offsets[i * 4 + of::ns] = i; // Номер строки.

        line_begin_size[i] = index;
        line_others_offset[i] = mas_offsets[i * 4 + of::bg];
      }
      for (int i = 0; i < num_lines; ++i)
        printf("Line: %s[end]\n\n", lines_read[i].c_str());

      for (int i = 0; i < num_lines * 4; i += 4)
        printf("From %lld through %lld to %lld\n", mas_offsets[i + of::bg],
               mas_offsets[i + of::md], mas_offsets[i + of::en]);
      vector_string = vector_to_string(lines_read);
      printf("\n%s\n", vector_string.c_str());

      for (int i = 0; std::getline(target_file, next_line); ++i) {
        chunk[i] = next_line;
        printf("%s\n", next_line.c_str());
      }

      target_file.close();
    }

    MPI_Bcast(&num_elems_to_mod, 1, MPI_LONG_LONG, root_id, MPI_COMM_WORLD);
    MPI_Bcast(&end_not_empty, 1, MPI_INT, root_id, MPI_COMM_WORLD);
    MPI_Bcast(&global_offset, 1, MPI_LONG_LONG, root_id, MPI_COMM_WORLD);
    MPI_Bcast(&line_begin_size, A_n, MPI_LONG_LONG, root_id, MPI_COMM_WORLD);
    MPI_Bcast(&line_others_offset, A_n, MPI_LONG_LONG, root_id, MPI_COMM_WORLD);

    num_lines_to_mod = (num_lines) / size;
    num_lines_to_mod += (rank == size - 1) ? (num_lines % size) : 0;

    // Разбиваем массивы офсетов и строк для рассылки.
    int send_counts[size], offset_offsets[size], offset_counts[size],
        string_counts[size], string_offsets[size];
    MPI_Offset local_offset = 0;
    int cur_line = 0;
    long long send_offset = 0;
    for (int i = 0; i < size; ++i) {
      send_counts[i] =
          num_lines / size + ((i == size - 1) ? num_lines % size : 0);
      offset_counts[i] = send_counts[i] * 4;
      if (i == 0)
        local_offset = 0;
      else
        local_offset += send_counts[i - 1];

      offset_offsets[i] = local_offset * 4;

      if (rank == root_id) {
        string_offsets[0] = 0;

        int lines_num = send_counts[i];
        while (lines_num > 0) {
          ++cur_line;
          --lines_num;
        }
        send_offset = mas_offsets[(cur_line) * 4 + of::bg];

        if (i != size - 1)
          string_offsets[i + 1] = send_offset;

        send_offset = mas_offsets[(cur_line - 1) * 4 + of::en];

        if (i == 0)
          string_counts[i] = send_offset;
        else
          string_counts[i] =
              send_offset - (string_offsets[i - 1] + string_counts[i - 1]);
      }
    }
    for (int i = 0; rank == root_id && i < size; ++i) {
      printf("%d: Send %d chars from %d, and %d offsets from %d\n", rank,
             string_counts[i], string_offsets[i], offset_counts[i],
             offset_offsets[i]);
    }
    long long local_offsets[offset_counts[rank]];

    MPI_Bcast(&string_counts, size, MPI_LONG_LONG, root_id, MPI_COMM_WORLD);
    MPI_Bcast(&string_offsets, size, MPI_LONG_LONG, root_id, MPI_COMM_WORLD);
    // printf("LOC %d from %d\n", string_counts[rank], string_offsets[rank]);

    char local_lines[string_counts[rank] + 1];
    const char *mas_char_string = vector_string.c_str();

    MPI_Scatterv(&mas_offsets, offset_counts, offset_offsets, MPI_LONG_LONG,
                 &local_offsets, num_lines * 4, MPI_LONG_LONG, root_id,
                 MPI_COMM_WORLD);

    /*for (int i = 0; i < num_lines_to_mod * 4; i += 4)
      printf("Offset %d: %lld + %lld\n", rank, global_offset,
      local_offsets[i]);*/

    MPI_Scatterv(mas_char_string, string_counts, string_offsets, MPI_CHAR,
                 &local_lines, num_elems_to_mod, MPI_CHAR, root_id,
                 MPI_COMM_WORLD);
    local_lines[string_counts[rank]] = '\0';

    printf("Str %d: %s[end]\n\n", rank, local_lines);

    // Добавляем свободное место в файле под строки матрицы.
    MPI_File_open(MPI_COMM_WORLD, charname, MPI_MODE_WRONLY, MPI_INFO_NULL,
                  &file);
    MPI_Offset offset = 0, line_offset = 0;
    for (int i = 0; i < offset_counts[rank]; i += 4) {
      int str_num = local_offsets[i + of::ns];
      offset = global_offset + local_offsets[i + of::bg] +
               str_num * (one_elem_size + delimiter_size) * A_m;
      MPI_File_seek(file, offset, MPI_SEEK_SET);
      // printf("%d - %lld\n", str_num, offset);
      if (local_offsets[i + of::md] != 0) {
        MPI_File_write(file, local_lines + line_offset, local_offsets[i + of::md],
                       MPI_CHAR, &status);
        if (end_not_empty == 0)
          MPI_File_write(file, &char_delimiter, 1, MPI_CHAR, &status);
      }
      MPI_File_seek(file,
                    offset + local_offsets[i + of::md] +
                        (one_elem_size + delimiter_size) * A_m,
                    MPI_SEEK_SET);
      // printf("ELEMPLACE: %lld > %lld\n", local_offsets[i + of::md],
      // (one_elem_size + delimiter_size) * A_m);
      MPI_File_write(file, local_lines + line_offset + local_offsets[i + of::md],
                     local_offsets[i + of::en] -
                         (local_offsets[i + of::bg] + local_offsets[i + of::md]),
                     MPI_CHAR, &status);
      line_offset += (local_offsets[i + of::en] - local_offsets[i + of::bg]);
    }
    // MPI_File_close(&file);
  } else {
    MPI_File_open(MPI_COMM_WORLD, charname, MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &file);
  }

  // MPI_Offset row_offset = 0; // отступ для каждой строки до нужного столбца
  MPI_Offset start_offset =
      global_offset +
      (1 - end_not_empty); // отступ до 1 строки, в которую записываем
  if (file_exists == 0)
    start_offset = 0;
  MPI_Offset offset = 0;

  for (size_t i = 0; i < A.local_n(); i++) {
    for (size_t j = 0; j < A.local_m(); j++) {
      auto cur_index = A.get_global_row(i) * A_m + A.get_global_col(j);
      // printf("SDVIG: %d + %lld\n", line_begin_size[A.get_global_row(i)],
      // line_others_offset[A.get_global_row(i)]);
      offset = start_offset + line_begin_size[A.get_global_row(i)] +
               line_others_offset[A.get_global_row(i)] +
               (one_elem_size + delimiter_size) * cur_index;
      MPI_File_seek(file, offset, MPI_SEEK_SET);
      if ((cur_index + 1) % A_m != 0) {
        auto num_str =
            to_string_complex_with_precision(A.data()[j * A.local_n() + i],
                                             num_accuracy, max_number_size) +
            char_delimiter;
        // if (cur_index != 0) std::cout << num_str << " " << num_str.length()
        // << " " << offset / cur_index << std::endl;
        MPI_File_write(file, num_str.c_str(), num_str.length(), MPI_CHAR,
                       &status);
      } else if (cur_index + 1 == A_m * A_n) {
        auto num_str = to_string_complex_with_precision(
            A.data()[j * A.local_n() + i], num_accuracy, max_number_size);
        if (!file_exists)
          num_str += "\n";
        else if (end_not_empty)
          num_str += char_delimiter;
        MPI_File_write(file, num_str.c_str(), num_str.length(), MPI_CHAR,
                       &status);
      } else {
        auto num_str = to_string_complex_with_precision(
            A.data()[j * A.local_n() + i], num_accuracy, max_number_size);
        if (!file_exists)
          num_str += "\n";
        else if (end_not_empty)
          num_str += char_delimiter;
        MPI_File_write(file, num_str.c_str(), num_str.length(), MPI_CHAR,
                       &status);
      }
    }
  }
  if (file_exists == 1) {
    // Записываем концовку файла.
    MPI_Offset written_elems =
        A_n * A_m * (one_elem_size + delimiter_size) * sizeof(char);
    std::ofstream out_file(filename, std::ios::app);
    out_file.seekp(written_elems + num_elems_to_mod + global_offset,
                   std::ios_base::beg);

    for (int i = 0; i < chunk.size(); ++i) {
      out_file << chunk[i];
      out_file << '\n';
    }
    out_file.close();
  }
  MPI_File_close(&file);
}

int main(int argc, char **argv) {
  using namespace QComputations;
  MPI::Init(argc, argv);
  const std::string filename = "matrix_2.csv";
  const int n = 4;
  const int m = 6;
  const int k = 4;
  int ctxt;
  mpi::init_grid(ctxt);

  std::function<COMPLEX(size_t i, size_t j)> func_2 = {
      [](size_t i, size_t j) { return 10 * int(i) - int(j); }};

  BLOCKED_Matrix<COMPLEX> K(ctxt, GE, m, k, func_2);

  K.show(mpi::ROOT_ID);
  if (argc < 3) {
    printf("No command line arguments m & n for start coords\n");
    return 0;
  }
  int m_place = atoi(argv[1]), n_place = atoi(argv[2]);

  cwfippcsv(K, filename, m_place, n_place);

  MPI::Finalize();

  return 0;
}

#include "plot.hpp"

#include <filesystem>
#include <fstream>

#include "functions.hpp"

#ifdef ENABLE_MATPLOTLIB
#include "matplotlibcpp.hpp"
#endif

namespace QComputations {

    namespace fs = std::filesystem;

#ifndef ENABLE_CLUSTER

    void check_dir(std::string& dir, bool remove_is_exist = false) {
        if (dir != "") {
            if (remove_is_exist and fs::is_directory(dir)) {
                fs::remove_all(dir);
            }
            fs::create_directory(dir);
        } else {
            dir = ".";
        }
    }

    void make_probs_files(const Hamiltonian& H,
                          const Probs& probs,
                          const std::vector<double>& time_vec,
                          const BasisType<Basis_State>& basis,
                          std::string dir) {
        check_dir(dir, true);

        hamiltonian_to_file("hamiltonian.csv", H, dir);
        basis_to_file("basis.csv", basis, dir);
        time_vec_to_file("time.csv", time_vec, dir);
        probs_to_file("probs.csv", probs, dir);
    }

#else
#ifdef ENABLE_MPI

    void check_dir(std::string& dir, ILP_TYPE main_rank, bool remove_is_exist = false) {
        if (dir != "") {
            ILP_TYPE rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);

            if (rank == main_rank) {
                if (remove_is_exist and fs::is_directory(dir)) {
                    fs::remove_all(dir);
                }
                fs::create_directory(dir);
            }
        } else {
            dir = ".";
        }
    }

    void make_probs_files(const Hamiltonian& H,
                          const Probs& probs,
                          const std::vector<double>& time_vec,
                          const BasisType<Basis_State>& basis,
                          std::string dir,
                          ILP_TYPE main_rank) {
        check_dir(dir, main_rank, true);

        hamiltonian_to_file("hamiltonian.csv", H, dir, main_rank);
        basis_to_file("basis.csv", basis, dir, main_rank);
        time_vec_to_file("time.csv", time_vec, dir, main_rank);
        probs_to_file("probs.csv", probs, dir, main_rank);
    }

#endif
#endif

#ifndef ENABLE_CLUSTER

    void hamiltonian_to_file(const std::string& filename, const Hamiltonian& H, std::string dir) {
        check_dir(dir);
        H.write_to_csv_file(dir + "/" + filename);
    }

    void probs_to_file(const std::string& filename, const Probs& probs, std::string dir) {
        check_dir(dir);
        auto probs_hermit = probs.transpose();
        probs_hermit.write_to_csv_file(dir + "/" + filename);
    }

    void basis_to_file(const std::string& filename, const BasisType<Basis_State>& basis, std::string dir) {
        check_dir(dir);

        auto filepath = dir + "/" + filename;

        std::ofstream file(filepath);

        for (size_t i = 0; i < basis.size(); i++) {
            std::string state_str = "\"" + get_state_from_basis<Basis_State>(basis, i)->to_string() + "\"";

            if (i != basis.size() - 1) {
                state_str += ",";
            }

            file << state_str;

            if (i == basis.size() - 1) {
                file << "\n";
            }
        }

        file.close();
    }

    void time_vec_to_file(const std::string& filename, const std::vector<double>& time_vec, std::string dir) {
        check_dir(dir);

        auto filepath = dir + "/" + filename;

        auto num_length = QConfig::instance().csv_max_number_size();
        auto accuracy = QConfig::instance().csv_num_accuracy();

        std::ofstream file(filepath);

        for (size_t i = 0; i < time_vec.size(); i++) {
            std::string num_str = to_string_double_with_precision(time_vec[i], accuracy, num_length);

            if (i != time_vec.size() - 1) {
                num_str += ",";
            }

            file << num_str;

            if (i == time_vec.size() - 1) {
                file << "\n";
            }
        }

        file.close();
    }

#endif

#ifdef ENABLE_MPI
#ifdef ENABLE_CLUSTER

    void hamiltonian_to_file(const std::string& filename, const Hamiltonian& H, std::string dir, ILP_TYPE main_rank) {
        ILP_TYPE rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if (rank == main_rank) {
            check_dir(dir, main_rank);
            H.write_to_csv_file(dir + "/" + filename);
        }
    }

    void probs_to_file(const std::string& filename, const Probs& probs, std::string dir, ILP_TYPE main_rank) {
        ILP_TYPE rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if (rank == main_rank) {
            check_dir(dir, main_rank);
            auto probs_hermit = probs.transpose();
            probs_hermit.write_to_csv_file(dir + "/" + filename);
        }
    }

    void hamiltonian_to_file(const std::string& filename,
                             const BLOCKED_Hamiltonian& H,
                             std::string dir,
                             ILP_TYPE main_rank) {
        check_dir(dir, main_rank);
        H.write_to_csv_file(dir + "/" + filename);
    }

    void basis_to_file(const std::string& filename,
                       const BasisType<Basis_State>& basis,
                       std::string dir,
                       ILP_TYPE main_rank) {
        ILP_TYPE rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if (rank == main_rank) {
            check_dir(dir, main_rank);

            auto filepath = dir + "/" + filename;

            std::ofstream file(filepath);

            for (size_t i = 0; i < basis.size(); i++) {
                std::string state_str = "\"" + get_state_from_basis<Basis_State>(basis, i)->to_string() + "\"";

                if (i != basis.size() - 1) {
                    state_str += ",";
                }

                file << state_str;

                if (i == basis.size() - 1) {
                    file << "\n";
                }
            }

            file.close();
        }
    }

    void time_vec_to_file(const std::string& filename,
                          const std::vector<double>& time_vec,
                          std::string dir,
                          ILP_TYPE main_rank) {
        ILP_TYPE rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if (rank == main_rank) {
            check_dir(dir, main_rank);

            auto filepath = dir + "/" + filename;

            auto num_length = QConfig::instance().csv_max_number_size();
            auto accuracy = QConfig::instance().csv_num_accuracy();

            std::ofstream file(filepath);

            for (size_t i = 0; i < time_vec.size(); i++) {
                std::string num_str = to_string_double_with_precision(time_vec[i], accuracy, num_length);

                if (i != time_vec.size() - 1) {
                    num_str += ",";
                }

                file << num_str;

                if (i == time_vec.size() - 1) {
                    file << "\n";
                }
            }

            file.close();
        }
    }

    void probs_to_file(const std::string& filename, const BLOCKED_Probs& probs, std::string dir, ILP_TYPE main_rank) {
        check_dir(dir, main_rank);
        auto probs_hermit = probs.hermit();
        probs_hermit.write_to_csv_file(dir + "/" + filename);
    }

    void make_probs_files(const BLOCKED_Hamiltonian& H,
                          const BLOCKED_Probs& probs,
                          const std::vector<double>& time_vec,
                          const BasisType<Basis_State>& basis,
                          std::string dir,
                          ILP_TYPE main_rank) {
        check_dir(dir, main_rank, true);

        hamiltonian_to_file("hamiltonian.csv", H, dir, main_rank);
        basis_to_file("basis.csv", basis, dir, main_rank);
        time_vec_to_file("time.csv", time_vec, dir, main_rank);
        probs_to_file("probs.csv", probs, dir, main_rank);
    }

#endif
#endif

#ifdef ENABLE_MATPLOTLIB
    namespace {
        namespace plt = matplotlibcpp;
    }

    template <>
    void matplotlib::plot<double, double>(const std::vector<double>& x,
                                          const std::vector<double>& y,
                                          std::map<std::string, std::string> keywords) {
        plt::plot(x, y, keywords);

        if (keywords.find("label") != keywords.end()) {
            plt::legend();
        }
    }

    void matplotlib::title(const std::string& name) { plt::title(name); }

    void matplotlib::xlabel(const std::string& name) { plt::xlabel(name); }

    void matplotlib::ylabel(const std::string& name) { plt::ylabel(name); }

    void matplotlib::zlabel(const std::string& name) { plt::set_zlabel(name); }

    void matplotlib::surface(const std::vector<std::vector<double>>& x,
                             const std::vector<std::vector<double>>& y,
                             const std::vector<std::vector<double>>& z,
                             std::map<std::string, std::string> keywords) {
        plt::plot_surface(x, y, z, keywords);

        if (keywords.find("label") != keywords.end()) {
            plt::legend();
        }
    }

    void matplotlib::probs_to_plot(const Probs& probs,
                                   const std::vector<double>& time_vec,
                                   const BasisType<Basis_State>& basis,
                                   std::vector<std::map<std::string, std::string>> keywords) {
        size_t index = 0;
        for (auto state : basis) {
            if (keywords.size() <= index) {
                std::map<std::string, std::string> tmp;
                keywords.emplace_back(tmp);
            }
            keywords[index]["label"] = state->to_string();
            plt::plot(time_vec, probs.row(index), keywords[index]);
            index++;
        }
    }

    void matplotlib::probs_to_plot(const Probs& probs,
                                   const std::vector<double>& time_vec,
                                   const std::vector<std::string>& basis_str,
                                   std::vector<std::map<std::string, std::string>> keywords) {
        size_t index = 0;
        for (const auto& state_str : basis_str) {
            if (keywords.size() <= index) {
                std::map<std::string, std::string> tmp;
                keywords.emplace_back(tmp);
            }
            keywords[index]["label"] = state_str;
            plt::plot(time_vec, probs.row(index), keywords[index]);
            index++;
        }
    }

#ifdef ENABLE_MPI
#ifdef ENABLE_CLUSTER

    void matplotlib::probs_to_plot(const BLOCKED_Probs& probs,
                                   const std::vector<double>& time_vec,
                                   const BasisType<Basis_State>& basis,
                                   std::vector<std::map<std::string, std::string>> keywords) {
        ILP_TYPE rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        size_t index = 0;
        for (auto state : basis) {
            if (rank == mpi::ROOT_ID) {
                if (keywords.size() <= index) {
                    std::map<std::string, std::string> tmp;
                    keywords.emplace_back(tmp);
                }
                keywords[index]["label"] = state->to_string();
            }

            std::vector<double> probs_vec(time_vec.size());
            for (size_t i = 0; i < time_vec.size(); i++) {
                probs_vec[i] = probs.get(index, i);
            }

            if (rank == mpi::ROOT_ID) plt::plot(time_vec, probs_vec, keywords[index]);
            index++;
        }
    }

    void matplotlib::probs_to_plot(const BLOCKED_Probs& probs,
                                   const std::vector<double>& time_vec,
                                   const std::vector<std::string>& basis_str,
                                   std::vector<std::map<std::string, std::string>> keywords) {
        ILP_TYPE rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        size_t index = 0;
        for (const auto& state_str : basis_str) {
            if (rank == mpi::ROOT_ID) {
                if (keywords.size() <= index) {
                    std::map<std::string, std::string> tmp;
                    keywords.emplace_back(tmp);
                }
                keywords[index]["label"] = state_str;
            }

            std::vector<double> probs_vec(time_vec.size());
            for (size_t i = 0; i < time_vec.size(); i++) {
                probs_vec[i] = probs.get(index, i);
            }

            if (rank == mpi::ROOT_ID) plt::plot(time_vec, probs_vec, keywords[index]);
            index++;
        }
    }

#endif
#endif

    void matplotlib::rho_probs_to_plot(const Probs& probs,
                                       const std::vector<double>& time_vec,
                                       const BasisType<Basis_State>& basis,
                                       std::vector<std::map<std::string, std::string>> keywords) {
        size_t from = 0;
        size_t to = 0;
        size_t basis_size = basis.size();
        for (auto state_from : basis) {
            for (auto state_to : basis) {
                size_t index = from * basis_size + to;
                if (keywords.size() <= index) {
                    while (keywords.size() <= index) {
                        std::map<std::string, std::string> tmp;
                        keywords.emplace_back(tmp);
                    }
                }

                if (from == to) {
                    keywords[index]["label"] = state_from->to_string();
                } else {
                    keywords[index]["label"] = state_from->to_string() + " -> " + state_to->to_string();
                }

                plt::plot(time_vec, probs.row(index), keywords[index]);
                to++;
            }
            from++;
            to = from;
        }
    }

    void matplotlib::rho_diag_to_plot(const Probs& probs,
                                      const std::vector<double>& time_vec,
                                      const BasisType<Basis_State>& basis,
                                      std::vector<std::map<std::string, std::string>> keywords) {
        size_t state_index = 0;
        size_t basis_size = basis.size();
        for (auto state : basis) {
            size_t index = state_index * basis_size + state_index;
            if (keywords.size() <= index) {
                while (keywords.size() <= index) {
                    std::map<std::string, std::string> tmp;
                    keywords.emplace_back(tmp);
                }
            }

            keywords[index]["label"] = state->to_string();

            plt::plot(time_vec, probs.row(index), keywords[index]);
            state_index++;
        }
    }

    void matplotlib::show(bool is_block) { plt::show(is_block); }

    void matplotlib::legend() { plt::legend(); }

    void matplotlib::make_figure(size_t x, size_t y, size_t dpi) {
        if (x == 0 or y == 0)
            plt::figure();
        else {
            plt::figure_size(x, y, dpi);
        }
    }

    void matplotlib::savefig(const std::string& filename, size_t dpi) { plt::save(filename, dpi); }

    void matplotlib::grid(bool is_enable) { plt::grid(is_enable); }

#endif

}  // namespace QComputations
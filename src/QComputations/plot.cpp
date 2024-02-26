#include "plot.hpp"
#include "functions.hpp"
#include <filesystem>
#include <fstream>

#ifdef ENABLE_MATPLOTLIB
#include "matplotlibcpp.hpp"
#endif

namespace QComputations {

#ifndef ENABLE_CLUSTER

void hamiltonian_to_file(const std::string& filename, const Hamiltonian& H, std::string dir) {
    check_dir(dir);
    H.write_to_csv_file(dir + "/" + filename);
}

void basis_to_file(const std::string& filename, const std::set<Basis_State>& basis, std::string dir) {
    check_dir(dir);

    auto filepath = dir + "/" + filename;

    std::ofstream file(filepath);

    for (size_t i = 0; i < basis.size(); i++) {
        std::string state_str = get_elem_from_set<Basis_State>(basis, i).to_string();

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

void probs_to_file(const std::string& filename, const Evolution::Probs& probs, std::string dir) {
    check_dir(dir);
    auto probs_hermit = probs.transpose();
    probs_hermit.write_to_csv_file(dir + "/" + filename);
}

#endif

#ifdef ENABLE_MPI
#ifdef ENABLE_CLUSTER

namespace fs = std::filesystem;

void check_dir(std::string& dir, bool remove_is_exist = false) {
    if (dir != "") {
        ILP_TYPE rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if (rank == mpi::ROOT_ID) {
            if (remove_is_exist and fs::is_directory(dir)) {
                fs::remove_all(dir);
            }
            fs::create_directory(dir);
        }
    } else {
        dir = ".";
    }
}

void hamiltonian_to_file(const std::string& filename, const BLOCKED_Hamiltonian& H, std::string dir) {
    check_dir(dir);
    H.write_to_csv_file(dir + "/" + filename);
}

void hamiltonian_to_file(const std::string& filename, const Hamiltonian& H, std::string dir) {
    check_dir(dir);
    H.write_to_csv_file(dir + "/" + filename);
}

void basis_to_file(const std::string& filename, const std::set<Basis_State>& basis, std::string dir) {
    check_dir(dir);

    auto filepath = dir + "/" + filename;

    std::ofstream file(filepath);

    for (size_t i = 0; i < basis.size(); i++) {
        std::string state_str = get_elem_from_set<Basis_State>(basis, i).to_string();

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

    ILP_TYPE rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_File file;

    auto filepath = dir + "/" + filename;
    MPI_File_open(MPI_COMM_WORLD, filepath.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_APPEND,
                  MPI_INFO_NULL, &file);

    auto num_length = QConfig::instance().csv_max_number_size();
    auto accuracy = QConfig::instance().csv_num_accuracy();

    for (size_t i = 0; i < time_vec.size(); i++) {
        std::string num_str = to_string_double_with_precision(time_vec[i], accuracy, num_length);

        if (i != time_vec.size() - 1) {
            num_str += ",";
        }

        MPI_File_write(file, num_str.c_str(), num_str.length(), MPI_CHAR, MPI_STATUS_IGNORE);

        if (i == time_vec.size() - 1) {
            std::string tmp = "\n";
            MPI_File_write(file, tmp.c_str(), tmp.length(), MPI_CHAR, MPI_STATUS_IGNORE);
        }
    }

    MPI_File_close(&file);
}

void probs_to_file(const std::string& filename, const Evolution::BLOCKED_Probs& probs, std::string dir) {
    check_dir(dir);
    auto probs_hermit = probs.hermit();
    probs_hermit.write_to_csv_file(dir + "/" + filename);
}

void probs_to_file(const std::string& filename, const Evolution::Probs& probs, std::string dir) {
    check_dir(dir);
    auto probs_hermit = probs.transpose();
    probs_hermit.write_to_csv_file(dir + "/" + filename);
}

void plot_from_files(const std::string& plotname,
                     std::string dir,
                     const std::string& python_script_path) {
    check_dir(dir);

    ILP_TYPE rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == mpi::ROOT_ID) {
        std::cout << "COMMAND_GIVE\n";
        std::string command = std::string("$SEABORN_PLOT") + " " + dir + " " + plotname + " " + std::to_string(QConfig::instance().fig_width()) + " " + std::to_string(QConfig::instance().fig_height());
        std::system(command.c_str());
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

void make_probs_files(const BLOCKED_Hamiltonian& H,
               const Evolution::BLOCKED_Probs& probs,
               const std::vector<double>& time_vec,
               const std::set<Basis_State>& basis,
               std::string dir) {
    check_dir(dir, true);

    hamiltonian_to_file("hamiltonian.csv", H, dir);
    basis_to_file("basis.csv", basis, dir);
    time_vec_to_file("time.csv", time_vec, dir);
    probs_to_file("probs.csv", probs, dir);
}

void make_probs_files(const Hamiltonian& H,
               const Evolution::Probs& probs,
               const std::vector<double>& time_vec,
               const std::set<Basis_State>& basis,
               std::string dir) {
    check_dir(dir, true);

    hamiltonian_to_file("hamiltonian.csv", H, dir);
    basis_to_file("basis.csv", basis, dir);
    time_vec_to_file("time.csv", time_vec, dir);
    probs_to_file("probs.csv", probs, dir);
}

void make_plot(const std::string& plotname,
               const BLOCKED_Hamiltonian& H,
               const Evolution::BLOCKED_Probs& probs,
               const std::vector<double>& time_vec,
               const std::set<Basis_State>& basis,
               std::string dir) {
    make_probs_files(H, probs, time_vec, basis, dir);
    plot_from_files(plotname, dir);

    ILP_TYPE rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == mpi::ROOT_ID) {
        fs::remove_all(dir);
    }
}

#endif
#endif

#ifdef ENABLE_MATPLOTLIB
namespace {
    namespace plt = matplotlibcpp;
}

template<>
void matplotlib::plot<double, double>(const std::vector<double>& x,
                                      const std::vector<double>& y,
                                      std::map<std::string, std::string> keywords) {
    plt::plot(x, y, keywords);

    if (keywords.find("label") != keywords.end()) {
        plt::legend();
    }
}

void matplotlib::title(const std::string& name) {
    plt::title(name);
}

void matplotlib::xlabel(const std::string& name) {
    plt::xlabel(name);
}

void matplotlib::ylabel(const std::string& name) {
    plt::ylabel(name);
}

void matplotlib::zlabel(const std::string& name) {
    plt::set_zlabel(name);
}

void matplotlib::surface(const std::vector<std::vector<double>>& x,
            const std::vector<std::vector<double>>& y,
            const std::vector<std::vector<double>>& z,
            std::map<std::string, std::string> keywords) {
    plt::plot_surface(x, y, z, keywords);

    if (keywords.find("label") != keywords.end()) {
        plt::legend();
    }
}

void matplotlib::probs_to_plot(const Evolution::Probs& probs, 
                               const std::vector<double>& time_vec,
                               const std::set<Basis_State>& basis,
                               std::vector<std::map<std::string, std::string>> keywords) {
    //std::cout << "HERE\n";
    size_t index = 0;
    for (const auto& state: basis) {
        if (keywords.size() <= index) {
            std::map<std::string, std::string> tmp;
            keywords.emplace_back(tmp);
        }
        keywords[index]["label"] = state.to_string();
        /*
        for (const auto& p: keywords[index]) {
            std::cout << p.first << " " << p.second << std::endl;
        }
        */
        plt::plot(time_vec, probs.row(index), keywords[index]);
        index++;
        //plt::plot(time_vec, state_probs);
    }
    plt::legend();
}

void matplotlib::probs_to_plot(const Evolution::Probs& probs, 
                               const std::vector<double>& time_vec,
                               const std::vector<std::string>& basis_str,
                               std::vector<std::map<std::string, std::string>> keywords) {
    //std::cout << "HERE\n";
    size_t index = 0;
    for (const auto& state_str: basis_str) {
        if (keywords.size() <= index) {
            std::map<std::string, std::string> tmp;
            keywords.emplace_back(tmp);
        }
        keywords[index]["label"] = state_str;
        /*
        for (const auto& p: keywords[index]) {
            std::cout << p.first << " " << p.second << std::endl;
        }
        */
        plt::plot(time_vec, probs.row(index), keywords[index]);
        index++;
        //plt::plot(time_vec, state_probs);
    }
    plt::legend();
}

#ifdef ENABLE_MPI
#ifdef ENABLE_CLUSTER
/*
void matplotlib::probs_in_cavity_to_plot(const Evolution::BLOCKED_Probs& probs_start,
                                const std::vector<double>& time_vec,
                                const std::set<Basis_State>& basis_start,
                                size_t cavity_id,
                                std::vector<std::map<std::string, std::string>> keywords) {
    ILP_TYPE rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Barrier(MPI_COMM_WORLD);
    auto p = Evolution::probs_to_cavity_probs(probs_start, basis_start, cavity_id);
    auto probs = p.first;
    auto basis = p.second;

    size_t index = 0;
    for (const auto& state: basis) {
        if (rank == mpi::ROOT_ID) {
            if (keywords.size() <= index) {
                std::map<std::string, std::string> tmp;
                keywords.emplace_back(tmp);
            }
            keywords[index]["label"] = state.to_string();
        }

        std::vector<double> probs_vec(time_vec.size());
        for (size_t i = 0; i < time_vec.size(); i++) {
            probs_vec[i] = probs.get(index, i);
        }

        if (rank == mpi::ROOT_ID) plt::plot(time_vec, probs_vec, keywords[index]);
        index++;
        //plt::plot(time_vec, state_probs);
    }
    if (rank == 0) plt::legend();
}
*/

void matplotlib::probs_to_plot(const Evolution::BLOCKED_Probs& probs, 
                               const std::vector<double>& time_vec,
                               const std::set<Basis_State>& basis,
                               std::vector<std::map<std::string, std::string>> keywords) {
    //std::cout << "HERE\n";
    ILP_TYPE rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    size_t index = 0;
    for (const auto& state: basis) {
        if (rank == mpi::ROOT_ID) {
            if (keywords.size() <= index) {
                std::map<std::string, std::string> tmp;
                keywords.emplace_back(tmp);
            }
            keywords[index]["label"] = state.to_string();
            /*
            for (const auto& p: keywords[index]) {
                std::cout << p.first << " " << p.second << std::endl;
            }
            */
        }

        std::vector<double> probs_vec(time_vec.size());
        for (size_t i = 0; i < time_vec.size(); i++) {
            probs_vec[i] = probs.get(index, i);
        }

        if (rank == mpi::ROOT_ID) plt::plot(time_vec, probs_vec, keywords[index]);
        index++;
        //plt::plot(time_vec, state_probs);
    }
    if (rank == 0) plt::legend();
}


void matplotlib::probs_to_plot(const Evolution::BLOCKED_Probs& probs, 
                               const std::vector<double>& time_vec,
                               const std::vector<std::string>& basis_str,
                               std::vector<std::map<std::string, std::string>> keywords) {
    //std::cout << "HERE\n";
    ILP_TYPE rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    size_t index = 0;
    for (const auto& state_str: basis_str) {
        if (rank == mpi::ROOT_ID) {
            if (keywords.size() <= index) {
                std::map<std::string, std::string> tmp;
                keywords.emplace_back(tmp);
            }
            keywords[index]["label"] = state_str;
            /*
            for (const auto& p: keywords[index]) {
                std::cout << p.first << " " << p.second << std::endl;
            }
            */
        }

        std::vector<double> probs_vec(time_vec.size());
        for (size_t i = 0; i < time_vec.size(); i++) {
            probs_vec[i] = probs.get(index, i);
        }

        if (rank == mpi::ROOT_ID) plt::plot(time_vec, probs_vec, keywords[index]);
        index++;
        //plt::plot(time_vec, state_probs);
    }
    if (rank == 0) plt::legend();
}

#endif
#endif

void matplotlib::rho_probs_to_plot(const Evolution::Probs& probs,
                       const std::vector<double>& time_vec,
                       const std::set<Basis_State>& basis,
                       std::vector<std::map<std::string, std::string>> keywords) {
    size_t from = 0;
    size_t to = 0;
    size_t basis_size = basis.size();
    for (const auto& state_from: basis) {
        for (const auto& state_to: basis) {
            size_t index = from * basis_size + to;
            if (keywords.size() <= index) {
                while(keywords.size() <= index) {
                    std::map<std::string, std::string> tmp;
                    keywords.emplace_back(tmp);
                }
            }

            if (from == to) {
                keywords[index]["label"] = state_from.to_string();
            } else {
                keywords[index]["label"] = state_from.to_string() + " -> " + state_to.to_string();
            }
            /*
            for (const auto& p: keywords[index]) {
                std::cout << p.first << " " << p.second << std::endl;
            }
            */
            plt::plot(time_vec, probs.row(index), keywords[index]);
            //plt::plot(time_vec, state_probs);
            to++;
        }
        from++;
        to = from;
    }
    plt::legend();
}

void matplotlib::rho_diag_to_plot(const Evolution::Probs& probs,
                                  const std::vector<double>& time_vec,
                                  const std::set<Basis_State>& basis,
                                  std::vector<std::map<std::string, std::string>> keywords) {
    size_t state_index = 0;
    size_t basis_size = basis.size();
    for (const auto& state: basis) {
        size_t index = state_index * basis_size + state_index;
        if (keywords.size() <= index) {
            while(keywords.size() <= index) {
                std::map<std::string, std::string> tmp;
                keywords.emplace_back(tmp);
            }
        }
        
        keywords[index]["label"] = state.to_string();
        /*
        for (const auto& p: keywords[index]) {
            std::cout << p.first << " " << p.second << std::endl;
        }
        */
        plt::plot(time_vec, probs.row(index), keywords[index]);
        //plt::plot(time_vec, state_probs);
        state_index++;
    }

    plt::legend();
}

/*
void matplotlib::rho_subdiag_to_plot(const Evolution::Probs& probs,
                                     const std::vector<double>& time_vec,
                                     const std::set<Basis_State>& basis,
                                     std::vector<std::map<std::string, std::string>> keywords) {
    size_t from = 0;
    size_t to = 0;
    size_t basis_size = basis.size();
    for (const auto& state_from: basis) {
        if (state_from.get_index() == 0) {
            from++;
            continue;
        }

        for (const auto& state_to: basis) {
            size_t index = from * basis_size + to;
            if (state_to.get_index() == 0 or from >= to or probs[index][0] == -1) {
                to++;
                continue;
            }
            if (keywords.size() <= index) {
                while(keywords.size() <= index) {
                    std::map<std::string, std::string> tmp;
                    keywords.emplace_back(tmp);
                }
            }

            keywords[index]["label"] = state_from.to_string() + " -> " + state_to.to_string();
*/
            /*
            for (const auto& p: keywords[index]) {
                std::cout << p.first << " " << p.second << std::endl;
            }
            */
/*
            plt::plot(time_vec, probs.row(index), keywords[index]);
            //plt::plot(time_vec, state_probs);
            to++;
        }
        from++;
        to = 0;
    }
    plt::legend();
}
*/

void matplotlib::show(bool is_block) {
    plt::show(is_block);
}

void matplotlib::make_figure(size_t x, size_t y, size_t dpi) {
    if (x == 0 or y == 0) plt::figure();
    else {
        plt::figure_size(x, y, dpi);    
    }
}

void matplotlib::savefig(const std::string& filename, size_t dpi) {
    plt::save(filename, dpi);
}

void matplotlib::grid(bool is_enable) {
    plt::grid(is_enable);
}

#endif

} // namespace QComputations
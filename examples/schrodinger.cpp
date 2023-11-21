#include "/home/kuzmaline/Quantum/diploma/src/QComputations_CPU_CLUSTER.hpp"
#include <iostream>
#include <regex>
#include <complex>

using COMPLEX = std::complex<double>;


// not ready
std::string make_state_regex_pattern(const std::string& format, const std::string& del, bool is_sequence) {
    using namespace QComputations;
    std::string res;
    std::regex format_regex("(\\$\\!)");

    auto regex_begin = std::sregex_iterator(format.begin(), format.end(), format_regex);
    auto regex_end = std::sregex_iterator();

    std::string delim = del;
    if (del == ",") {
        delim = "\\,";
    }

    assert(regex_begin != regex_end); // В формате должен быть разделитель - "$!"

    std::smatch match = *regex_begin;
    std::string first_part = match.prefix();
    std::string second_part = match.suffix();

    format_regex = std::regex("(\\$[N,W])");

    regex_begin = std::sregex_iterator(first_part.begin(), first_part.end(), format_regex);
    regex_end = std::sregex_iterator();

    std::string left_part;
    std::sregex_iterator word = regex_begin;
    match = *word;

    if (match.str() == "$N") {
        for (const auto c: std::string(match.prefix())) {
            if (c == '|' or c == '_') left_part += "\\";
            left_part += c;
        }

        left_part += "(\\d";
        ++word;
        if (word != regex_end) {
            match = *word;
            left_part += "([";
            for (const auto c: std::string(match.prefix())) {
                if (c == '|' or c == '_') left_part += "\\";
                left_part += c;
            }
            left_part += "][\\d]{2})?";
            left_part += match.suffix();
        } else {
            left_part += match.suffix();
        }
    } else if (match.str() == "$W") {
        left_part += "[";
        left_part += match.prefix();
        left_part += "\\d\\d]?";
        ++word;
        if (word != regex_end) {
            match = *word;
            left_part += match.prefix();
            left_part += "\\d+";
            left_part += match.suffix();
        } else {
            left_part += match.suffix();
        }
    } else if (match.str() == "$M") {
        left_part += match.prefix();
        left_part += "[0-9]+";
    } else {
        std::cerr << "FORMAT ERROR ON FIRST PART!" << std::endl;
    }

    format_regex = std::regex("(\\$M)");

    regex_begin = std::sregex_iterator(second_part.begin(), second_part.end(), format_regex);
    std::string right_part;

    match = *regex_begin;

    if (match.str() == "$N") {
        right_part += match.prefix();
        right_part += "\\d+";
        ++word;
        if (word != regex_end) {
            match = *word;
            right_part += "[";
            right_part += match.prefix();
            right_part += "\\d\\d]?";
            right_part += match.suffix();
        } else {
            right_part += match.suffix();
        }
    } else if (match.str() == "$W") {
        right_part += "[";
        right_part += match.prefix();
        right_part += "\\d\\d]?";
        ++word;
        if (word != regex_end) {
            match = *word;
            right_part += match.prefix();
            right_part += "\\d+";
            right_part += match.suffix();
        } else {
            right_part += match.suffix();
        }
    } else if (match.str() == "$M") {
        right_part += "[";
        right_part += match.prefix();
        right_part += "]?";
        if (!is_sequence) {
            right_part += "(";
        }
        right_part += "[0-9]*";

        if (!is_sequence) {
            right_part += "[" + delim + "]?";
            right_part += ")*";
        }

        right_part += match.suffix();
    } else {
        std::cerr << "FORMAT ERROR ON SECOND PART!" << std::endl;
    }

    if (is_sequence) {
        res = "((" + left_part + "[" + delim + "]?)+" + right_part + ")+)";
    } else {
        res = "(" + left_part + "[" + delim + "]?)+" + right_part + ")"; 
    }

    return res;
}

int main(int argc, char** argv) {
    using namespace QComputations;
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    QConfig::instance().set_width(30);

    size_t grid_size = 1;
    size_t atoms_num = 3;
    std::vector<size_t> grid_config;

    for (size_t i = 0; i < grid_size; i++) {
        if (i == 0 or i == grid_size - 1) {
            grid_config.emplace_back(atoms_num);
        } else {
            grid_config.emplace_back(0);
        }
    }

    State grid(grid_config);
    //grid.set_qubit(0, 0, 1);
    //grid.set_qubit(0, 1, 1);
    grid.set_n(1);

    int ctxt;
    mpi::init_grid(ctxt);
    BLOCKED_H_TCH H(ctxt, grid);

    if (rank == 0) { show_basis(H.get_basis()); }

    H.show();

    std::vector<COMPLEX> init_state(H.size(), 0);
    init_state[grid.get_index(H.get_basis())] = COMPLEX(1, 0);

    auto time_vec = linspace(0, 1000, 2000);

    QConfig::instance().set_state_format("|$N$!;$M>");
    std::cout << "0\u2082\u2083\u29FD" << std::endl;
    std::cout << QConfig::instance().state_format() << std::endl;
    std::string pattern = make_state_regex_pattern(QConfig::instance().state_format(), ",", false);
    std::cout << pattern << std::endl;
    std::regex reg(pattern);
    std::regex test_reg("\\|");

    std::string expr = "1+|0,12,13;001,012>+|0>+|0_01,1_12;112>";
    //std::string expr = "1+|0;001>|12_01,12_02;112>+|12>|12;001> + |12>";
    std::cout << expr << std::endl;
    auto regex_begin = std::sregex_iterator(expr.begin(), expr.end(), reg);
    auto regex_end = std::sregex_iterator();

    for (auto i = regex_begin; i != regex_end; i++) {
        std::smatch match = *i;
        std::cout << match.prefix() << " " << match.str() << std::endl;
    }

    MPI_Finalize();
    return 0;
    auto probs = Evolution::schrodinger(init_state, H, time_vec);
    //auto probs = Evolution::quantum_master_equation(init_state, H, time_vec);

    if (rank == 0) {
        matplotlib::make_figure(1920, 1080);
    }

    matplotlib::probs_to_plot(probs, time_vec, H.get_basis());
    if (rank == 0) {
        matplotlib::grid();
        matplotlib::show();
    }

    MPI_Finalize();
    return 0;
}

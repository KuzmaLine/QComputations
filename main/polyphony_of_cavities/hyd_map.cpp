#include <iostream>
#include <complex>
#include "QComputations_CPU_CLUSTER_NO_PLOTS.hpp"

double find_amplitude(double amplitude, size_t path_length, size_t count_paths, double alpha = 1) {
    return std::pow(amplitude / count_paths, double(1) / (alpha * path_length));
}

std::string make_filename(const std::vector<size_t>& shapes) {
    return std::string(std::string("map_") + std::to_string(shapes[0]) + "_" + std::to_string(shapes[1]) + "_" + std::to_string(shapes[2]));
}


namespace QComputations {

class Hydrogen_System: public Basis_State {
    public:
        explicit Hydrogen_System(size_t cavities_with_hydrogen, const std::vector<size_t>& grid_config): Basis_State(cavities_with_hydrogen * 3 + (grid_config.size() > 1 ? (grid_config.size() - 2) : 0)),
                                                                                                              grid_map(C_STYLE, grid_config.size(), grid_config.size(), std::pair<double, double>(0, 0)) {
            //std::cout << this->to_string() << ": " << this->get_max_val(0, 0) << " " << this->get_max_val(1, 0) << " " << this->get_max_val(2, 0) << std::endl;
            std::vector<size_t> groups;
            x_ = grid_config.size();
            y_ = 1;
            z_ = 1;
            long qudit_id = -1;
            for (size_t i = 0; i < grid_config.size(); ++i) {
                if (grid_config[i] != 0) {
                    qudit_id += 1;
                }

                qudit_id += grid_config[i] + 1;
                groups.emplace_back(qudit_id);
            }

            groups_ = groups;
        }

        COMPLEX get_g_bond() const { return g_bond_; }
        COMPLEX get_g_dist() const { return g_dist_; }

        COMPLEX get_prot_trans_gamma(size_t i, size_t j) const {
            int coef = -1;

            if (i > j) {
                coef = 1;
                auto tmp = i;
                i = j;
                j = tmp;
            }

            return grid_map[i][j].first * std::exp(std::complex<double>(0, coef) * grid_map[i][j].second / QConfig::instance().h());
        }
        
        void reshape(size_t x, size_t y, size_t z) {
            x_ = x; y_ = y; z_ = z;
        }

        void set_prot_trans_gamma(size_t i, size_t j, double amplitude, double length) {
            grid_map[i][j] = std::make_pair(amplitude, length);
            grid_map[j][i] = std::make_pair(amplitude, length);
        }

        void set_waveguide(double amplitude, double length) {
            auto neighbours = update_neighbours(x_, y_, z_);

            for (size_t from_id = 0; from_id < groups_.size(); from_id++) {
                auto cur_neighbours = neighbours[from_id];

                for (const auto to_id: cur_neighbours) {
                    grid_map[from_id][to_id] = std::make_pair(amplitude, length);
                }
            }
        }

        void set_waveguide(size_t from, size_t to, double amplitude, double length) {
            if (from > to) {
                auto tmp = from;
                from = to;
                to = tmp;
            }

            grid_map[from][to] = std::make_pair(amplitude, length);
        }
    private:
        size_t x_, y_, z_;
        Matrix<std::pair<double, double>> grid_map;
        COMPLEX g_bond_ = 0.2;
        COMPLEX g_dist_ = 0.01;
};

using OpType = Operator<Hydrogen_System>;

State<Hydrogen_System> H_dist(const Hydrogen_System& state) {
    State<Hydrogen_System> res;

    for (size_t i = 0; i < state.get_groups_count(); i++) {
        if (state.get_group_size(i) > 1 and state.get_qudit(1, i) == 1 and state.get_qudit(0, i) == 1) {
            res += sigma_x(state, 2, i) * state.get_g_dist();
        }
    }

    return res;
}

State<Hydrogen_System> H_bond(const Hydrogen_System& state) {
    State<Hydrogen_System> res;

    for (size_t i = 0; i < state.get_groups_count(); i++) {
        if (state.get_group_size(i) > 1 and state.get_qudit(2, i) == 0) {
            res += sigma_x(state, 1, i) * state.get_g_bond();
        }
    }

    return res;
}

State<Hydrogen_System> bond_energy(const Hydrogen_System& state) {
    State<Hydrogen_System> res;

    for (size_t i = 0; i < state.get_groups_count(); i++) {
        if (state.get_group_size(i) > 1 and state.get_qudit(1, i) == 1) {
            res += get_qudit(state, 1, i) * QConfig::instance().h() * QConfig::instance().w();
        }
    }

    return res;
}

State<Hydrogen_System> prot_transfer(const Hydrogen_System& state) {
    State<Hydrogen_System> res;

    auto res_state = state;
    for (size_t i = 0; i < state.get_groups_count(); i++) {
        if (state.get_qudit(0, i) == 1) {
            if (state.get_group_size(i) > 1 and state.get_qudit(2, i) == 0) {
                continue;
            }

            res_state.set_qudit(0, 0, i);

            for (size_t j = 0; j < state.get_groups_count(); j++) {
                if (j != i) {
                    res += set_qudit(res_state, 1, 0, j) * state.get_prot_trans_gamma(i, j);
                }
            }

            res_state.set_qudit(1, 0, i);
        }
    }

    return res;
}

State<Hydrogen_System> A_in(const Hydrogen_System& state) {
    return set_qudit(state, state.get_qudit(1) + 1, 1);
}

State<Hydrogen_System> A_out(const Hydrogen_System& state) {
    return set_qudit(state, state.get_qudit(1) - 1, 1);
}

void set_init_hyd_state(Hydrogen_System& state, size_t target_cavity = 0) {
    state.set_qudit(1, 0, 0);
    state.set_qudit(1, 1, 0);
    state.set_qudit(1, 2, 0);

    state.set_qudit(1, 2, target_cavity);
    state.set_qudit(1, 1, target_cavity);
}

} // namespace QComputations

int main(int argc, char** argv) {
    using namespace QComputations;
    int world_size, rank;
    double amplitude = 0.001;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int ctxt;
    mpi::init_grid(ctxt);

    std::vector<size_t> grid_config = {1};
    Hydrogen_System state(1, grid_config);
    set_init_hyd_state(state);

    OpType my_H;
    my_H = my_H + OpType(H_dist) + OpType(H_bond) + OpType(prot_transfer) + OpType(bond_energy);

    auto time_vec = linspace(0, 8000, 8000);

    H_by_Operator H(State(state), my_H);

    //auto probs = Evolution::quantum_master_equation(State(state).fit_to_basis(H.get_basis()), H, time_vec);
    auto probs = Evolution::schrodinger(State(state).fit_to_basis(H.get_basis()), H, time_vec);

    if (rank == 0) {
        std::cout << "calculated\n";
        make_probs_files(H, probs, time_vec, H.get_basis(), "hyd_map/original_g_bond=" + std::to_string(state.get_g_bond().real()), rank);
    }

// ---------------------- graphs ----------------------------

    std::vector<std::vector<size_t>> grid_configs = {{1, 1},
                                                     {1, 0, 1},
                                                     {1, 0, 0, 1},
                                                     {1, 0, 0, 1},
                                                     {1, 0, 0,
                                                      0, 1, 0,
                                                      0, 0, 0},
                                                      {1, 0, 0, 0, 0, 0, 0, 0, 0,
                                                      0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                      0, 0, 0, 0, 0, 0, 0, 0, 1},
                                                      {1, 0, 0, 0, 0, 0, 0, 0, 0,
                                                      0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                      0, 0, 0, 0, 0, 0, 0, 0, 1}};
                                                      

    std::vector<std::vector<size_t>> shapes = {{2, 1, 1},
                                               {3, 1, 1},
                                               {2, 2, 1},
                                               {4, 1, 1},
                                               {3, 3, 1},
                                               {3, 3, 3},
                                               {3, 3, 3}};

    std::vector<size_t> target_cavity = {1, 2, 3, 3, 4, 26, 26};

    double alpha = 1;
    std::vector<double> amplitudes = {amplitude,
                                      0.01,
                                      0.01 / std::sqrt(2),
                                      0.02,
                                      0.01 / std::sqrt(2),
                                      0.01,
                                      0.001};

    //std::vector<double> amplitudes(5, amplitude);

    for (size_t i = 0; i < shapes.size(); ++i) {
        auto time_vec = linspace(0, 16000, 16000);
        auto new_amplitude = amplitudes[i];
        Hydrogen_System new_state(2, grid_configs[i]);

        //new_state.set_n(atoms_count, 0);
        set_init_hyd_state(new_state, target_cavity[i]);
        //new_state.set_atom(1, 0, 0);

        new_state.reshape(shapes[i][0], shapes[i][1], shapes[i][2]);
        new_state.set_waveguide(new_amplitude, 0);

        std::string second;
        if (i == shapes.size() - 1) {
            new_state.set_waveguide(0, 26, amplitude, 0);
            second = "_second";
        }

        BLOCKED_H_by_Operator H(ctxt, State(new_state), my_H);

        if (rank == 0) show_basis(H.get_basis());

        H.show();

        /*
        for (size_t i = 0; i < new_state.get_groups_count(); i++) {
            for (size_t j = 0; j < new_state.get_groups_count(); j++) {
                if (i == j) std::cout << 0 << " ";
                else {
                    std::cout << std::setw(10) << new_state.get_gamma(i, j).real() << " ";
                }
            }
            std::cout << std::endl;
        }
        */

        //H.show();
        std::cout << new_amplitude << std::endl;

        auto probs = Evolution::schrodinger(State(new_state).fit_to_basis(H.get_basis()), H, time_vec);

        make_probs_files(H, probs, time_vec, H.get_basis(), "hyd_map/" + make_filename(shapes[i]) + second, 0);

        auto p_0 = Evolution::probs_to_cavity_probs(probs, H.get_basis(), 0);
        auto p_1 = Evolution::probs_to_cavity_probs(probs, H.get_basis(), target_cavity[i]);

        make_probs_files(H, p_0.first, time_vec, p_0.second, "hyd_map/0_" + make_filename(shapes[i]) + second, 0);
        make_probs_files(H, p_1.first, time_vec, p_1.second, "hyd_map/" + std::to_string(target_cavity[i]) + "_" + make_filename(shapes[i]) + second, 0);
    }

    MPI_Finalize();
    return 0;
}

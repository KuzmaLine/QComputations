#include <iostream>
#include <complex>
#include "QComputations_CPU_CLUSTER_NO_PLOTS.hpp"

namespace QComputations {

class Hydrogen_System: public Basis_State {
    public:
        explicit Hydrogen_System(size_t cavities_with_hydrogen, const std::vector<size_t>& grid_config): Basis_State(cavities_with_hydrogen * 3 + (grid_config.size() > 1 ? (grid_config.size() - 2) : 0)),
                                                                                                              grid_map(C_STYLE, grid_config.size(), grid_config.size(), std::pair<double, double>(0, 0)) {
            //std::cout << this->to_string() << ": " << this->get_max_val(0, 0) << " " << this->get_max_val(1, 0) << " " << this->get_max_val(2, 0) << std::endl;
            std::vector<size_t> groups;
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
            }

            return grid_map[i][j].first * std::exp(std::complex<double>(0, coef) * grid_map[i][j].second / QConfig::instance().h());
        }

        void set_prot_trans_gamma(size_t i, size_t j, double amplitude, double length) {
            grid_map[i][j] = std::make_pair(amplitude, length);
            grid_map[j][i] = std::make_pair(amplitude, length);
        }
    private:
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


} // namespace QComputations

int main(int argc, char** argv) {
    using namespace QComputations;
    int world_size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::vector<size_t> grid_config = {1};
    Hydrogen_System one_state(1, grid_config);
    one_state.set_qudit(1, 0, 0);
    one_state.set_qudit(1, 1, 0);
    one_state.set_qudit(1, 2, 0);

    grid_config = {1, 1};
    Hydrogen_System state(2, grid_config);
    state.set_qudit(1, 0, 0);
    state.set_qudit(1, 1, 0);
    state.set_qudit(1, 2, 0);

    state.set_qudit(1, 2, 1);
    state.set_qudit(1, 1, 1);

    //std::cout << state.to_string() << std::endl;

    OpType my_H;
    my_H = my_H + OpType(H_dist) + OpType(H_bond) + OpType(prot_transfer) + OpType(bond_energy);

    /*
    OpType my_A_in(A_in);
    OpType my_A_out(A_out);

    std::vector<std::pair<double, OpType>> dec;
    dec.emplace_back(std::make_pair(0.01, my_A_in));
    dec.emplace_back(std::make_pair(0.01, my_A_out));

    auto res = my_A_in.run(state);
    std::cout << res.to_string() << std::endl;
    */
    auto time_vec = linspace(0, 8000, 8000);
    size_t steps_count = 900;
    double a = 0.001, b = 0.03;
    auto amplitude_range = linspace(a, b, steps_count);

    H_by_Operator H(State(one_state), my_H);

    auto probs = Evolution::quantum_master_equation(State(one_state).fit_to_basis(H.get_basis()), H, time_vec);

    if (rank == 0) {
        std::cout << "calculated\n";
        make_probs_files(H, probs, time_vec, H.get_basis(), "hydrogen_waveguide_amplitude_2/original_g_bond=" + std::to_string(state.get_g_bond().real()), rank);
    }

    size_t start, count;
    make_rank_map(amplitude_range.size(), rank, world_size, start, count);


    for (size_t i = start; i < count + start; i++) {
        auto amplitude = amplitude_range[i];
        state.set_prot_trans_gamma(0, 1, amplitude, 1);
        H = H_by_Operator(State(state), my_H);

        //if (rank == 0) show_basis(H.get_basis());

        //H.show();

        probs = Evolution::quantum_master_equation(State(state).fit_to_basis(H.get_basis()), H, time_vec);

        //basis_to_file("basis_check.csv", H.get_basis());    
        make_probs_files(H, probs, time_vec, H.get_basis(), "hydrogen_waveguide_amplitude_2/hyd_g_bond=" + std::to_string(state.get_g_bond().real()) + "_amplitude_" + std::to_string(amplitude), rank);

        auto p_0 = Evolution::probs_to_cavity_probs(probs, H.get_basis(), 0);
        auto p_1 = Evolution::probs_to_cavity_probs(probs, H.get_basis(), 1);

        make_probs_files(H, p_0.first, time_vec, p_0.second, "hydrogen_waveguide_amplitude_2/0_hyd_g_bond=" + std::to_string(state.get_g_bond().real()) + "_amplitude_" + std::to_string(amplitude), rank);
        make_probs_files(H, p_1.first, time_vec, p_1.second, "hydrogen_waveguide_amplitude_2/1_hyd_g_bond=" + std::to_string(state.get_g_bond().real()) + "_amplitude_" + std::to_string(amplitude), rank);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}


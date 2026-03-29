#include "QComputations_SINGLE.hpp"

namespace QComputations {

constexpr int HYDROGEN_SIZE_STATE = 1;
constexpr int OXYGEN_VACANT_COUNT = 2;

//                oxygen = 2
//hyd_postions    vacantposions  covbonds
//|013>          |1;1>|0;1>      |0>|1>|0>
//|001>

class WaterState : public Basis_State {
    public:
        explicit WaterState(int oxygen_count, int hydrogen_count): 
                            Basis_State(OXYGEN_VACANT_COUNT * oxygen_count + hydrogen_count
                            + hydrogen_count * HYDROGEN_SIZE_STATE,
                            OXYGEN_VACANT_COUNT * oxygen_count),
                            oxygen_count_(oxygen_count), hydrogen_count_(hydrogen_count),
                            g_(C_STYLE, OXYGEN_VACANT_COUNT * oxygen_count, OXYGEN_VACANT_COUNT * oxygen_count, COMPLEX(0.01, 0)),
                            g_cov_(oxygen_count * OXYGEN_VACANT_COUNT, 1),
                            g_dist_(oxygen_count * OXYGEN_VACANT_COUNT, 1) {
            std::vector<size_t> groups = {size_t(hydrogen_count)};
            for (size_t i = 0; i < oxygen_count; i++) {
                groups.emplace_back(OXYGEN_VACANT_COUNT);
            }
            for (size_t i = 0; i < hydrogen_count; i++) {
                groups.emplace_back(HYDROGEN_SIZE_STATE);
            }

            this->set_groups(groups);
        }

        COMPLEX g(size_t i, size_t j) const {
            if (i > j) {
                auto tmp = i;
                i = j;
                j = tmp;
            }

            return g_[i][j];
        }

        void set_g(size_t i, size_t j, COMPLEX val) {
            if (i > j) {
                auto tmp = i;
                i = j;
                j = tmp;
            }

            g_[i][j] = val;
        }

        
        COMPLEX g_cov(size_t i) const {
            return g_cov_[i];
        }

        void set_g_cov(const std::vector<COMPLEX>& g_cov) {
            g_cov_ = g_cov;
        }

        COMPLEX g_dist(size_t i) const {
            return g_dist_[i];
        }

        void set_g_dist(const std::vector<COMPLEX>& g_dist) {
            g_dist_ = g_dist;
        }

        int hydrogen_count() const { return hydrogen_count_; }
        int positions_count() const { return OXYGEN_VACANT_COUNT * oxygen_count_; }
        int oxygen_count() const { return oxygen_count_; }

        bool is_free_position(size_t i) const {
            return this->get_qudit(i, 1) == 0;
        }

        int get_hyd_position(size_t i) const {
            return this->get_qudit(i, 0);
        }

        std::string to_string() const override {
            std::string res;
            for (size_t i = 0; i < this->get_group_size(0); i++) {
                res += ";" + std::to_string(this->get_qudit(i, 0) / 2);
            }

            res[0] = '|';
            res += ">";

            for (size_t i = 1; i < this->get_groups_count(); i++) {
                res += this->group_to_string(i);
            }

            return res;
        }

    private:
        int oxygen_count_;
        int hydrogen_count_;
        Matrix<COMPLEX> g_;
        std::vector<COMPLEX> g_cov_;
        //deprecated
        std::vector<COMPLEX> g_dist_;
};

enum DIST{FAR = 0, CLOSE};

State<WaterState> MoveHydrogenFunc(const WaterState& state) {
    State<WaterState> res;
    auto tmp = state;

    for (size_t i = 0; i < state.hydrogen_count(); i++) {
        auto cur_position = state.get_hyd_position(i);
        for (int j = 0; j < state.positions_count(); j++) {
            if (j != cur_position && state.is_free_position(j)) {
                tmp.set_qudit(j, i, 0); // Меняем позицию водорода
                tmp.set_qudit(0, cur_position % 2, 1 + cur_position / 2); // Освобождаем позицию
                res += //check(tmp, int(FAR), 0, 1 + state.oxygen_count() + i) * 
                       set_qudit(tmp, 1, j % 2, 1 + j / 2) * state.g(i, j); // Вернуть суперпозицию с установленной занятой позицией

                tmp = state;
            }
        }
    }

    return res;
}


/*
State<WaterState> DistHydOxygenFunc(const WaterState& state) {
    State<WaterState> res;
    auto tmp = state;

    for (size_t i = 0; i < state.hydrogen_count(); i++) {
        res += sigma_x(state, 0, 1 + state.oxygen_count() + i) * state.g_dist(state.get_hyd_position(i));
    }

    return res;
}
*/

State<WaterState> CovHydOxygenFunc(const WaterState& state) {
    State<WaterState> res;

    for (size_t i = 0; i < state.hydrogen_count(); i++) {
        res += sigma_x(state, 0, 1 + state.oxygen_count() + i) * state.g_cov(state.get_hyd_position(i));
    }

    return res;
}

State<WaterState> Energy(const WaterState& state) {
    return State<WaterState>(state) * state.hydrogen_count();
}

//return State<WaterState>(tmp.set_qudit(state.oxygen_count() * 2, i, 0));

}

int main(void) {
    using namespace QComputations;

    std::vector<COMPLEX> g_cov = {0.05, 0.1};
    WaterState state(1, 1);
    state.set_group(0, "|0>"); // Устанавливаем значения водорода
    state.set_group(1, "|1;0>"); // Устанавливаем значения для вакантных позиций
    state.set_g_cov(g_cov);
    std::cout << state.to_string() << std::endl;

    using OpType = Operator<WaterState>;

    auto MoveHydrogenOperator = OpType(MoveHydrogenFunc);
    auto res = MoveHydrogenOperator.run(state);

    //std::cout << res.to_string() << std::endl;

    OpType H_op = OpType(Energy) + MoveHydrogenOperator + OpType(CovHydOxygenFunc);

    H_by_Operator<WaterState> H(state, H_op);

    show_basis(H.get_basis());
    H.show();

    auto time_vec = linspace(0, 100, 101);
    auto probs = quantum_master_equation(state, H, time_vec);

    matplotlib::make_figure(1200, 1000);
    matplotlib::probs_to_plot(probs, time_vec, H.get_basis());
    matplotlib::legend();
    matplotlib::grid();
    matplotlib::show();

    return 0;
}
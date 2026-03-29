#include "QComputations_SINGLE.hpp"

#include <iostream>

namespace {
    constexpr double SAMPLES_COUNT = 800;
    constexpr double a = 0.1;
    constexpr double M = 1;
    //constexpr double V = 49/2 * (1/(M*a*a));
    constexpr double V = 210;
    constexpr double P0 = 30;
    constexpr double D = 0.5;
    constexpr double start_pos = -2;
}

namespace QComputations {
class X_State: public Basis_State {
    public:
        X_State(double start, double end, int samples_count): start_(start),
                                                              end_(end),
                                                              dx_((end-start) / samples_count),
                                                              Basis_State(1, samples_count - 1) {}
        double coord() const {
            return start_ + dx_ / 2 + dx_*this->get_qudit(0);
        }

        X_State left_step(int steps_count = 1) const {
            auto res = *this;
            if (this->get_qudit(0) >= steps_count) {
                res.set_qudit(this->get_qudit(0) - steps_count, 0);
            }

            return res;
        }

        X_State right_step(int steps_count = 1) const {
            auto res = *this;
            if (this->get_qudit(0) <= this->get_max_val(0) - steps_count) {
                res.set_qudit(this->get_qudit(0) + steps_count, 0);
            }

            return res;
        }

        bool operator<(const Basis_State& other) const override {
            return this->get_qudit(0) < other.get_qudit(0);
        }

        double dx() const { return dx_; }

    private:
        double start_;
        double end_;
        double dx_;
};

State<X_State> P2M(const X_State& state) {
    auto dx = state.dx();
    State<X_State> res(state, 1/(M*dx*dx));
    res.insert(state.left_step(), -1/(2*M*dx*dx));
    res.insert(state.right_step(), -1/(2*M*dx*dx));

    return res;
}

State<X_State> V_x(const X_State& st) {
    return (std::abs(st.coord()) <= (a / 2) ? State<X_State>(st, V) : State<X_State>(st, 0));
}

COMPLEX ksi(const X_State& st) {
    double x = st.coord();
    return COMPLEX(1, 0) / (std::pow(M_PI * D * D, 1/4)) *
           std::exp(COMPLEX(0, P0*x)) * 
           std::exp(-(x - start_pos) * (x - start_pos) / (2 * D * D));  
}

}
 
int main(void) {
    using namespace QComputations;
    using OpType = Operator<X_State>;

    int start_coord = -3;

    X_State st(-6, 6, SAMPLES_COUNT);
    //st.set_qudit(SAMPLES_COUNT / 2, 0);

    std::cout << st.to_string() << std::endl;

    OpType H_op = OpType(V_x) + OpType(P2M);

    auto basis = State_Graph<X_State>(st, H_op).get_basis();

    show_basis(basis);

    H_by_Operator<X_State> H(st, H_op);

    //H.show();

    auto time_vec = linspace(double(0), 0.8, 400);
    auto coord_vec = linspace(start_coord, -start_coord, SAMPLES_COUNT / 2);
    auto dt = (time_vec[time_vec.size() - 1] - time_vec[0]) / (time_vec.size() - 1);

    Matrix<double> plots(C_STYLE, 2, coord_vec.size());

    auto state = State<X_State>(st, basis);

    state.init_state_by_func(ksi);
    state.normalize();

    size_t start_index = 0;

    for (auto b_state : basis) {
        if (b_state->coord() >= start_coord) {
            break;
        }

        start_index++;
    }

    auto probs = state.to_probs();
    for (size_t i = 0; i < coord_vec.size(); i++) {
        plots[0][i] = probs[i + start_index];
        plots[1][i] = (std::abs(coord_vec[i]) <= (a / 2) ? 1 : 0);
    }

    make_plot_files(plots, coord_vec, {"State", "V"}, "Quantum_Tunnel_Res/t=" + std::to_string(time_vec[0]));

    for (size_t i = 1; i < time_vec.size(); i++) {
        state = schrodinger_step(state, H, dt, basis);
        probs = state.to_probs();

        double res = 0;
        for (size_t j = 0; j < coord_vec.size(); j++) {
            res += probs[j + start_index];   
            plots[0][j] = probs[j + start_index];
        }

        make_plot_files(plots, coord_vec, {"State", "V"}, "Quantum_Tunnel_Res/t=" + std::to_string(time_vec[i]));
    }

    return 0;
}
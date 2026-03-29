#include <utility>
#include <filesystem>
#include <fstream>

#include "QComputations_SINGLE_NO_PLOTS.hpp"

using namespace QComputations;

const COMPLEX g_dist = 0.02;
const COMPLEX omega_dist = 0.3;
const COMPLEX g_prot = 0.05;
const COMPLEX omega_prot = 0.25;
const COMPLEX omega_phn = 0.1;

const double kB = 8.617e-5;
const double hbar = 6.582119569e-16;
const vector<double> Tk = {10, 95, 200, 273.15, 300, 373.15, 400, 600, 800, 1000, 2000, 5000, 10000};

const double gamma_out = 0.02;
const double gamma_deph = 0.01;

class WaterBondState : public Basis_State {
    public:
        explicit WaterBondState(int dist, int prot, int phn)
            : Basis_State(vector<int>{dist, prot, phn}, 
                1, vector<size_t>{1, 1, 1}) {}
        
        int dist_value() const {return this->get_qudit(0, 0);}
        int prot_value() const {return this->get_qudit(0, 1);}
        int phn_value() const {return this->get_qudit(0, 2);}

        void set_dist(int value) {this->set_qudit(value, 0, 0);}
        void set_prot(int value) {this->set_qudit(value, 0, 1);}
        void set_phn(int value) {this->set_qudit(value, 0, 2);}

        string to_string() const override {
            return "|" + std::to_string(dist_value()) + "," +
                            std::to_string(prot_value()) + "," +
                            std::to_string(phn_value()) + ">";
        };
};

using Op = Operator<WaterBondState>;

Op H_dist() {
    return Op([=](const WaterBondState state) {
        State<WaterBondState> result;

        if (state.dist_value() == 0 && state.prot_value() == 1 && 
            state.phn_value() == 0) {
            WaterBondState new_state = state; 
            new_state.set_dist(1);
            result += State<WaterBondState>(new_state) * g_dist;
        }

        if (state.dist_value() == 1 && state.prot_value() == 1 &&
            state.phn_value() == 0) {
            WaterBondState new_state = state;
            new_state.set_dist(0);
            result += State<WaterBondState>(new_state) * conj(g_dist);

            result += State<WaterBondState>(state) * hbar * omega_dist;
        }

        return result;
    });
}

Op H_prot() {
    return Op([=](const WaterBondState& state) {
        State<WaterBondState> result;

        if (state.dist_value() == 0 && state.prot_value() == 0 &&
            state.phn_value() == 1) {
            WaterBondState new_state = state;
            new_state.set_prot(1);
            new_state.set_phn(0);
            result += State<WaterBondState>(new_state) * g_prot;
        }

        if (state.dist_value() == 0 && state.prot_value() == 1 &&
            state.phn_value() == 0) {
            WaterBondState new_state = state;
            new_state.set_prot(0);
            new_state.set_phn(1);
            result += State<WaterBondState>(new_state) * g_prot;
        
            result += State<WaterBondState>(state) * hbar * omega_prot;
        }

        return result;
    });
}

Op H_phn() {
    return Op([=](const WaterBondState& state) {
        State<WaterBondState> result;

        if (state.dist_value() == 0 && state.prot_value() == 0 &&
            state.phn_value() == 1) {
            result += State<WaterBondState>(state) * omega_phn;
        }

        return result;
    });
}

vector<pair<double, Op>> make_lindblad_channels(double temp_K) {
    double temp_eV = kB * temp_K;
    double mu = exp(-omega_phn.real() / temp_eV);
    double gamma_in = gamma_out * mu;

    std::function<State<WaterBondState>(const WaterBondState&)> L_out =
        [](const WaterBondState& state) {
            State<WaterBondState> result;
            if (state.phn_value() == 1) {
                WaterBondState new_state = state;
                new_state.set_phn(0);
                result += State<WaterBondState>(new_state);
            }
            return result;
        };

    std::function<State<WaterBondState>(const WaterBondState&)> L_in =
        [](const WaterBondState& state) {
            State<WaterBondState> result;
            if (state.phn_value() == 0) {
                WaterBondState new_state = state;
                new_state.set_phn(1);
                result += State<WaterBondState>(new_state);
            }
            return result;
        };

    std::function<State<WaterBondState>(const WaterBondState&)> L_deph =
        [](const WaterBondState& state) {
            State<WaterBondState> result;
            result += State<WaterBondState>(state) * state.prot_value();
            return result;
        };

    return {
        {gamma_out, Op(L_out)},
        {gamma_in,  Op(L_in)},
        {gamma_deph, Op(L_deph)}
    };
}

int main() {
    WaterBondState psi0(0, 1, 0);

    Op H_total = H_dist() + H_prot() + H_phn();

    vector<double> t_vec;
    const double t = 1400.0;
    const int steps = 1600;

    for (int i = 0; i <= steps; i++) {
        t_vec.emplace_back(t * i / steps);
    }

    for (double temp : Tk) {
        std::cout << "temperature: " << temp << "\n";

        auto Ls = make_lindblad_channels(temp);
        H_by_Operator<WaterBondState> H(psi0, H_total, Ls);

        auto probs = quantum_master_equation(psi0, H, t_vec);
        
        string filename = "water_cluster_res/T_" + std::to_string(temp);

        make_probs_files(filename, probs, H.get_basis(), t_vec);
    }

    return 0;
}
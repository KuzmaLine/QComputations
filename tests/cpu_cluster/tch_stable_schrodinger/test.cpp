#include <signal.h>
#include <stdio.h>
#include <unistd.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "QComputations_CPU_CLUSTER_NO_PLOTS.hpp"

using COMPLEX = std::complex<double>;

namespace QComputations {

    class My_State : public Basis_State {
       public:
        My_State(size_t N, COMPLEX a, COMPLEX b, double wa, double wc)
            : Basis_State(N + 2, 1, 2), a_(a), b_(b), wa_(wa), wc_(wc) {
            this->set_max_val(N, 0, 0);
            this->set_max_val(N, 0, 1);
        }

        double wa() const { return wa_; }
        double wc() const { return wc_; }
        COMPLEX a() const { return a_; }
        COMPLEX b() const { return b_; }

        int n(size_t cavity_id) const { return this->get_qudit(0, cavity_id); }

        void set_n(int n, size_t cavity_id) { this->set_qudit(n, 0, cavity_id); }

        int m(size_t cavity_id) const { return this->group_size(cavity_id) - 1; }

        int get_atom(size_t atom_id, size_t cavity_id) const { return this->get_qudit(atom_id + 1, cavity_id); }

        void set_atom(int atom_val, size_t atom_id, size_t cavity_id) {
            this->set_qudit(atom_val, atom_id + 1, cavity_id);
        }

        int get_energy() const {
            int res = QConfig::instance().h() * wa_ * (this->n(0) + this->n(1));
            for (size_t cavity_id = 0; cavity_id < 2; cavity_id++) {
                for (size_t i = 0; i < this->m(cavity_id); i++) {
                    res += QConfig::instance().h() * wc_ * this->get_atom(i, cavity_id);
                }
            }

            return res;
        }

       private:
        COMPLEX a_;
        COMPLEX b_;
        double wa_;
        double wc_;
    };

    //H = hwa_1+a_1 + hwa_2+a_2 + sum(i, hwsigma_i+sigma_i) + b((sigma+)a + (a+)sigma) + a_*a_1+a_2 + *a_*a_1a_2+

    State<My_State> energy(const My_State& state) { return State<My_State>(state, state.get_energy()); }

    State<My_State> exc_relax(const My_State& state) {
        auto cur_state = state;
        State<My_State> res;

        //asigma+
        for (size_t cavity_id = 0; cavity_id < 2; cavity_id++) {
            auto cur_n = cur_state.n(cavity_id);
            for (size_t i = 0; i < cur_state.m(cavity_id); i++) {
                auto cur_atom = cur_state.get_atom(i, cavity_id);
                if (cur_n != 0 && cur_atom != 1) {
                    cur_state.set_n(cur_n - 1, cavity_id);
                    cur_state.set_atom(cur_atom + 1, i, cavity_id);

                    res += State<My_State>(cur_state) * cur_state.b() * std::sqrt(cur_n);

                    cur_state.set_n(cur_n, cavity_id);
                    cur_state.set_atom(cur_atom, i, cavity_id);
                }
            }
        }

        //a+sigma
        for (size_t cavity_id = 0; cavity_id < 2; cavity_id++) {
            auto cur_n = cur_state.n(cavity_id);
            for (size_t i = 0; i < cur_state.m(cavity_id); i++) {
                auto cur_atom = cur_state.get_atom(i, cavity_id);
                if (cur_atom == 1) {
                    cur_state.set_n(cur_n + 1, cavity_id);
                    cur_state.set_atom(cur_atom - 1, i, cavity_id);

                    res += State<My_State>(cur_state) * cur_state.b() * std::sqrt(cur_n + 1);

                    cur_state.set_n(cur_n, cavity_id);
                    cur_state.set_atom(cur_atom, i, cavity_id);
                }
            }
        }

        return res;
    }

    State<My_State> photon_transfer(const My_State& state) {
        auto cur_state = state;
        State<My_State> res;

        if (cur_state.n(0) != 0) {
            cur_state.set_n(cur_state.n(0) - 1, 0);
            cur_state.set_n(cur_state.n(1) + 1, 1);

            res +=
                State<My_State>(cur_state) * cur_state.a() * std::sqrt(cur_state.n(0) + 1) * std::sqrt(cur_state.n(1));

            cur_state.set_n(cur_state.n(0) + 1, 0);
            cur_state.set_n(cur_state.n(1) - 1, 1);
        }

        if (cur_state.n(1) != 0) {
            cur_state.set_n(cur_state.n(0) + 1, 0);
            cur_state.set_n(cur_state.n(1) - 1, 1);

            res += State<My_State>(cur_state) * std::conj(cur_state.a()) * std::sqrt(cur_state.n(0)) *
                   std::sqrt(cur_state.n(1) + 1);

            cur_state.set_n(cur_state.n(0) - 1, 0);
            cur_state.set_n(cur_state.n(1) + 1, 1);
        }

        return res;
    }

}  // namespace QComputations

int main(int argc, char** argv) {
    using namespace QComputations;
    size_t N = 2;
    COMPLEX a = 0.01;
    COMPLEX b = 0.01;
    double wa = 1;
    double wc = 1;

    MPI_Init(&argc, &argv);

    My_State state(N, a, b, wa, wc);
    state.set_state("|1;0>|0;0>");

    using OpType = Operator<My_State>;

    auto H_op = OpType(energy) + OpType(exc_relax) + OpType(photon_transfer);

    int ctxt;
    mpi::init_grid(ctxt);

    BLOCKED_H_by_Operator<My_State> H(ctxt, state, H_op);

    auto time_vec = linspace(0, 5000, 5001);
    auto probs = schrodinger(state, H, time_vec);

    for (int j = 0; j < H.size(); j++) {
        double p = probs.get(j, 5000);
        if (is_main_proc()) {
            printf("%lf ", p);
        }
    }

    MPI_Finalize();
    return 0;
}

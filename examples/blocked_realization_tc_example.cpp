/*
Демонстрация реализации собственного понятия состояния и 
операторов. Многоядерная версия.
*/

#include "QComputations_CPU_CLUSTER.hpp"


#include <iostream>

using namespace QComputations;

class TC_State : public Basis_State {
    public:
        // нулевой кудит - число фотонов,
        // остальные - состояния атомов
        explicit TC_State(int m,
                          int n = 0) :
                 Basis_State(m + 1) {
            this->set_max_val(std::max(QConfig::instance().max_photons(), n),
                              0);
            this->set_qudit(n, 0);
        }

        explicit TC_State(const std::string& str) :
            Basis_State(str) {
            this->set_max_val(std::max(QConfig::instance().max_photons(), 1),
                              0);
        }

        void set_n(int n) {this->set_qudit(n, 0);}
        int n() const { return this->get_qudit(0);}
        int m() const { return this->qudits_count() - 1;}

        void set_max_photons(int max_photons) {
            this->set_max_val(max_photons, 0);    
        }

        void set_atom(int val, size_t qudit_index = 0) { 
            this->set_qudit(val, qudit_index + 1);
        }
        int get_atom(size_t qudit_index = 0) const { 
            return this->get_qudit(qudit_index + 1);
        }

        void set_leak(double leak) { leak_ = leak; }
        double get_leak() const { return leak_;}

        void set_gain(double gain) { gain_ = gain;}
        double get_gain() const { return gain_;}

        void set_g(double g) { g_ = g;}
        double g() const { return g_;}

        std::string to_string() const override;

        bool operator<(const Basis_State& other) const override {
            TC_State* b = (TC_State*)(&other);

            if (this->n() < b->n()) return false;
            else if (this->n() == b->n()) {
                for (size_t i = 0; i < this->m(); i++) {
                    if (this->get_atom(i) < b->get_atom(i)) return false;
                    else if (this->get_atom(i) > b->get_atom(i)) return true;
                }
            }

            return true;
        }
    private:
        double leak_; // Интенсивность утечки фотонов
        double gain_; // Интенсивность притока фотонов
        double g_ = QConfig::instance().g(); // Сила взаимодействия электронов с полем
};

std::string TC_State::to_string() const {
    auto res = this->Basis_State::to_string();
    size_t start_pos = 0;
    int index = 1;
    while((start_pos = res.find(";", start_pos)) != std::string::npos) {
        res[start_pos++] = (index > 0 ? ';' : ',');
        index--;
    }

    return res;
}

State<TC_State> photons_count(const TC_State& st) {
    return get_qudit(st, 0);
}

State<TC_State> atoms_count(const TC_State& st) {
    State<TC_State> res(st); // Инициализация всегда происходит с коэффициентом 1
    res[st] = 0;

    for (int i = 0; i < st.m(); i++) {
        res[st] += st.get_atom(i);
    }

    return res;
}

State<TC_State> exc_relax_atoms(const TC_State& st) {
    State<TC_State> res;

    for (int i = 0; i < st.m(); i++) {
        auto tmp = st;
        int n = st.n();

        if (st.get_atom(i) == 1) {
            tmp.set_atom(0, i);
            tmp.set_n(n + 1);
            res += State<TC_State>(tmp) * std::sqrt(n + 1) * st.g(); 
        } else if (st.n() > 0) {
            tmp.set_n(n - 1);
            tmp.set_atom(1, i);
            res += State<TC_State>(tmp) * std::sqrt(n) * st.g(); 
        }
    }

    return res;
}

State<TC_State> a_destroy(const TC_State& st) {
    return set_qudit(st, st.n() - 1, 0) * std::sqrt(st.n());
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    using OpType = Operator<TC_State>;
    double h = QConfig::instance().h();
    double w = QConfig::instance().w();
    std::cout << "h = " << h << " w = " << w << std::endl;

    TC_State state(2);
    state.set_max_photons(1);
    state.set_n(1);
    state.set_leak(0.01);
    std::cout << "Вывод остояния: " << state.to_string() << std::endl;

    OpType H_op = OpType(atoms_count) * h * w + OpType(photons_count) * h * w + OpType(exc_relax_atoms);

    auto res = H_op.run(State<TC_State>(state));

    std::cout << "Вывод состояния: " << res.to_string() << std::endl;

    std::vector<std::pair<double, OpType>> dec;
    OpType A_out(a_destroy);
    dec.emplace_back(state.get_leak(), A_out);

    /* Построить вручную базис + получить оператор декогеренции на нём 
       в матричном виде из оператора

    std::vector<OpType> dec_op = {A_out};
    auto basis = State_Graph<TC_State>(state, H_op, dec_op).get_basis();
    auto A = operator_to_matrix(A_out, basis);
    A.show();
    */

    int ctxt;
    mpi::init_grid(ctxt);

    BLOCKED_H_by_Operator<TC_State> H(ctxt, state, H_op, dec);

    show_basis(H.get_basis());
    H.show();

    if (is_main_proc()) std::cout << "H_size: " << H.size() << std::endl; 

    auto time_vec = linspace(0, 1000, 1000);

    auto probs = quantum_master_equation(state, H, time_vec);

    make_probs_files(H, probs, time_vec, H.get_basis(), "seaborn_results/tc_example_seaborn");

    mpi::blacs_gridexit(ctxt);
    MPI_Finalize();

    return 0;
}
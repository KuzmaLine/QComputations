/*
Демонстрация реализации собственного понятия состояния и 
операторов. Многоядерная версия.
*/

#include "QComputations_CPU_CLUSTER_NO_PLOTS.hpp"


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

using OpType = Operator<TC_State>;

class H_TC : public H_by_Operator<TC_State> {
    public:
        explicit H_TC(const State<TC_State>& state);
};

OpType H_TC_OP() {
    OpType H_op = OpType(atoms_count) * (QConfig::instance().h() * QConfig::instance().w()) + 
                  OpType(photons_count) * (QConfig::instance().h() * QConfig::instance().w()) + OpType(exc_relax_atoms);

    return H_op;
}

std::vector<std::pair<double, OpType>> make_decs(const State<TC_State>& st) {
    std::vector<std::pair<double, OpType>> res;
    res.emplace_back(st(0)->get_leak(), OpType(a_destroy));

    return res;
}

H_TC::H_TC(const State<TC_State>& st): H_by_Operator(st, H_TC_OP(), make_decs(st)) {}

int main(int argc, char** argv) {
    int world_size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    using OpType = Operator<TC_State>;
    double h = QConfig::instance().h();
    double w = QConfig::instance().w();
    //std::cout << "h = " << h << " w = " << w << std::endl;

    TC_State state(1);
    state.set_atom(1, 0);
    state.set_leak(0.01);
    //std::cout << "Вывод остояния: " << state.to_string() << std::endl;

    OpType H_op = OpType(atoms_count) * h * w + OpType(photons_count) * h * w + OpType(exc_relax_atoms);

    auto res = H_op.run(State<TC_State>(state));

    //std::cout << "Вывод состояния: " << res.to_string() << std::endl;

    size_t steps_count = 500;
    double a = 0.01, b = 0.15;
    auto g_leak_range = linspace(a, b, steps_count);

    size_t start, count;
    make_rank_map(g_leak_range.size(), rank, world_size, start, count);

    auto time_vec = linspace(0, 1000, 1000);

    for (size_t i = start; i < count + start; i++) {
        auto g_leak = g_leak_range[i];
        state.set_leak(g_leak);
        H_TC H(state);

        auto probs = quantum_master_equation(state, H, time_vec);

        make_probs_files(H, probs, time_vec, H.get_basis(), "gif_results/tc_g_leak=" + std::to_string(g_leak), rank);
    }

    MPI_Finalize();

    return 0;
}
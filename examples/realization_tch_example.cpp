/*
Данный пример демонстрирует
*/
#ifdef MPI_VERSION
#include "QComputations_CPU_CLUSTER.hpp"
#else
#include "QComputations_SINGLE.hpp"
#endif

#include <tbb/tbb.h>
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
#ifdef MPI_VERSION
    MPI_Init(&argc, &argv);
#endif
    //auto mp = tbb::global_control::max_allowed_parallelism;
    //int one = 1;
    //tbb::global_control gc(mp, one);
    using OpType = Operator<TC_State>;
    double h = QConfig::instance().h();
    double w = QConfig::instance().w();
    double g_leak = 0.01;
    std::cout << "h = " << h << " w = " << w << std::endl;

    TC_State state(2);
    state.set_max_photons(1000);
    state.set_n(1000);
    std::cout << "Вывод состояния: " << state.to_string() << std::endl;

    OpType H_op; 
    H_op = H_op + OpType(atoms_count) * h * w + OpType(photons_count) * h * w + OpType(exc_relax_atoms);

    auto res = H_op.run(State<TC_State>(state));

    std::cout << "Вывод состояния: " << res.to_string() << std::endl;

    std::vector<std::pair<double, OpType>> dec;
    OpType A_out(a_destroy);
    dec.emplace_back(g_leak, A_out);

#ifndef MPI_VERSION
    /* Построить вручную базис + получить оператор декогеренции на нём 
       в матричном виде из оператора

    std::vector<OpType> dec_op = {A_out};
    auto basis = State_Graph<TC_State>(state, H_op, dec_op).get_basis();
    auto A = operator_to_matrix(A_out, basis);
    A.show();
    */

    H_by_Operator<TC_State> H(state, H_op, dec);

    //show_basis(H.get_basis());
    //H.show();

    std::cout << "H_size: " << H.size() << std::endl; 

    auto time_vec = linspace(0, 1000, 1000);

    auto start = std::chrono::high_resolution_clock::now();
    auto probs = quantum_master_equation(state, H, time_vec);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

#else

    /* Построить вручную базис + получить оператор декогеренции на нём 
       в матричном виде из оператора
    */
    std::vector<OpType> dec_op = {A_out};
    auto basis = State_Graph<TC_State>(state, H_op, dec_op).get_basis();
    //auto A = operator_to_matrix(A_out, basis);
    //A.show();
    

    auto start = std::chrono::high_resolution_clock::now();
    H_by_Operator<TC_State> H(state, H_op, dec);
    auto end = std::chrono::high_resolution_clock::now();

    if (is_main_proc()) {
        std::cout << "H GEN: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;
    }

    //show_basis(H.get_basis());
    //H.show();

    std::cout << "H_size: " << H.size() << std::endl; 

    auto time_vec = linspace(0, 1000, 1000);

    /*
    auto start = std::chrono::high_resolution_clock::now();
    auto probs = quantum_master_equation(state, H, time_vec);
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    if (is_main_proc()) {
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;
    }
    */
    auto H_m = H.get_matrix();
    auto A = operator_to_matrix(A_out, basis);
    start = std::chrono::high_resolution_clock::now();
    auto T = H_m * A;
    end = std::chrono::high_resolution_clock::now();
    

    MPI_Barrier(MPI_COMM_WORLD);
    if (is_main_proc()) {
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;
        /*
        matplotlib::make_figure(1900, 1200, 80);
        matplotlib::xlabel("Time");
        matplotlib::ylabel("Probability");
        matplotlib::grid();
        matplotlib::title("TC: leak_photons=" + std::to_string(g_leak));
        matplotlib::probs_to_plot(probs, time_vec, H.get_basis());
        // Для слишком больших базисов легенда бесполезна без интерактивных графиков.
        // Они планируются позже
        matplotlib::legend();
        //std::cout << "HERE\n";
        matplotlib::show();
        */
    }

    int ctxt;
    mpi::init_grid(ctxt);

    start = std::chrono::high_resolution_clock::now();
    BLOCKED_H_by_Operator<TC_State> H_b(ctxt, state, H_op, dec);
    end = std::chrono::high_resolution_clock::now();

    if (is_main_proc()) {
        std::cout << "BLOCKED_H GEN: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;
    }

    //show_basis(H.get_basis());
    //H.show();

    if (is_main_proc()) std::cout << "H_size: " << H_b.size() << std::endl; 

    time_vec = linspace(0, 1000, 1000);

    /*
    start = std::chrono::high_resolution_clock::now();
    auto b_probs = quantum_master_equation(state, H_b, time_vec);
    MPI_Barrier(MPI_COMM_WORLD);
    end = std::chrono::high_resolution_clock::now();

    if (is_main_proc()) {
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;
    }
    */

    auto H_b_m = H_b.get_blocked_matrix();
    auto A_b = operator_to_matrix(ctxt, A_out, basis);
    start = std::chrono::high_resolution_clock::now();
    auto T_b = H_b_m * A_b;
    MPI_Barrier(MPI_COMM_WORLD);
    end = std::chrono::high_resolution_clock::now();


    if (is_main_proc()) {
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;
        /*
        matplotlib::make_figure(1900, 1200, 80);
        matplotlib::xlabel("Time");
        matplotlib::ylabel("Probability");
        matplotlib::grid();
        matplotlib::title("TC: leak_photons=" + std::to_string(g_leak));
    }

    matplotlib::probs_to_plot(b_probs, time_vec, H_b.get_basis());

    if (is_main_proc()) {
        // Для слишком больших базисов легенда бесполезна без интерактивных графиков.
        // Они планируются позже
        matplotlib::legend();
        //std::cout << "HERE\n";
        matplotlib::show();
        */
    }

    /*
    probs = schrodinger(state, H, time_vec);

    if (is_main_proc()) {
        matplotlib::make_figure(1900, 1200, 80);
        matplotlib::xlabel("Time");
        matplotlib::ylabel("Probability");
        matplotlib::grid();
        matplotlib::title("TC: leak_photons=" + std::to_string(g_leak));
    }
    matplotlib::probs_to_plot(probs, time_vec, H.get_basis());
    if (is_main_proc()) {
        std::cout << "HERE\n";
        //matplotlib::show();
    }

    H_by_Operator<TC_State> H_one(state, H_op, dec);

    //show_basis(H.get_basis());
    //H.show();

    start = std::chrono::high_resolution_clock::now();
    auto probs_one = schrodinger(state, H_one, time_vec);
    end = std::chrono::high_resolution_clock::now();

    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

    if (is_main_proc()) {
        matplotlib::make_figure(1900, 1200, 80);
        matplotlib::xlabel("Time");
        matplotlib::ylabel("Probability");
        matplotlib::grid();
        matplotlib::title("TC: leak_photons=" + std::to_string(g_leak));
    }
    matplotlib::probs_to_plot(probs_one, time_vec, H_one.get_basis());
    if (is_main_proc()) {
        std::cout << "HERE\n";
        matplotlib::show();
    }
    */
#endif

#ifdef MPI_VERSION
    MPI_Finalize();
#endif
    return 0;
}
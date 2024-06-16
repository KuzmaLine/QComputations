/*
Данный пример демонстрирует
*/
#ifdef MPI_VERSION
#include "QComputations_CPU_CLUSTER.hpp"
#else
#include "QComputations_SINGLE.hpp"
#endif

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
    private:
        double leak_; // Интенсивность утечки фотонов
        double gain_; // Интенсивность притока фотонов
        double g_ = QConfig::instance().g(); // Сила взаимодействия электронов с полем
};

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

int main(int argc, char** argv) {
#ifdef MPI_VERSION
    MPI_Init(&argc, &argv);
#endif
    using OpType = Operator<TC_State>;
    double h = QConfig::instance().h();
    double w = QConfig::instance().w();
    std::cout << "h = " << h << " w = " << w << std::endl;

    TC_State state(2);
    state.set_atom(1, 0);
    
    std::cout << "Вывод состояния: " << state.to_string() << std::endl;

    OpType H = (OpType(atoms_count) * h * w + OpType(photons_count) * h * w + OpType(exc_relax_atoms)) * double(3);

    //H.show();
    auto res = H.run(State<TC_State>(state));

    std::cout << "Вывод состояния: " << res.to_string() << std::endl;

#ifdef MPI_VERSION
    MPI_Finalize();
#endif
    return 0;
}
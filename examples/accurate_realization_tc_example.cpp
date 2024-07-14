/*
Демонстрация реализации собственного понятия состояния и 
операторов. Одноядерная версия.
*/
#include "QComputations_SINGLE.hpp"

#include <iostream>

namespace QComputations {

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

State<TC_State> a_destroy(const TC_State& st) {
    return set_qudit(st, st.n() - 1, 0) * std::sqrt(st.n());
}

State<TC_State> a_create(const TC_State& st) {
    return set_qudit(st, st.n() + 1, 0) * std::sqrt(st.n() + 1);
}

int main(int argc, char** argv) {
    using OpType = Operator<TC_State>;
    double h = QConfig::instance().h();
    double w = QConfig::instance().w();
    double g_leak = 0.01;
    

    TC_State state(2);
    state.set_max_photons(1);
    state.set_n(1);
    double g = state.g();

    std::cout << "h = " << h << " w = " << w << " g = " << g << std::endl;
    std::cout << "Вывод состояния: " << state.to_string() << std::endl;
    
    OpType H_op = OpType(a_create) * OpType(a_destroy) * (h * w); // hwaa+

    // Для каждого атома добавляем его собственные операторы
    for (size_t i = 0; i < state.m(); i++) {
        //sigma
        auto sigma = std::function<State<TC_State>(const TC_State&)> {
            [i](const TC_State& st) {
                return set_qudit(st, st.get_atom(i) - 1, i + 1);
            }
        };

        //sigma+
        auto sigma_exc = std::function<State<TC_State>(const TC_State&)> {
            [i](const TC_State& st) {
                return set_qudit(st, st.get_atom(i) + 1, i + 1);
            }
        };

        // Энергия атома
        H_op = H_op + OpType(sigma_exc) * OpType(sigma) * (h * w);

        // Возбуждение и релаксация атома
        H_op = H_op + (OpType(a_destroy) * OpType(sigma_exc) + 
               OpType(a_create) * OpType(sigma)) * g;
    }

    //auto res = H_op.run(State<TC_State>(state));

    //std::cout << "Вывод состояния: " << res.to_string() << std::endl;

    std::vector<std::pair<double, OpType>> dec;
    OpType A_out(a_destroy);
    dec.emplace_back(g_leak, A_out);

    H_by_Operator<TC_State> H(state, H_op, dec);

    show_basis(H.get_basis());
    H.show();

    return 0;

    std::cout << "H_size: " << H.size() << std::endl; 

    auto time_vec = linspace(0, 1000, 1000);

    auto probs = quantum_master_equation(state, H, time_vec);

    matplotlib::make_figure(1200, 800, 80);
    matplotlib::xlabel("Time");
    matplotlib::ylabel("Probability");
    matplotlib::grid();
    matplotlib::title("TC: leak_photons=" + std::to_string(g_leak));
    matplotlib::probs_to_plot(probs, time_vec, H.get_basis());

    // Для слишком больших базисов легенда бесполезна без интерактивных графиков.
    // Они планируются позже
    matplotlib::legend();
    matplotlib::show();

    return 0;
}

} // namespace QComputations
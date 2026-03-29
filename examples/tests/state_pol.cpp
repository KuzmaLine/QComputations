#include "QComputations_SINGLE_NO_PLOTS.hpp"

using namespace QComputations;



class MyState : public Basis_State {
    public:
        MyState(size_t n, size_t m, size_t start, size_t end): start_(start), end_(end), Basis_State(n, 1, m) {
            for (size_t i = start; i <= end; i++) {
                this->set_qudit(1, i);
            }
        }

        std::string to_string() const override {
            size_t num = 0;
            size_t power2 = 1;
            for (size_t i = 0; i < this->qudits_count(); i++) {
                num += power2 * this->get_qudit(i);
                power2 *= 2;
            }
            std::string res = "|" + std::to_string(num) + ">";
            return res;
        }
    private:
        size_t start_;
        size_t end_;
};

void show_basis(const State<MyState>& state) {
    auto basis = state.get_basis();

    for (auto s: basis) {
        std::cout << s->to_string() << " ";
    }

    std::cout << std::endl;
}

void show_map(const State<MyState>& state) {
    auto map = state.state_map();
    for (auto p: map) {
        std::cout << (p.first)->to_string() << ": " << p.second << " | ";
    }

    std::cout << std::endl;
}

int main(int argc, char** argv) {
    MyState st(4, 2, 1, 2);
    std::cout << st.to_string() << std::endl;

    State<MyState> state_st(st);
    std::cout << state_st.to_string() << std::endl;

    show_basis(state_st);
    show_map(state_st);

    st.set_qudit(1, 0);
    state_st.insert(st, 0.5);

    show_basis(state_st);
    show_map(state_st);

    std::cout << state_st.to_string() << std::endl;

    st.set_qudit(1, 3);
    state_st.insert(st, 0.2);
    st.set_qudit(0, 1);
    state_st.insert(st, 0.4);

    std::cout << state_st.to_string() << " " << state_st.size() << std::endl;
    state_st.sort();
    std::cout << state_st.to_string() << " " << state_st.size() << std::endl;

    show_basis(state_st);

    return 0;
}
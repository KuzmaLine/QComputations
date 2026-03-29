#include "QComputations_SINGLE_NO_PLOTS.hpp"

using namespace QComputations;

class CreateState : public Basis_State {
    public:
        CreateState(size_t n, size_t m, size_t start, size_t end): start_(start), end_(end), Basis_State(n, 1, m) {
            for (size_t i = start; i <= end; i++) {
                this->set_qudit(1, i);
            }
        }
    private:
        size_t start_;
        size_t end_;
};

class PolyState : public Basis_State {
    public:
        PolyState(size_t n, size_t start, size_t end): start_(start), end_(end), Basis_State(n) {
            for (size_t i = start; i <= end; i++) {
                this->set_qudit(1, i);
            }
        }

        std::string to_string() const override {
            std::string res = "|" + std::to_string(this->qudits_count()) + ">";
            return res;
        }
    private:
        size_t start_;
        size_t end_;
};

int main(int argc, char** argv) {
    CreateState st(6, 2, 1, 4);
    std::cout << st.to_string() << " " << st.get_group_end(0) <<  " " << st.get_group_start(1) << std::endl;

    st.set_group(0, "|0;0;1>");

    std::cout << st.to_string() << std::endl;

    auto st_group = st.get_group(1);

    std::cout << st_group.to_string() << std::endl;

    PolyState st_pol(6, 1, 4);
    std::cout << st_pol.to_string() << " ";
    Basis_State& ptr = st_pol;
    std::cout << ptr.to_string() << std::endl;

    return 0;
}
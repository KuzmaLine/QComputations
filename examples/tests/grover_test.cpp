#include "QComputations_SINGLE_NO_PLOTS.hpp"

namespace QComputations {

class Grover_State : public Basis_State {
    public:
        Grover_State(size_t n): Basis_State(n) {}

        size_t to_num(const Grover_State& state) {
            size_t power2 = 1;
            size_t num = 0;

            for (size_t i = this->qudits_count(); i > 0; i--) {
                num += power2 * this->get_qudit(i - 1);
                power2 *= 2;
            }

            return num;
        }
};

}

int main(int argc, char** argv) {
    using namespace QComputations;
    Grover_State st(3);
    st.set_qudit(1);

    using OpType = Operator<Grover_State>;
    auto WH_op = OpType(WH);
    auto state_res = WH_op.run(st);

    std::cout << state_res.to_string() << std::endl;

    auto basis = make_full_basis(std::make_shared<Grover_State>(st));

    show_basis(sort_basis(basis));

    auto WH_mat = operator_to_matrix(WH_op, sort_basis(basis));

    WH_mat.show();

    return 0;
}
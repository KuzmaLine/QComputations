#include "QComputations_CPU_CLUSTER.hpp"

namespace QComputations {

inline bool is_greater(const std::vector<ValType>& a, const std::vector<ValType>& b) {
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] > b[i]) return true;
    }

    return false;
}

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

        inline bool operator<(const Basis_State& other) const override { return is_greater(this->qudits_, other.qudits_ref()); }
};

}

int main(int argc, char** argv) {
    using namespace QComputations;
    MPI_Init(&argc, &argv);
    ILP_TYPE ctxt;
    mpi::init_grid(ctxt);

    Grover_State st(14);
    st.set_qudit(1);

    using OpType = Operator<Grover_State>;
    auto WH_op = OpType(WH);
    auto state_res = WH_op.run(st);

    // std::cout << state_res.to_string() << std::endl;

    auto basis = make_full_basis(std::make_shared<Grover_State>(st));

    // if (is_main_proc()) show_basis(sort_basis(basis));

    auto WH_mat = operator_to_matrix(ctxt, WH_op, sort_basis(basis));
    // auto WH_mat = WH_Matrix(ctxt, sort_basis(basis));

    // WH_mat.show();

    MPI_Finalize();
    return 0;
}
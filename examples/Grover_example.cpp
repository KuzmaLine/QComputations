#include "QComputations_CPU_CLUSTER.hpp"
#include <iostream>
#include <chrono>

bool func(int num) {
    return num == 7;
}

int vec_to_num(const std::vector<int>& v) {
    int res = 0;

    for (size_t i = 0; i < v.size(); i++) {
        res <<= 1;
        res += v[i];
    }

    return res;
}

namespace QComputations {

class GROVER_State: public Basis_State {
    public:
        explicit GROVER_State(size_t bits_count, int num = 0): bits_count_(bits_count),
                                                  Basis_State(bits_count) {
            for (size_t i = 0; i < bits_count; i++) {
                this->set_qudit(num % 2, bits_count - i - 1);
                num /= 2;
            }
        }

        size_t size() const { return bits_count_;}

        int to_num() const {
            return vec_to_num(this->qudits_);
        }

        bool is_zero() const {
            return this->to_num() == 0;
        }

        std::string to_string() const override {
            std::string res = "|" + std::to_string(this->to_num()) + ">";

            return res;
        }

        bool operator<(const Basis_State& other) const override {
            return this->to_num() < vec_to_num(other.qudits());
        }
    private:
        size_t bits_count_;
};

State<GROVER_State> I_x_tar(const GROVER_State& st) {
    return State<GROVER_State>(st) * std::pow(-1, int(func(st.to_num())));
}

// U = 2|0><0| - I
State<GROVER_State> I_zero(const GROVER_State& st) {
    if (st.is_zero()) return State<GROVER_State>(st);
    else return State<GROVER_State>(st) * (-1);
}

void eff_I_x_tar(const GROVER_State& st, State<GROVER_State>& res) {
    res.insert(st, COMPLEX(std::pow(-1, int(func(st.to_num()))), 0));
    //return State<GROVER_State>(st) * std::pow(-1, int(func(st.to_num())));
}

// U = 2|0><0| - I
void eff_I_zero(const GROVER_State& st, State<GROVER_State>& res) {
    res.insert(st, COMPLEX(-1 + 2 * int(st.is_zero()), 0));
    //if (st.is_zero()) return State<GROVER_State>(st);
    //else return State<GROVER_State>(st) * (-1);
}

}

int main(int argc, char** argv) {
    int world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    using namespace QComputations;
    using OpType = Operator<GROVER_State>;
    GROVER_State st(10, 0);

    //std::cout << st.to_string() << " " << func(st.to_num()) << std::endl;

    State<GROVER_State> init_state(st, make_full_basis(st));
    init_state.set_all(COMPLEX(1 / std::sqrt(init_state.state_components().size()), 0));
    auto start = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();

    const size_t size = init_state.size() * init_state.size();
    auto b = init_state.get_state_components();
    start = std::chrono::steady_clock::now();
    std::vector<std::shared_ptr<GROVER_State>> b_vec;
    std::copy(b.begin(), b.end(), std::back_inserter(b_vec));
    auto get_time = std::chrono::duration_cast<std::chrono::microseconds>(start - start).count();
    auto index_time = std::chrono::duration_cast<std::chrono::microseconds>(start - start).count();
    auto for_time = std::chrono::duration_cast<std::chrono::microseconds>(start - start).count();
    auto and_time = std::chrono::duration_cast<std::chrono::microseconds>(start - start).count();
    for (int i = 0; i < size; i++) {
        auto start_1 = std::chrono::steady_clock::now();
        auto a_1 = get_state_from_basis(b, i % init_state.size())->qudits_data();
        auto a_2 = get_state_from_basis(b, ((i + 1) % init_state.size()))->qudits_data();
        auto end_1 = std::chrono::steady_clock::now();
        get_time += std::chrono::duration_cast<std::chrono::microseconds>(end_1 - start_1).count();

        start_1 = std::chrono::steady_clock::now();
        a_1 = b_vec[i % init_state.size()]->qudits_data();
        a_2 = b_vec[((i + 1) % init_state.size())]->qudits_data();
        end_1 = std::chrono::steady_clock::now();
        index_time += std::chrono::duration_cast<std::chrono::microseconds>(end_1 - start_1).count();

        auto start_2 = std::chrono::steady_clock::now();
        and_sum(a_1, a_2, init_state.size());
        auto end_2 = std::chrono::steady_clock::now();
        and_time += std::chrono::duration_cast<std::chrono::microseconds>(end_2 - start_2).count();
    }
    end = std::chrono::steady_clock::now();
    std::cout << "SCALAR = " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << std::endl;
    std::cout << get_time << ' ' << index_time << " " << and_time << std::endl;

    std::vector<int> tmp(init_state.size());
    size_t index_tmp = 0;
    start = std::chrono::steady_clock::now();
    for (auto ptr: init_state.state_components()) {
        tmp[index_tmp++] = ptr->size();
    }
    end = std::chrono::steady_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()<< std::endl;

    int vec_ctxt;
    mpi::init_vector_grid(vec_ctxt);

    //OpType G = OpType(-1) * OpType(WH) * OpType(I_zero) * OpType(WH) * OpType(I_x_tar);
    

    start = std::chrono::steady_clock::now();
    auto WH_mat = WH_Matrix(vec_ctxt, init_state.state_components());
    MPI_Barrier(MPI_COMM_WORLD);
    end = std::chrono::steady_clock::now();
    if (is_main_proc()) std::cout << "WH_mat = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

    start = std::chrono::steady_clock::now();
    auto I_x_tar_mat = operator_to_matrix(vec_ctxt, OpType(I_x_tar), init_state.state_components());
    MPI_Barrier(MPI_COMM_WORLD);
    end = std::chrono::steady_clock::now();
    if (is_main_proc()) std::cout << "I_x_tar = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

    start = std::chrono::steady_clock::now();
    auto I_zero_mat = operator_to_matrix(vec_ctxt, OpType(I_zero), init_state.state_components());
    MPI_Barrier(MPI_COMM_WORLD);
    end = std::chrono::steady_clock::now();
    if (is_main_proc()) std::cout << "I_zero = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    start = std::chrono::steady_clock::now();
    auto eff_I_x_tar_mat = operator_to_matrix(vec_ctxt, OpType(eff_I_x_tar), init_state.state_components());
    MPI_Barrier(MPI_COMM_WORLD);
    end = std::chrono::steady_clock::now();
    if (is_main_proc()) std::cout << "eff_I_x_tar = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    start = std::chrono::steady_clock::now();
    auto eff_I_zero_mat = operator_to_matrix(vec_ctxt, OpType(eff_I_zero), init_state.state_components());
    MPI_Barrier(MPI_COMM_WORLD);
    end = std::chrono::steady_clock::now();
    if (is_main_proc()) std::cout << "eff_I_zero = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;

    auto G_mat = WH_mat * I_zero_mat * WH_mat * I_x_tar_mat * (-1);

    auto steps_count = int(M_PI * std::sqrt(std::pow(2, st.size())) / 4);

    int nums_count = 10;

    //show_basis(init_state.state_components());

    BLOCKED_Vector<COMPLEX> blocked_state_vec(vec_ctxt, init_state.vector());
 
    BLOCKED_Matrix<double> probs(vec_ctxt, GE, nums_count, steps_count + 1, G_mat.NB(), G_mat.MB());

    for (size_t j = 0; j < probs.local_n(); j++) {
        auto prob = std::abs(init_state[probs.get_global_row(j)]) * std::abs(init_state[probs.get_global_row(j)]);
        probs(j, 0) = prob;
        //std::cout << std::setw(QConfig::instance().width()) << prob << " ";
    }

    for (size_t step = 0; step < steps_count; step++) {
        //std::cout << "PROGRESS: " << step << " of " << steps_count << std::endl;
        blocked_state_vec = G_mat * blocked_state_vec;

        for (size_t j = 0; j < probs.local_n(); j++) {
            auto prob = std::abs(blocked_state_vec[j]) * std::abs(blocked_state_vec[j]);
            probs(j, step + 1) = prob;
            //std::cout << std::setw(QConfig::instance().width()) << prob << " ";
        }

        //std::cout << std::endl;
    }

    auto steps = linspace(0, steps_count, steps_count + 1);

    std::vector<std::string> basis_str;

    for (size_t i = 0; i < nums_count; i++) {
        if (i != nums_count - 1) {
            basis_str.emplace_back(get_state_from_basis(init_state.state_components(), i)->to_string());
        } else {
            basis_str.emplace_back("...");
        }
    }

    make_plot_files(probs, steps, basis_str, "GROVER_RESULT_12_BITS");

    MPI_Finalize();

    return 0;
}
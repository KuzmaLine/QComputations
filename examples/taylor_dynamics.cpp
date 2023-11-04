//P #include "/home/kuzmaline/Quantum/diploma/src/QComputations_CPU_CLUSTER.hpp"
#include "/home/kuzmaline/Quantum/diploma/src/QComputations_CPU_CLUSTER_NO_PLOTS.hpp"
#include <iostream>
#include <vector>
#include <complex>
#include <unistd.h>

using COMPLEX = std::complex<double>;
constexpr int MAX_LEVEL = 1; // максимальный энергетический уровень атомов
constexpr double h = 1; // постоянная планка
constexpr double w_ph = 1; // частота фотонов
constexpr double w_at = 1; // частота атомов
constexpr double g = 0.01; // энергия взаимодействия поля с атомом

class C_State {
    public:
        explicit C_State(size_t n, const std::vector<int> atoms_states): n_(n), atoms_states_(atoms_states) {}
        std::string to_string() const;
        size_t n() const { return n_; }
        void set_n(size_t n) { n_ = n; }
        std::vector<int> atoms_states() const { return atoms_states_; }
        int get_qubit(size_t atom_id) const { return atoms_states_[atom_id]; }
        void set_qubit(size_t atom_id, size_t level) { atoms_states_[atom_id] = level; }
        size_t size() const { return atoms_states_.size(); }

        bool operator==(const C_State& other) const { return n_ == other.n_ and atoms_states_ == other.atoms_states_; }
        bool operator!=(const C_State& other) const { return n_ != other.n_ or atoms_states_ != other.atoms_states_; }
        bool operator<(const C_State& other) const { return this->to_string() > other.to_string(); }
    private:
        std::vector<int> atoms_states_; // уровни атомов
        size_t n_; // число фотонов
};

/*
* * * * * * * |7>

|1> -> |0>   |0> -> |1>
a+sigma      asigma+

----- |1>

----0- |0>
*/

std::string C_State::to_string() const {
    std::string res = "|" + std::to_string(this->n());

    if (this->size() != 0) {
        res += ";";
        for (size_t i = 0; i < this->size(); i++) {
            res += std::to_string(this->get_qubit(i));
        }
    }

    res += ">";

    return res;
}

//H = hwa+a + hwsigma+sigma + g(a+sigma + asigma+)

//|1>|001> -> |0>|101>, |0>|011>, |2>|000>

bool is_in_basis(const std::set<C_State>& basis, const C_State& state) {
    return std::find(basis.begin(), basis.end(), state) != basis.end();
}

std::set<C_State> generate_basis(const C_State& init_state) {
    std::set<C_State> basis;
    std::queue<C_State> queue;
    basis.insert(init_state);
    queue.push(init_state);

    while(!queue.empty()) {
        auto cur_state = queue.front();
        queue.pop();

        //a+sigma
        for (size_t i = 0; i < cur_state.size(); i++) {
            if (cur_state.get_qubit(i) == MAX_LEVEL) {
                cur_state.set_qubit(i, 0);
                cur_state.set_n(cur_state.n() + 1);
                if (!is_in_basis(basis, cur_state)) {
                    basis.insert(cur_state);
                    queue.push(cur_state);
                }

                cur_state.set_qubit(i, MAX_LEVEL);
                cur_state.set_n(cur_state.n() - 1);
            }
        }

        //asigma+
        if (cur_state.n() != 0) {
            for (size_t i = 0; i < cur_state.size(); i++) {
                if (cur_state.get_qubit(i) == 0) {
                    cur_state.set_qubit(i, MAX_LEVEL);
                    cur_state.set_n(cur_state.n() - 1);
                    if (!is_in_basis(basis, cur_state)) {
                        basis.insert(cur_state);
                        queue.push(cur_state);
                    }

                    cur_state.set_qubit(i, 0);
                    cur_state.set_n(cur_state.n() + 1);
                }
            }
        }
    }

    return basis;
}

//a+a
// a+a = ((0 0 0 0)
//        (0 hw 0 0)
//        (0 0 2hw 0) ... )
// a+a|1;001> = hw|1;001>
// a+a|ksi> = hw*n|ksi>
COMPLEX self_energy_photons(const C_State& state_from, const C_State& state_to) {
    if (state_from != state_to) {
        return 0;
    }

    return state_from.n() * h * w_ph;
}


// sigma+sigma
COMPLEX self_energy_atoms(const C_State& state_from, const C_State& state_to) {
    if (state_from != state_to) {
        return 0;
    }

    COMPLEX res(0, 0);
    for (size_t i = 0; i < state_from.size(); i++) {
        res += state_from.get_qubit(i);
    }

    return res * h * w_at;
}


// asigma+
COMPLEX excitation_atom(const C_State& state_from, const C_State& state_to) {
    long atom_pos = -1;

    if (state_from.n() != state_to.n() + 1) return 0;

    for (size_t j = 0; j < state_from.size(); j++) {
        if (state_from.get_qubit(j) != state_to.get_qubit(j)) {
            if (state_from.get_qubit(j) == 0 and state_to.get_qubit(j) == MAX_LEVEL and atom_pos == -1) {
                atom_pos = j;
            } else {
                return 0;
            }
        }
    }

    if (atom_pos == -1) {
        return 0;
    }

    //std::cout << "exc - " << photon_pos << " " << state_from.to_string() << " " << state_to.to_string() << std::endl;

    return g * COMPLEX(std::sqrt(state_from.n()));
}

//a+sigma
COMPLEX de_excitation_atom(const C_State& state_from, const C_State& state_to) {
    long atom_pos = -1;

    if (state_from.n() != state_to.n() - 1) return 0;

    for (size_t j = 0; j < state_from.size(); j++) {
        if (state_from.get_qubit(j) != state_to.get_qubit(j)) {
            if (state_from.get_qubit(j) == 1 and state_to.get_qubit(j) == 0 and atom_pos == -1) {
                atom_pos = j;
            } else {
                return 0;
            }
        }
    }

    if (atom_pos == -1) {
        return 0;
    }

    //std::cout << "exc - " << photon_pos << " " << state_from.to_string() << " " << state_to.to_string() << std::endl;

    return g * COMPLEX(std::sqrt(state_from.n() + 1));
}

/*
template<typename T>
T get_elem_from_set(const std::set<T>& st, size_t index) {
    auto it = st.begin();
    std::advance(it, index);
    return *it;
}
*/

COMPLEX TC_ADD(const C_State& state_from, const C_State& state_to) {
    COMPLEX res(0, 0);

    res += self_energy_atoms(state_from, state_to);
    res += self_energy_photons(state_from, state_to);
    res += excitation_atom(state_from, state_to);
    res += de_excitation_atom(state_from, state_to);

    return res;
}

/*
void mpi::init_grid(ILP_TYPE& ctxt, ILP_TYPE proc_rows, ILP_TYPE proc_cols) {
    ILP_TYPE iZERO = 0;
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    ILP_TYPE myid, numproc, myrow, mycol;
    char order = 'R';
    if (proc_rows == 0 or proc_cols == 0) {
        proc_rows = std::sqrt(world_size);
        proc_cols = world_size / proc_rows;
    }
    //std::cout << rank << " Here1\n";
    blacs_pinfo(&myid, &numproc);
    ILP_TYPE iMINUS = -1;
    blacs_get(&iMINUS, &iZERO, &ctxt);
    //std::cout << rank << " Here3\n";
    blacs_gridinit(&ctxt, &order, &proc_rows, &proc_cols);
}
*/

int main(int argc, char** argv) {
    using namespace QComputations;
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    size_t width = 17;
    double dt = 1e-3;
    size_t steps_count = 1e6;
    size_t cout_delay = 1000;

    int ctxt;
    mpi::init_grid(ctxt);

    //P std::vector<size_t> grid_config = {3}; 
    //P State new_state(grid_config);
    //P new_state.set_n(1);
    C_State state(1, {0, 0, 0});


    //P auto basis = State_Graph(new_state).get_basis();
    auto basis = generate_basis(state);

    if (rank == 0) {
        for (const auto& st: basis) {
            std::cout << std::setw(width) << st.to_string() << " ";
        }

        std::cout << std::endl;
    }

    //P std::function<COMPLEX(size_t, size_t)> func = [&basis, &new_state](size_t i, size_t j) {
    std::function<COMPLEX(size_t, size_t)> func = [&basis](size_t i, size_t j) {
        C_State state_from = get_elem_from_set(basis, j);
        //P State state_from = get_elem_from_set(basis, j);
        //P State state_to = get_elem_from_set(basis, i);
        C_State state_to = get_elem_from_set(basis, i);
        return TC_ADD(state_from, state_to);
        //P return TC_ADD(state_from, state_to, new_state);
    };

    BLOCKED_Matrix<COMPLEX> H(ctxt, HE, basis.size(), basis.size(), func);

    H.show(width);
    H.print_distributed("H_TC");

    auto p = Hermit_Lanczos(H);
    auto eigen_values = p.first;
    auto eigen_vectors = p.second;

    BLOCKED_Matrix<COMPLEX> D(ctxt, GE, basis.size(), basis.size(), 0);

    for (size_t i = 0; i < basis.size(); i++) {
        D.set(i, i, std::exp(COMPLEX(0, -1) * eigen_values[i] * dt / h));
    }

    auto U = eigen_vectors * D * eigen_vectors.hermit();

    std::vector<COMPLEX> init_state(basis.size(), 0);
    init_state[0] = COMPLEX(1, 0);

    // |ksi><ksi|;
    std::function<COMPLEX(size_t, size_t)> rho_func = [&init_state](size_t i, size_t j) {
        return init_state[i] * std::conj(init_state[j]);
    };

    BLOCKED_Matrix<COMPLEX> rho(ctxt, GE, basis.size(), basis.size(), rho_func);
    auto init_rho = rho;

    /*P
    Matrix<double> probs(C_STYLE, basis.size(), steps_count + 1);

    for (size_t i = 0; i < rho.n(); i++) { 
        probs[i][0] = std::abs(rho.get(i, i));
        //std::cout << std::setw(width) << std::abs(diag[i]) << " ";
    }
    */

    size_t t = 0;
    std::vector<double> time_vec = {0};
    while(t < steps_count) {
        auto cur_rho = U * init_rho * U.hermit();
        auto diag = mpi::get_diagonal_elements(cur_rho.get_local_matrix(), cur_rho.desc());
        init_rho = cur_rho;

        usleep(cout_delay);
        for (size_t i = 0; i < diag.size(); i++) { 
            if (rank == 0) {
                //P probs[i][t + 1] = std::abs(diag[i]);
                std::cout << std::setw(width) << std::abs(diag[i]) << " ";
            }
        }

        if (rank == 0) std::cout << std::endl;
        t++;
        //P time_vec.emplace_back(dt * t);
    }

    /* P
    if (rank == 0) {
        matplotlib::make_figure(1920, 1080);
    }

    if (rank == 0) {
        matplotlib::probs_to_plot(probs, time_vec, basis);
    }
    //matplotlib::probs_in_cavity_to_plot(probs, time_vec, H.get_basis(), 0);
    if (rank == 0) {
        matplotlib::grid();
        matplotlib::show();
    }
    */

    MPI_Finalize();
    return 0;
}
#include "hamiltonian.hpp"

#include <complex>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>

#include "config.hpp"
#include "functions.hpp"
#include "graph.hpp"

#ifdef ENABLE_MPI
#include "mpi_functions.hpp"
#endif

namespace QComputations {

    namespace {
        typedef std::complex<double> COMPLEX;

        /*
    Matrix<COMPLEX> a_destroy(size_t n) {
        size_t size = n + 1;

        Matrix<COMPLEX> a(DEFAULT_MATRIX_STYLE, size, size, 0);

        int j = 1;
        for (int i = 0; i < size - 1; i++) {
            a[i][j] = std::sqrt(j);
            j++;
        }

        return a;
    }

    Matrix<COMPLEX> a_create(size_t n) {
        return a_destroy(n).transpose();
    }

    Matrix<COMPLEX> E_photons(int n) {
        Matrix<COMPLEX> E(DEFAULT_MATRIX_STYLE, n, n, 0);
        for (int i = 0; i < n; i++) {
            E[i][i] = i * QConfig::instance().h() * QConfig::instance().w();
        }

        return E;
    }


    const Matrix<COMPLEX> eye({{1, 0},
                               {0, 1}});

    const Matrix<COMPLEX> sigma_energy({{0, 0},
                                        {0, 1}});

    const Matrix<COMPLEX> sigma_down({{0, 1},
                                      {0, 0}});

    const Matrix<COMPLEX> sigma_up({{0, 0},
                                    {1, 0}});

    Matrix<COMPLEX> sum_sigma_down(size_t n) {
        size_t size = std::pow(2, n);
        Matrix<COMPLEX> sum_sigma(DEFAULT_MATRIX_STYLE, size, size, 0);

        for (int i = 0; i < n; i++) {
            size_t left_size = std::pow(2, i);
            size_t right_size = std::pow(2, n - i - 1);

            Matrix<COMPLEX> eye_left(DEFAULT_MATRIX_STYLE, left_size, left_size, 0);
            Matrix<COMPLEX> eye_right(DEFAULT_MATRIX_STYLE, right_size, right_size, 0);

            for (int j = 0; j < left_size; j++) {
                eye_left[j][j] = 1;
            }
            for (int j = 0; j < right_size; j++) {
                eye_right[j][j] = 1;
            }

            Matrix<COMPLEX> sigma = sigma_down;

            sigma = tensor_multiply(eye_left, sigma);
            sigma = tensor_multiply(sigma, eye_right);

            sum_sigma += sigma;
        }

        return sum_sigma;
    }

    void add_a_operators(Matrix<COMPLEX>& H, size_t from_id, size_t to_id, const State& grid) {
        size_t size_left = 1, size_middle = 1, size_right = 1;
        size_t cavity_id = 0;

        COMPLEX gamma = grid.get_gamma(from_id, to_id);

        for (cavity_id = 0; cavity_id < from_id; cavity_id++) {
            size_left *= grid.cavity_max_size(cavity_id);
        }

        for (cavity_id = from_id + 1; cavity_id < to_id; cavity_id++) {
            size_middle *= grid.cavity_max_size(cavity_id);
        }

        for (cavity_id = to_id + 1; cavity_id < grid.cavities_count(); cavity_id++) {
            size_right *= grid.cavity_max_size(cavity_id);
        }

        H += tensor_multiply(tensor_multiply(E_Matrix<COMPLEX>(size_left), a_destroy(grid.max_N())),
             tensor_multiply(
             tensor_multiply(E_Matrix<COMPLEX>(size_middle), a_create(grid.max_N())), E_Matrix<COMPLEX>(size_right))) * gamma;

        H += tensor_multiply(tensor_multiply(E_Matrix<COMPLEX>(size_left), a_create(grid.max_N())),
             tensor_multiply(
             tensor_multiply(E_Matrix<COMPLEX>(size_middle), a_destroy(grid.max_N())), E_Matrix<COMPLEX>(size_right))) * std::conj(gamma);
    }
    */

        std::set<TCH_State> update_basis(const std::set<TCH_State>& basis, const std::set<TCH_State>& addition) {
            std::set<TCH_State> res;

            for (const auto& basis_state : basis) {
                for (const auto& state : addition) {
                    auto tmp = basis_state.add_state(state[0]);
                    res.insert(tmp);
                }
            }

            return res;
        }

        void next_permutation(std::vector<size_t>& v, size_t max_num) {
            if (v[v.size() - 1] == max_num) {
                v[v.size() - 1] = 0;
                v[0] = max_num;
            } else {
                bool is_next = false;
                for (size_t i = 0; i < v.size() - 1; i++) {
                    if (v[i] == max_num) {
                        is_next = true;
                        v[i] = 0;
                        v[0] = max_num - 1;
                        v[i + 1] = 1;
                        break;
                    }
                }

                if (!is_next) {
                    if (v[0] == 0) {
                        for (size_t i = 1; i < v.size(); i++) {
                            if (v[i] != 0) {
                                v[0] = v[i] - 1;
                                v[i + 1]++;
                                v[i] = 0;
                                break;
                            }
                        }
                    } else {
                        v[0]--;
                        v[1]++;
                    }
                }
            }
        }

        /*
    Cavity_State get_energy_state(size_t energy, size_t m) {
        std::vector<int> state_vec(m, 0);
        for (size_t i = 0; i < std::min(m, energy); i++) {
            state_vec[i] = 1;
        }

        return Cavity_State(std::max(long(0), long(energy) - long(m)), state_vec);
    }

    bool next_index(std::vector<size_t>& index_vec, const std::vector<std::set<Cavity_State>>& cavity_bases) {
        if (index_vec[0] + 1 == cavity_bases[0].size()) {
            index_vec[0] = 0;

            bool is_end = true;
            for (size_t i = 1; i < index_vec.size(); i++) {
                if (index_vec[i] + 1 < cavity_bases[i].size()) {
                    index_vec[i]++;
                    is_end = false;
                    break;
                }
            }

            return !is_end;
        } else {
            index_vec[0]++;
            return true;
        }
    }
    */
    }  // namespace

    // N >= 1
    /*
std::set<TCH_State> define_basis_of_hamiltonian(const TCH_State& grid) {
    std::set<TCH_State> basis;

    long max_energy;
    for (max_energy = grid.max_N(); max_energy >= long(grid.min_N()); max_energy--) {
        //std::cout << max_energy << " " << target_N << std::endl;
        TCH_State state = grid; // copy base structure of grid

        std::vector<size_t> energy_map(grid.cavities_count(), 0);
        energy_map[0] = max_energy;

        //std::cout << "MAP: " << max_energy << std::endl;
        //show_vector(energy_map);
        std::vector<std::set<Cavity_State>> cavity_bases(grid.cavities_count());
        std::vector<size_t> cavity_basis_index(grid.cavities_count(), 0);

        while(true) {
            for (size_t i = 0; i < grid.cavities_count(); i++) {
                //std::cout << energy_map[i] << " " << get_energy_state(energy_map[i], grid[i].m()).to_string() << std::endl;
                //cavity_bases[i] = State_Graph(get_energy_state(energy_map[i], grid[i].m()), false, false).get_basis();
                //show_basis(cavity_bases[i]);
            }

            do {
                for (size_t i = 0; i < grid.cavities_count(); i++) {
                    state.set_state(i, get_elem_from_set(cavity_bases[i], cavity_basis_index[i]));
                }
        
                basis.insert(state);
            } while(next_index(cavity_basis_index, cavity_bases));

            cavity_basis_index = std::vector<size_t>(grid.cavities_count(), 0);

            if (energy_map[energy_map.size() - 1] == max_energy) break;

            next_permutation(energy_map, max_energy);
            //std::cout << "MAP: " << max_energy << std::endl;
            //show_vector(energy_map);
        }
        //show_basis(basis);
    }

    return basis;
}
*/

    /*
void Hamiltonian::show(const size_t width) const {
    H_.show(width);
}


std::pair<std::vector<double>, Matrix<COMPLEX>> Hamiltonian::eigen() {
    if (is_eigen_) {
        return std::make_pair(eigenvalues_, eigenvectors_);
    }

    auto res = Hermit_Lanczos(H_);
    eigenvectors_ = res.second;
    eigenvalues_ = res.first;
    is_eigen_ = true;

    return std::make_pair(eigenvalues_, eigenvectors_);
}
*/

    // -------------------------------   H_by_func   -------------------------------

    /*
H_by_func::H_by_func(size_t n, std::function<COMPLEX(size_t, size_t)> func) : func_(func) {
    auto size = n;
    H_ = Matrix<COMPLEX>(DEFAULT_MATRIX_STYLE, size, size);
    for (size_t i = 0; i < this->size(); i++) {
        for (size_t j = 0; j < this->size(); j++) {
            H_[i][j] = func(i, j);
        }
    }
//#endif // ENABLE_MPI
}
*/

    // ---------------------------- H_TC ----------------------------

    /*
H_TC::H_TC(const State& init_state) {
    assert(init_state.cavities_count() == 1);
    H_TCH H(init_state);

    basis_ = H.get_basis();
    grid_ = init_state;
    H_ = H.get_matrix();

    auto size_m_ = std::pow(2, init_state.m(0));
    assert(init_state.cavities_count() == 1);
    State_Graph graph(init_state[0], (std::abs(init_state.get_leak_gamma(0)) >= config::eps) ? true : false, false);
    basis_ = Cavity_State_to_State(graph.get_basis());
    auto size_H_ = basis_.size();

    auto n_ = init_state.n(0);
    auto n = n_;
    auto m = init_state.m(0);

    std::vector<size_t> state_index;
    //std::unordered_map<size_t, size_t> state_to_index;
    //auto H_index = 0;
    for (const auto& state: basis_) {
        auto index = state.get_index();
        state_index.emplace_back(index);
        //state_to_index[index] = H_index++;
    }

    H_ = Matrix<COMPLEX>(size_H_, size_H_, 0);
    init_state_ = init_state;

    for (size_t i = 0; i < size_H_; i++) {
        H_[i][i] = 1;
    }

    Matrix<COMPLEX> eye_photons(n_ + 1, n_ + 1, 0);
    Matrix<COMPLEX> eye_atoms(size_m_, size_m_, 0);

    for (int i = 0; i < n_ + 1; i++) {
        eye_photons[i][i] = 1;
    }

    for (int i = 0; i < size_m_; i++) {
        eye_atoms[i][i] = 1;
    }

    auto tmp_H_ = tensor_multiply(E_photons(n + 1), eye_atoms);

    for (int k = 0; k < m; k++) {
        auto size_left = std::pow(2, k);
        auto size_right = std::pow(2, m - k - 1);

        Matrix<COMPLEX> eye_left = E_Matrix<COMPLEX>(size_left);
        Matrix<COMPLEX> eye_right = E_Matrix<COMPLEX>(size_right);
        //Matrix<COMPLEX> eye_left(size_left, size_left, 0);
        /Matrix<COMPLEX> eye_right(size_right, size_right, 0);

        //for (int i = 0; i < size_left; i++) eye_left[i][i] = 1;
        //for (int i = 0; i < size_right; i++) eye_right[i][i] = 1;
        Matrix<COMPLEX> H_sigma = tensor_multiply(eye_left, sigma_energy);
        //std::cout << "LEFT: " << std::endl;
        //show_matrix(H_sigma);
        H_sigma = tensor_multiply(H_sigma, eye_right);
        //std::cout << "RIGHT: " << std::endl;
        //show_matrix(H_sigma);
        H_sigma = tensor_multiply(eye_photons, H_sigma);
        //std::cout << "PHOTONS: " << std::endl;
        //show_matrix(H_sigma);
        tmp_H_ += H_sigma * std::complex<double>(config::h * config::w);
        //std::cout << k << std::endl;
        //show_matrix(H_);
        //std::cout << std::endl;
    }

    Matrix<COMPLEX> sum_sigma_down_instance = sum_sigma_down(m);
    Matrix<COMPLEX> sum_sigma_up_instance = sum_sigma_down_instance.transpose();
    Matrix<COMPLEX> a = a_destroy(n);
    Matrix<COMPLEX> a_creation = a.transpose();

    auto tmp_left = tensor_multiply(a, sum_sigma_up_instance);
    auto tmp_right = tensor_multiply(a_creation, sum_sigma_down_instance);

    tmp_H_ += tmp_left * g(0);
    tmp_H_ += tmp_right * g(0);

    for (size_t i = 0; i < size_H_; i++) {
        for (size_t j = 0; j < size_H_; j++) {
            H_[i][j] = tmp_H_[state_index[i]][state_index[j]];
        }
    }
}
*/

    // ---------------------------- H_JC ----------------------------

    /*
H_JC::H_JC(const State& init_state) {
    assert(init_state.cavities_count() == 1 and init_state.m(0) == 1);
    H_TC tmp_H(init_state);

    H_ = tmp_H.get_matrix();
    grid_ = init_state;
    basis_ = tmp_H.get_basis();
}
*/

    /*
H_JC::H_JC(const TCH_State& grid) {
    assert(grid.cavities_count() == 1 and grid.m(0) == 1);
    grid_ = grid;

    auto x_size = grid.x_size();
    auto y_size = grid.y_size();
    auto z_size = grid.z_size();

    auto basis = State_Graph<TCH_State>(grid).get_basis();
    
    basis_ = convert_to(basis);

    size_t size = basis_.size();
    H_ = Matrix<COMPLEX>(DEFAULT_MATRIX_STYLE, size, size, 0);
    size_t i = 0, j = 0;
    for (const auto& state_from: basis_) {
        for (const auto& state_to: basis_) {
            H_[i][j] += JC_ADD(TCH_State(state_from), TCH_State(state_to), grid);
            j++;
        }
        j = 0;
        i++;
    }
}

void H_JC::make_exact() {
    size_t i = 0, j = 0;
    for (const auto& state_from: basis_) {
        for (const auto& state_to: basis_) {
            H_[i][j] += JC_addition(TCH_State(state_from), TCH_State(state_to));
            j++;
        }
        j = 0;
        i++;
    }
}

H_TC::H_TC(const TCH_State& grid) {
    assert(grid.cavities_count() == 1);
    grid_ = grid;

    auto x_size = grid.x_size();
    auto y_size = grid.y_size();
    auto z_size = grid.z_size();

    //basis_ = define_basis_of_hamiltonian(grid);
    auto basis = State_Graph<TCH_State>(grid).get_basis();
    basis_ = convert_to(basis);

    size_t size = basis_.size();
    H_ = Matrix<COMPLEX>(DEFAULT_MATRIX_STYLE, size, size, 0);

    //std::cout << H_.is_c_style() << " " << H_.n() << " " << H_.m() << std::endl;
    size_t i = 0, j = 0;
    for (const auto& state_from: basis_) {
        for (const auto& state_to: basis_) {
            H_[i][j] += TC_ADD(state_from, state_to, grid);
            j++;
        }
        j = 0;
        i++;
    }
//#endif
}

// --------------------------- H_TCH ------------------------------------

H_TCH::H_TCH(const TCH_State& grid) {
    grid_ = grid;

    auto x_size = grid.x_size();
    auto y_size = grid.y_size();
    auto z_size = grid.z_size();

    auto basis = State_Graph<TCH_State>(grid).get_basis();
    basis_ = convert_to(basis);

    size_t size = basis_.size();
    H_ = Matrix<COMPLEX>(DEFAULT_MATRIX_STYLE, size, size, 0);

    //std::cout << H_.is_c_style() << " " << H_.n() << " " << H_.m() << std::endl;
    size_t i = 0, j = 0;
    for (const auto& state_from: basis_) {
        for (const auto& state_to: basis_) {
            H_[i][j] += TCH_ADD(state_from, state_to, grid);
            j++;
        }
        j = 0;
        i++;
    }
//#endif
}
*/

    namespace {
        Operator<TCH_State> H_TCH_OP() {
            using OpType = Operator<TCH_State>;

            OpType my_H;
            my_H = my_H + OpType(photons_count) + OpType(atoms_exc_count) + OpType(exc_relax_atoms) +
                   OpType(photons_transfer);

            return my_H;
        }

        std::vector<std::pair<double, Operator<TCH_State>>> decs(
            const State<TCH_State>& state,
            std::vector<std::pair<double, Operator<TCH_State>>> dec) {
            using OpType = Operator<TCH_State>;

            auto st = *(state.get_state_components().begin());

            for (size_t i = 0; i < st->cavities_count(); i++) {
                if (!is_zero(st->get_leak_gamma(i))) {
                    std::function<State<TCH_State>(const TCH_State&)> a_destroy_i = {[i](const TCH_State& che_state) {
                        return set_qudit(che_state, che_state.n(i) - 1, 0, i) * std::sqrt(che_state.n(i));
                    }};

                    OpType my_A_out(a_destroy_i);

                    dec.emplace_back(std::make_pair(st->get_leak_gamma(i), my_A_out));
                }

                if (!is_zero(st->get_gain_gamma(i))) {
                    std::function<State<TCH_State>(const TCH_State&)> a_create_i = {[i](const TCH_State& che_state) {
                        return set_qudit(che_state, che_state.n(i) + 1, 0, i) * std::sqrt(che_state.n(i) + 1);
                    }};

                    OpType my_A_in(a_create_i);

                    dec.emplace_back(std::make_pair(st->get_gain_gamma(i), my_A_in));
                }
            }

            return dec;
        }
    }  // namespace

    H_TCH::H_TCH(const State<TCH_State>& state, const std::vector<std::pair<double, Operator<TCH_State>>>& dec)
        : H_by_Operator<TCH_State>(state, H_TCH_OP(), decs(state, dec)) {}

}  // namespace QComputations

#pragma once
#include <complex>
#include "state.hpp"
#include "config.hpp"
#include <functional>
#include "blocked_matrix.hpp"

namespace QComputations {

namespace {
#ifdef MKL_ILP64
    using ILP_TYPE = long long;
#else
    using ILP_TYPE = int;
#endif

    template<typename StateType>
        using OperatorType = std::function<State<StateType>(const StateType& state)>;

    using ValType = int;
    using COMPLEX = std::complex<double>;
}

template<typename StateType>
class Operator {
    public:
        explicit Operator(): cur_id_(-1) {}
        explicit Operator(OperatorType<StateType> op): cur_id_(0), operators_(1, std::vector<OperatorType<StateType>>(1, op)) {}

        /*
        void operator+(OperatorType<StateType> other) {
            operators_.emplace_back(1, other);
            cur_id_++;
        }

        void operator*(OperatorType<StateType> other) {
            operators_[cur_id_].push_back(other);
        }
        */

        Operator<StateType> operator+(const Operator<StateType>& other) const {
            auto res = *this;

            res.cur_id_ = this->operators_.size() + other.operators_.size();

            for (size_t i = 0; i < other.operators_.size(); i++) {
                res.operators_.push_back(other.operators_[i]);
            }

            return res;
        }

        Operator<StateType> operator*(const Operator<StateType>& other) const {
            Operator<StateType> res(*this);
            assert(other.operators_.size() <= 1);

            for (const auto& op: other.operators_) {
                for (const auto& cur_op: op) {
                    res.operators_[res.cur_id_].push_back(cur_op);
                }
            }
            return res;
        }

        Operator<StateType> operator*(const COMPLEX& num) {
            OperatorType<StateType> func = {[num](const StateType& state) {
                return State<StateType>(state) * num;
            }};

            return (*this) * Operator<StateType>(func);
        }


        State<StateType> run(const State<StateType>& init_state) const;
    private:
        int cur_id_; // 
        std::vector<std::vector<OperatorType<StateType>>> operators_; // Сам оператор
};

template<typename StateType>
State<StateType> Operator<StateType>::run(const State<StateType>& init_state) const {
    State<StateType> states_ = init_state;

    for (size_t i = 0; i < states_.size(); ++i) {
        states_[i] = 0;
    }

    //std::cout << states_.to_string() << std::endl;

    for (const auto& op: operators_) {
        for (const auto& cur_state: init_state.get_state_components())  {
            State<StateType> new_state;

            //std::cout << cur_state.to_string() << std::endl;
            auto res = op[op.size() - 1](cur_state);
            //std::cout << res.to_string() << std::endl;
            new_state = res;

            for (int i = op.size() - 2; i >= 0; i--) {
                for (const auto& cur_new_state: res.get_state_components()) {
                    new_state += op[i](cur_new_state); 
                }

                res = new_state;
            }

            for (const auto& st: new_state.get_state_components()) {
                states_.insert(st);
                states_[states_.get_index(st)] += new_state[new_state.get_index(st)];
            }

            //std::cout << states_.to_string() << std::endl;
        }
    }

    return states_;
}

template<typename StateType>
Matrix<COMPLEX> operator_to_matrix(const Operator<StateType>& op, const std::set<StateType>& basis) {
    size_t dim = basis.size();

    std::function<COMPLEX(size_t i, size_t j)> func = {
        [&basis, &op](size_t i, size_t j) {
            auto state_from = get_elem_from_set(basis, j);
            auto state_to = get_elem_from_set(basis, i);
            auto res_state = op.run(State<StateType>(state_from));
            
            COMPLEX res = COMPLEX(0, 0);

            size_t index = 0;
            for (const auto& state: res_state.get_state_components()) {
                if (state == state_to) {
                    res = res_state[index];
                    break;
                }

                index++;
            }

            return res;
        }
    };

    Matrix<COMPLEX> A(C_STYLE, dim, dim, func);

    return A;
}

template<typename StateType>
BLOCKED_Matrix<COMPLEX> operator_to_matrix(ILP_TYPE ctxt, const Operator<StateType>& op, const std::set<StateType>& basis) {
    size_t dim = basis.size();

    std::function<COMPLEX(size_t i, size_t j)> func = {
        [&basis, &op](size_t i, size_t j) {
            auto state_from = get_elem_from_set(basis, j);
            auto state_to = get_elem_from_set(basis, i);
            auto res_state = op.run(State<StateType>(state_from));
            
            COMPLEX res = COMPLEX(0, 0);

            size_t index = 0;
            for (const auto& state: res_state.get_state_components()) {
                if (state == state_to) {
                    res = res_state[index];
                    break;
                }

                index++;
            }

            return res;
        }
    };

    BLOCKED_Matrix<COMPLEX> A(ctxt, GE, dim, dim, func);

    return A;
}

/*
template<typename StateType>
class Formule: public Operator<StateType> {
    public:
        //explicit Formule(const Basis_State& state) {states_.insert(state);}
        explicit Formule(const Operator<StateType>& operator);

        std::set<Basis_State> get_states() const { return states_; }
    private:
        std::set<Basis_State> states_;
};

template<typename StateType>
void Formule<StateType>::make_work_area() {

}
*/

// ---------------------------- OPERATORS ---------------------------

template<typename StateType>
State<StateType> set_qudit(const StateType& state, ValType val, size_t qudit_index, size_t group_id = 0, const std::string& info = "") {
    auto res = state;
    if (val > state.get_max_val(qudit_index, group_id) or val < 0) {
        res.clear();
    } else {
        res.set_qudit(val, qudit_index, group_id);
    }

    return State<StateType>(res);
}

template<typename StateType>
State<StateType> sigma_x(const StateType& state, size_t qudit_index, size_t group_id = 0, const std::string& info = "") {
    StateType res = state;
    auto qudit = state.get_qudit(qudit_index, group_id);
    assert(qudit == 0 or qudit == 1);

    if (qudit == 0) qudit = 1;
    else qudit = 0;

    res.set_qudit(qudit, qudit_index, group_id);

    return State<StateType>(res);
}

template<typename StateType>
State<StateType> check(const StateType& state, ValType check_val, size_t qudit_index, size_t group_id = 0, const std::string& info = "") {
    auto res = state;
    if (res.get_qudit(qudit_index, group_id) != check_val) {
        res.clear();
    }

    return State<StateType>(res);
}

template<typename StateType>
State<StateType> get_qudit(const StateType& state, size_t qudit_index, size_t group_id = 0, const std::string& info = "") {
    auto res = State<StateType>(state);
    res[0] = state.get_qudit(qudit_index, group_id);

    return res;
}

State<TCH_State> photons_transfer(const TCH_State& state);
State<TCH_State> photons_count(const TCH_State& state);
State<TCH_State> atoms_exc_count(const TCH_State& state);
State<TCH_State> exc_relax_atoms(const TCH_State& state);

/*
State<TCH_State> a_destroy(const TCH_State& state);
State<TCH_State> a_create(const TCH_State& state);
State<TCH_State> sigma_destroy(const TCH_State& state);
State<TCH_State> sigma_create(const TCH_State& state);
State<TCH_State> photons_count(const TCH_State& state);
State<TCH_State> atoms_exc_count(const TCH_State& state);
*/

// ------------------------------ СТАРЫЕ ВЕРСИИ. БУДЕТ УДАЛЕНО ---------------------------
/*
COMPLEX self_energy_photon(const TCH_State& state_from, const TCH_State& state_to, COMPLEX h = QConfig::instance().h());
COMPLEX self_energy_atom(const TCH_State& state_from, const TCH_State& state_to, COMPLEX h = QConfig::instance().h());
COMPLEX excitation_atom(const TCH_State& state_from, const TCH_State& state_to, COMPLEX g = QConfig::instance().g());
COMPLEX de_excitation_atom(const TCH_State& state_from, const TCH_State& state_to, COMPLEX g = QConfig::instance().g());
COMPLEX photon_exchange(const TCH_State& state_from, const TCH_State& state_to, const TCH_State& grid);
COMPLEX photon_destroy(const TCH_State& state_from, const TCH_State& state_to, COMPLEX gamma = COMPLEX(1));
COMPLEX photon_create(const TCH_State& state_from, const TCH_State& state_to, COMPLEX gamma = COMPLEX(1));

COMPLEX JC_addition(const TCH_State& state_from, const TCH_State& state_to, COMPLEX g = QConfig::instance().g());
COMPLEX TCH_ADD(const TCH_State& state_from, const TCH_State& state_to, const TCH_State& grid);
COMPLEX TC_ADD(const TCH_State& state_from, const TCH_State& state_to, const TCH_State& grid);
COMPLEX JC_ADD(const TCH_State& state_from, const TCH_State& state_to, const TCH_State& grid);
*/

} // namespace QComputations
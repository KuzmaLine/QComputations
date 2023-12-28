#pragma once
#include <complex>
#include "state.hpp"
#include "config.hpp"
#include <functional>

namespace QComputations {

namespace {
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

        void operator+(OperatorType<StateType> other) {
            operators_.emplace_back(1, other);
            cur_id_++;
        }

        void operator*(OperatorType<StateType> other) {
            operators_[cur_id_].push_back(other);
        }

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


        State<StateType> run(const State<StateType>& init_state) const;
    private:
        int cur_id_;
        std::vector<std::vector<OperatorType<StateType>>> operators_;
};

template<typename StateType>
State<StateType> Operator<StateType>::run(const State<StateType>& init_state) const {
    State<StateType> states_ = init_state;
    for (size_t i = 0; i < states_.size(); ++i) {
        states_[i] = 0;
    }

    std::cout << states_.to_string() << std::endl;

    State<StateType> new_state;

    for (const auto& op: operators_) {
        for (const auto& cur_state: init_state.get_state_components())  {
            auto res = op[op.size() - 1](cur_state);

            new_state = res;

            for (int i = op.size() - 2; i >= 0; i--) {
                State<StateType> tmp_state;

                for (const auto& cur_new_state: new_state.get_state_components()) {
                    res = op[i](cur_new_state);
                    tmp_state = res;
                }

                new_state = tmp_state;
            }

            for (const auto& st: new_state.get_state_components()) {
                states_.insert(st);
                states_[states_.get_index(st)] = new_state[new_state.get_index(st)];
            }

            std::cout << states_.to_string() << std::endl;
        }
    }

    return states_;
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
State<StateType> set_qudit(const StateType& state, size_t qudit_index, ValType val) {
    auto res = state;
    if (val > state.get_max_val(qudit_index) or val < 0) {
        res.clear();
    }

    return State<StateType>(res);
}

template<typename StateType>
State<StateType> sigma_x(const StateType& state, size_t qudit_index) {
    StateType res = state;
    auto qudit = state.get_qudit(qudit_index);
    assert(qudit == 0 or qudit == 1);

    if (qudit == 0) qudit = 1;
    else qudit = 0;

    res.set_qudit(qudit, qudit_index);

    return State<StateType>(res);
}

template<typename StateType>
State<StateType> check(const StateType& state, ValType check_val, size_t qudit_index) {
    auto res = state;
    if (res.get_qudit(qudit_index) != check_val) {
        res.clear();
    }

    return State<StateType>(res);
}

State<CHE_State> a_destroy_qudit(const CHE_State& state, size_t photon_index, size_t cavity_id = 0);
State<CHE_State> a_create_qudit(const CHE_State& state, size_t photon_index, size_t cavity_id = 0);
State<CHE_State> sigma_destroy_qudit(const CHE_State& state, size_t atom_index, size_t cavity_id = 0);
State<CHE_State> sigma_create_qudit(const CHE_State& state, size_t atom_index, size_t cavity_id = 0);
State<CHE_State> photons_count_qudit(const CHE_State& state, size_t photon_index, size_t cavity_id = 0);
State<CHE_State> atoms_exc_count_qudit(const CHE_State& state, size_t atom_index, size_t cavity_id = 0);
State<CHE_State> check_qudit(const CHE_State& state, ValType check_val, size_t qudit_index, size_t group_id = 0);

State<CHE_State> a_destroy(const CHE_State& state);
State<CHE_State> a_create(const CHE_State& state);
State<CHE_State> sigma_destroy(const CHE_State& state);
State<CHE_State> sigma_create(const CHE_State& state);
State<CHE_State> photons_count(const CHE_State& state);
State<CHE_State> atoms_exc_count(const CHE_State& state);

// ------------------------------ СТАРЫЕ ВЕРСИИ. БУДЕТ УДАЛЕНО ---------------------------
COMPLEX self_energy_photon(const CHE_State& state_from, const CHE_State& state_to, COMPLEX h = QConfig::instance().h());
COMPLEX self_energy_atom(const CHE_State& state_from, const CHE_State& state_to, COMPLEX h = QConfig::instance().h());
COMPLEX excitation_atom(const CHE_State& state_from, const CHE_State& state_to, COMPLEX g = QConfig::instance().g());
COMPLEX de_excitation_atom(const CHE_State& state_from, const CHE_State& state_to, COMPLEX g = QConfig::instance().g());
COMPLEX photon_exchange(const CHE_State& state_from, const CHE_State& state_to, const CHE_State& grid);
COMPLEX photon_destroy(const CHE_State& state_from, const CHE_State& state_to, COMPLEX gamma = COMPLEX(1));
COMPLEX photon_create(const CHE_State& state_from, const CHE_State& state_to, COMPLEX gamma = COMPLEX(1));

COMPLEX JC_addition(const CHE_State& state_from, const CHE_State& state_to, COMPLEX g = QConfig::instance().g());
COMPLEX TCH_ADD(const CHE_State& state_from, const CHE_State& state_to, const CHE_State& grid);
COMPLEX TC_ADD(const CHE_State& state_from, const CHE_State& state_to, const CHE_State& grid);
COMPLEX JC_ADD(const CHE_State& state_from, const CHE_State& state_to, const CHE_State& grid);


} // namespace QComputations
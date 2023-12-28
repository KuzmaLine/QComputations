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
        explicit Operator() = default;
        explicit Operator(OperatorType<StateType> op): cur_id(0), operators_(1, std::vector<OperatorType<StateType>>(1, op)) {}

        void operator+(OperatorType<StateType> other) {
            operators_.emplace_back(1, other);
            cur_id++;
        }
        void operator*(OperatorType<StateType> other) {
            operators_[cur_id].push_back(other);
        }

        std::set<StateType> run(const State<StateType>& init_state) const;
    private:
        int cur_id;
        std::vector<std::vector<OperatorType<StateType>>> operators_;
};

template<typename StateType>
std::set<StateType> Operator<StateType>::run(const State<StateType>& init_state) const {
    std::set<StateType> states_;
    states_ = init_state.get_state_components();
    std::set<StateType> new_states;

    for (const auto& op: operators_) {
        for (const auto& cur_state: init_state.get_state_components())  {
            auto res = op[op.size() - 1](cur_state);

            new_states = res.get_state_components();

            for (int i = op.size() - 2; i >= 0; i--) {
                std::set<StateType> tmp_states;

                for (const auto& cur_new_state: new_states) {
                    res = op[i](cur_new_state);
                    for (const auto& st: res.get_state_components()) {
                        tmp_states.insert(st);
                    }
                }

                new_states = tmp_states;
            }

            for (const auto& state: new_states) {
                states_.insert(state);
            }
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
#pragma once
#include <complex>
#include <functional>
#include <stack>

#include "blocked_matrix.hpp"
#include "config.hpp"
#include "state.hpp"

namespace QComputations {

namespace {
template <typename T>
void _printName(std::function<State<T>(const T&)> func, const std::string& funcName) {
    std::cout << funcName << std::endl;
}
#define printName(f) _printName(f, #f)

#ifdef MKL_ILP64
using ILP_TYPE = long long;
#else
using ILP_TYPE = int;
#endif

template <typename StateType>
using OperatorT = std::function<State<StateType>(const StateType& state)>;

using ValType = int;
using COMPLEX = std::complex<double>;
}  // namespace

/*

Переделать весь класс операторов в дерево.
Ход влево - умножение
Ход вправо - сложение

*/

template <typename StateType>
class Operator {
    using OperatorType = OperatorT<StateType>;

   public:
    explicit Operator() = default;
    Operator(State<StateType> (*op)(const StateType&)) { root_ = new OperatorNode(op); }
    Operator(OperatorType op) { root_ = new OperatorNode(op); }

    Operator<StateType> operator+(Operator<StateType> other) {
        if (other.root_ == NULL) return (*this);
        if (this->root_ == NULL) return other;

        OperatorNode* cur_node = this->root_;
        while (cur_node->right != NULL) {
            cur_node = cur_node->right;
        }

        if (this->root_ != NULL) {
            other.root_->insert_up(cur_node);
            cur_node->right = other.root_;
        }

        return (*this);
    }

    Operator<StateType> operator*(Operator<StateType> other) {
        if (other.root_ == NULL) return (*this);
        if (this->root_ == NULL) assert(false);  // Пустой оператор умножить на не пустой?

        std::stack<OperatorNode*> st;
        OperatorNode* cur_op = this->root_;

        while (cur_op != NULL or !st.empty()) {
            while (cur_op != NULL) {
                st.push(cur_op);
                cur_op = cur_op->left;
            }

            if (st.top()->left == NULL) {
                other.root_->insert_up(st.top());
                st.top()->left = other.root_;
            }

            cur_op = st.top()->right;
            st.pop();
        }

        return (*this);
    }

    Operator<StateType> operator*(const COMPLEX& num) {
        if (this->root_ == NULL) assert(false);  // Пустой оператор умножить на число?

        OperatorType func = {[num](const StateType& state) { return State<StateType>(state) * num; }};

        return Operator<StateType>(func) * (*this);
    }

    State<StateType> run(const State<StateType>& init_state) const;
    void show() const;

   private:
    void refresh_tree() const;

    struct OperatorNode {
        OperatorNode(OperatorType op) : func_(op) {}
        OperatorNode(State<StateType> (*op)(const StateType&)) : func_(op) {}

        OperatorNode* up() {
            if (up_.size() == 0) return this;
            which_way_++;
            return up_[which_way_ % up_.size()];
        }

        void insert_up(OperatorNode* up) { up_.emplace_back(up); }

        State<StateType> invoke(const State<StateType>& st) const;

        OperatorType func_;
        std::vector<OperatorNode*> up_;
        OperatorNode* left = NULL;   // умножение
        OperatorNode* right = NULL;  // Сложение

        mutable State<StateType> cur_res_;
        mutable int from_tree_ = 0;   // 0 - не спускался
                                      // 1 - пройден левый путь
                                      // 2 - пройден правый путь
                                      // 3 - пройдены обе ветви
        mutable int which_way_ = -1;  // По какому пути нужно пройти вверх
    };

    OperatorNode* root_ = NULL;  // Корневой оператор
};

template <typename StateType>
State<StateType> Operator<StateType>::OperatorNode::invoke(const State<StateType>& st) const {
    State<StateType> res;

    for (auto cur_state : st.get_state_components()) {
        res += this->func_(*cur_state) * st[*cur_state];
    }

    return res;
}

template <typename StateType>
void Operator<StateType>::refresh_tree() const {
    std::stack<OperatorNode*> st;
    OperatorNode* cur_op = this->root_;

    while (cur_op != NULL or !st.empty()) {
        while (cur_op != NULL) {
            cur_op->from_tree_ = 0;
            cur_op->which_way_ = -1;
            cur_op->cur_res_.clear();

            st.push(cur_op);
            cur_op = cur_op->left;
        }

        cur_op = st.top()->right;
        st.pop();
    }
}

template <typename StateType>
void Operator<StateType>::show() const {
    std::stack<OperatorNode*> st;
    OperatorNode* cur_op = this->root_;

    while (cur_op != NULL or !st.empty()) {
        while (cur_op != NULL) {
            st.push(cur_op);
            cur_op = cur_op->left;
        }

        printName(cur_op->func_);
        cur_op = st.top()->right;
        st.pop();
    }
}

template <typename StateType>
State<StateType> Operator<StateType>::run(const State<StateType>& init_state) const {
    OperatorNode* cur_op = this->root_;

    while (this->root_->from_tree_ != 3) {
        if (cur_op->from_tree_ == 0 and cur_op->left != NULL) {
            cur_op->from_tree_ = 1;
            cur_op = cur_op->left;

        } else if (cur_op->from_tree_ == 0 and cur_op->left == NULL) {
            cur_op->from_tree_ = 2;
            cur_op->cur_res_ = cur_op->invoke(init_state);

            if (cur_op->right == NULL) {
                cur_op->from_tree_ = 3;
                cur_op = cur_op->up();
            } else {
                cur_op = cur_op->right;
            }
        } else if (cur_op->from_tree_ == 1 and cur_op->right != NULL) {
            cur_op->from_tree_ = 2;
            cur_op->cur_res_ = cur_op->invoke(cur_op->left->cur_res_);
            cur_op = cur_op->right;
        } else if (cur_op->from_tree_ == 1 and cur_op->right == NULL) {
            cur_op->from_tree_ = 3;
            cur_op->cur_res_ = cur_op->invoke(cur_op->left->cur_res_);
            cur_op = cur_op->up();
        } else if (cur_op->from_tree_ == 2) {
            cur_op->from_tree_ = 3;
            cur_op->cur_res_ += cur_op->right->cur_res_;
        } else if (cur_op->from_tree_ == 3) {
            cur_op = cur_op->up();
        }
    }

    auto res = root_->cur_res_.copy();
    this->refresh_tree();

    return res;
}

template <typename StateType>
Matrix<COMPLEX> operator_to_matrix(const Operator<StateType>& op, const BasisType<StateType>& basis) {
    size_t dim = basis.size();
    Matrix<COMPLEX> A(C_STYLE, dim, dim, COMPLEX(0, 0));

    size_t col_state = 0;
    for (auto state : basis) {
        auto res_state = op.run(State<StateType>(*state));

        size_t index = 0;
        for (auto state_res : res_state.state_components()) {
            A[get_index_state_in_basis(*state_res, basis)][col_state] = res_state[index++];
        }

        col_state++;
    }

    return A;
}

#ifdef ENABLE_MPI
#ifdef ENABLE_CLUSTER
template <typename StateType>
BLOCKED_Matrix<COMPLEX> operator_to_matrix(ILP_TYPE ctxt,
                                           const Operator<StateType>& op,
                                           const BasisType<StateType>& basis) {
    size_t dim = basis.size();

    BLOCKED_Matrix<COMPLEX> A(ctxt, GE, dim, dim, COMPLEX(0, 0), 0, 0);

    for (size_t j = 0; j < A.local_m(); j++) {
        size_t global_state_index = A.get_global_col(j);
        auto state_from = get_state_from_basis<StateType>(basis, global_state_index);
        auto res_state = op.run(State<StateType>(*state_from));

        size_t index = 0;
        for (auto state : res_state.state_components()) {
            auto cur_global_row = get_index_state_in_basis(*state, basis);
            if (A.is_my_elem_row(cur_global_row)) {
                A(A.get_local_row(cur_global_row), j) = res_state[index];
            }

            index++;
        }
    }

    return A;
}
#endif
#endif

// ---------------------------- OPERATORS ---------------------------

template <typename StateType>
State<StateType> set_qudit(const StateType& state,
                           ValType val,
                           size_t qudit_index = 0,
                           size_t group_id = 0,
                           const std::string& info = "") {
    auto res = state;
    if (val > state.get_max_val(qudit_index, group_id) or val < 0) {
        res.clear();
    } else {
        res.set_qudit(val, qudit_index, group_id);
    }

    return State<StateType>(res);
}

template <typename StateType>
State<StateType> get_qudit(const StateType& state,
                           size_t qudit_index = 0,
                           size_t group_id = 0,
                           const std::string& info = "") {
    auto res = State<StateType>(state);
    res[0] = state.get_qudit(qudit_index, group_id);

    return res;
}

template <typename StateType>
State<StateType> sigma_x(const StateType& state,
                         size_t qudit_index = 0,
                         size_t group_id = 0,
                         const std::string& info = "") {
    StateType res = state;
    auto qudit = state.get_qudit(qudit_index, group_id);
    assert(qudit == 0 or qudit == 1);

    if (qudit == 0)
        qudit = 1;
    else
        qudit = 0;

    res.set_qudit(qudit, qudit_index, group_id);

    return State<StateType>(res);
}

template <typename StateType>
State<StateType> sigma_y(const StateType& state,
                         size_t qudit_index = 0,
                         size_t group_id = 0,
                         const std::string& info = "") {
    StateType res = state;
    auto qudit = state.get_qudit(qudit_index, group_id);
    assert(qudit == 0 or qudit == 1);

    if (qudit == 0) {
        qudit = 1;
    } else {
        qudit = 0;
    }

    res.set_qudit(qudit, qudit_index, group_id);

    State<StateType> stateres(res);
    stateres[0] *= COMPLEX(0, std::pow(-1, qudit + 1));

    return stateres;
}

template <typename StateType>
State<StateType> sigma_z(const StateType& state,
                         size_t qudit_index = 0,
                         size_t group_id = 0,
                         const std::string& info = "") {
    auto res = get_qudit(state, qudit_index, group_id);
    auto qudit = state.get_qudit(qudit_index, group_id);
    assert(qudit == 0 or qudit == 1);

    if (qudit == 1) {
        res[0] *= -1;
    }

    return res;
}

template <typename StateType>
State<StateType> check(const StateType& state,
                       ValType check_val,
                       size_t qudit_index = 0,
                       size_t group_id = 0,
                       const std::string& info = "") {
    auto res = state;
    if (res.get_qudit(qudit_index, group_id) != check_val) {
        res.clear();
    }

    return State<StateType>(res);
}

template <typename StateType>
State<StateType> check_func(const StateType& state, const std::function<bool(const StateType&)>& func) {
    auto res = state;
    if (func(state)) {
        res.clear();
    }

    return State<StateType>(res);
}

State<TCH_State> photons_transfer(const TCH_State& state);
State<TCH_State> photons_count(const TCH_State& state);
State<TCH_State> atoms_exc_count(const TCH_State& state);
State<TCH_State> exc_relax_atoms(const TCH_State& state);

}  // namespace QComputations
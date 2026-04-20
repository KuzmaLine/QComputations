#pragma once
#include "wave_func_state.hpp"
#include <complex>
#include "config.hpp"
#include <functional>
#include "blocked_matrix.hpp"
#include <stack>
#include "csr_matrix.hpp"

namespace QComputations {

namespace {
enum FUNC_TYPE { INTUITIVE_TYPE, MEM_EFFECTIVE_TYPE };

template<typename T>
void _printName(std::function<State<T>(const T&)> func, const std::string& funcName){
    std::cout << funcName << std::endl;
}
#define printName(f) _printName(f, #f)

#ifdef MKL_ILP64
    using ILP_TYPE = long long;
#else
    using ILP_TYPE = int;
#endif

    template<typename StateType>
        using OperatorT = std::function<State<StateType>(const std::shared_ptr<StateType>& state)>;
    template<typename StateType>
        using OperatorMemT = std::function<void(const StateType& state, State<StateType>&)>;
    //template<typename StateType>
        //using OperatorT = std::function<State<StateType>(const State<StateType>& state)>;      

    using ValType = int;
    using COMPLEX = std::complex<double>;
}

/*

Переделать весь класс операторов в дерево.
Ход влево - умножение
Ход вправо - сложение

*/

template<typename StateType>
class Operator {
    using OperatorType = OperatorT<StateType>;
    using OperatorMemType = OperatorMemT<StateType>;

    public:
        explicit Operator() = default;
        Operator(State<StateType>(*op)(const std::shared_ptr<StateType>&)) { root_ = new OperatorNode(OperatorType(op));}
        Operator(void(*op)(const StateType&, State<StateType>&)) { root_ = new OperatorNode(OperatorMemType(op));}
        Operator(OperatorType op) { root_ = new OperatorNode(op);}
        Operator(OperatorMemType op) { root_ = new OperatorNode(op);}
        // TO QCONFIG
        Operator(int num) { root_ = new OperatorNode(num, MEM_EFFECTIVE_TYPE);}

        Operator<StateType> operator+(Operator<StateType> other) {
            if (other.root_ == NULL) return (*this);
            if (this->root_ == NULL) return other;

            OperatorNode* cur_node = this->root_;
            while(cur_node->right != NULL) {
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
            if (this->root_ == NULL) assert(false); // Пустой оператор умножить на не пустой?

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
            if (this->root_ == NULL) assert(false); // Пустой оператор умножить на число?

            /*
            OperatorType func = {[num](const StateType& state) {
                return State<StateType>(state) * num;
            }};
            */

            OperatorMemType func = {[num](const StateType& state, State<StateType>& res) {
                res *= num;
            }};

            return  Operator<StateType>(func) * (*this);
        }

        State<StateType> run(const State<StateType>& init_state, const BasisType<StateType>& = {}) const;
        void show() const;
    private:
        void refresh_tree() const;

        struct OperatorNode {
            OperatorNode(OperatorType op): func_(op), func_type_(INTUITIVE_TYPE) {}
            OperatorNode(OperatorMemType op): effective_func_(op), func_type_(MEM_EFFECTIVE_TYPE) {}
            OperatorNode(int num, FUNC_TYPE func_type): func_type_(func_type) {
                if (func_type_ == INTUITIVE_TYPE) { 
                    func_ = {[num](const StateType& state) {
                        return State<StateType>(state) * num;
                    }};
                } else if (num == MEM_EFFECTIVE_TYPE) {
                    effective_func_ = {[num](const StateType& state, State<StateType>& res) {
                        res *= num;
                    }};
                }
            }

            OperatorNode* up() { 
                if (up_.size() == 0) return this;
                which_way_++;
                return up_[which_way_ % up_.size()];
            }

            void insert_up(OperatorNode* up) {
                up_.emplace_back(up);
            }

            void invoke(const State<StateType>& st, State<StateType>& global_state) const;

            OperatorType func_;
            OperatorMemType effective_func_;
            FUNC_TYPE func_type_;
            std::vector<OperatorNode*> up_;
            OperatorNode* left = NULL; // умножение
            OperatorNode* right = NULL; // Сложение

            mutable State<StateType> cur_res_;
            mutable int from_tree_ = 0; // 0 - не спускался
                                // 1 - пройден левый путь
                                // 2 - пройден правый путь
                                // 3 - пройдены обе ветви
            mutable int which_way_ = -1; // По какому пути нужно пройти вверх
        };

        OperatorNode* root_ = NULL; // Корневой оператор
};

template<typename StateType>
void Operator<StateType>::OperatorNode::invoke(const State<StateType>& st, State<StateType>& global_state) const {
    if (this->func_type_ == INTUITIVE_TYPE) {
        for (auto cur_state: st.get_basis()) {
            this->cur_res_ += this->func_(cur_state) * st[cur_state];
        }
    // Is effective?
    } else if (this->func_type_ == MEM_EFFECTIVE_TYPE) {
        global_state.set_zero();
        for (auto cur_state: st.get_basis()) {
            this->effective_func_(*cur_state, global_state);
            this->cur_res_ += global_state * st[*cur_state];
            global_state.set_zero();
        }
    }
}


template<typename StateType>
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

// don't work
template<typename StateType>
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

template<typename StateType>
State<StateType> Operator<StateType>::run(const State<StateType>& init_state, const BasisType<StateType>& basis) const {
    OperatorNode* cur_op = this->root_;
    State<StateType> global_state((BasisType<StateType>(basis)));

    while(this->root_->from_tree_ != 3) {
        if (cur_op->from_tree_ == 0 and cur_op->left != NULL) {
            cur_op->from_tree_ = 1;
            cur_op = cur_op->left;

        } else if (cur_op->from_tree_ == 0 and cur_op->left == NULL) {
            cur_op->from_tree_ = 2;
            cur_op->invoke(init_state, global_state);

            if (cur_op->right == NULL) {
                cur_op->from_tree_ = 3;
                cur_op = cur_op->up();
            } else {
                cur_op = cur_op->right;
            }
        } else if (cur_op->from_tree_ == 1 and cur_op->right != NULL) {
            cur_op->from_tree_ = 2;
            cur_op->invoke(cur_op->left->cur_res_, global_state);
            cur_op = cur_op->right;
        } else if (cur_op->from_tree_ == 1 and cur_op->right == NULL) {
            cur_op->from_tree_ = 3;
            cur_op->invoke(cur_op->left->cur_res_, global_state);
            cur_op = cur_op->up();
        } else if (cur_op->from_tree_ == 2) {
            cur_op->from_tree_ = 3;
            cur_op->cur_res_ += cur_op->right->cur_res_;
        } else if (cur_op->from_tree_ == 3) {
            cur_op = cur_op->up();
        }
    }

    // auto res = root_->cur_res_.copy();
    auto res = root_->cur_res_;
    this->refresh_tree();

    return res;
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!! REWRITE TO res_state.set_sorted(true) like in hamiltonian !!!!!!!!!!!!!!!!!!!!!

template<typename StateType>
Matrix<COMPLEX> operator_to_matrix(const Operator<StateType>& op, const std::vector<std::shared_ptr<StateType>>& basis, MATRIX_STYLE matrix_style = C_STYLE) {
    size_t dim = basis.size();
    State<StateType> basis_map(basis);
    Matrix<COMPLEX> A(matrix_style, dim, dim, COMPLEX(0, 0));

    size_t col_state = 0;

    for (auto state: basis) {
        auto res_state = op.run(State<StateType>(state));
        res_state.sort();

        size_t index = 0;
        auto res_state_vec = res_state.get_basis();
        for (auto state_res: res_state_vec) {
            // if (matrix_style == C_STYLE) A[get_index_state_in_basis(*state_res, basis)][col_state] = res_state[index++];
            // else A(get_index_state_in_basis(*state_res, basis), col_state) = res_state[index++];
            if (matrix_style == C_STYLE) A[basis_map.get_index(state_res)][col_state] = res_state[index++];
            else A(basis_map.get_index(state_res), col_state) = res_state[index++];
        }

        col_state++;
    }
    /*
    std::function<COMPLEX(size_t i, size_t j)> func = {
        [&basis, &op](size_t i, size_t j) {
            auto state_from = get_state_from_basis(basis, j);
            auto state_to = get_state_from_basis(basis, i);
            auto res_state = op.run(State<StateType>(*state_from));
            
            if (res_state.is_in_state(*state_to)) {
                return res_state[*state_to];
            } else {
                return COMPLEX(0, 0);
            }
        }
    };

    Matrix<COMPLEX> A(C_STYLE, dim, dim, func);
    */

    return A;
}

template<typename StateType>
inline Matrix<COMPLEX> operator_to_matrix(const Operator<StateType>& op, const BasisType<StateType>& basis, MATRIX_STYLE matrix_style = C_STYLE) {
    // size_t dim = basis.size();
    // Matrix<COMPLEX> A(matrix_style, dim, dim, COMPLEX(0, 0));

    // size_t col_state = 0;

    // for (auto state: basis) {
    //     auto res_state = op.run(State<StateType>(*state));

    //     size_t index = 0;
    //     for (auto state_res: res_state.state_components()) {
    //         if (matrix_style == C_STYLE) A[get_index_state_in_basis(*state_res, basis)][col_state] = res_state[index++];
    //         else A(get_index_state_in_basis(*state_res, basis), col_state) = res_state[index++];
    //     }

    //     col_state++;
    // }
    // /*
    // std::function<COMPLEX(size_t i, size_t j)> func = {
    //     [&basis, &op](size_t i, size_t j) {
    //         auto state_from = get_state_from_basis(basis, j);
    //         auto state_to = get_state_from_basis(basis, i);
    //         auto res_state = op.run(State<StateType>(*state_from));
            
    //         if (res_state.is_in_state(*state_to)) {
    //             return res_state[*state_to];
    //         } else {
    //             return COMPLEX(0, 0);
    //         }
    //     }
    // };

    // Matrix<COMPLEX> A(C_STYLE, dim, dim, func);
    // */

    return operator_to_matrix(op, sort_basis(basis), matrix_style);
}

// !!!!!!!!!!!!!!!!!! REWRITE to CSR_Matrix manipulations vals, ia, ja without copy !!!!!!!!!!!!!!!!!!!!!!

#ifdef ENABLE_ONEAPI

template<typename StateType>
CSR_Matrix<COMPLEX> operator_to_matrix_csr(const Operator<StateType>& op, const std::vector<std::shared_ptr<StateType>>& basis) {
    size_t dim = basis.size();
    State<StateType> basis_map(basis);
    size_t col_state = 0;

    std::vector<COMPLEX> vals;
    std::vector<ILP_TYPE> ia({0});
    std::vector<ILP_TYPE> ja;

    size_t index = 0;
    for (auto state: basis) {
        auto res_state = op.run(State<StateType>(state));
        res_state.set_sorted(true);

        auto res_state_vec = res_state.get_basis();
        for (auto state_res: res_state_vec) {
            // if (matrix_style == C_STYLE) A[get_index_state_in_basis(*state_res, basis)][col_state] = res_state[index++];
            // else A(get_index_state_in_basis(*state_res, basis), col_state) = res_state[index++];
            vals.emplace_back(std::conj(res_state[index++]));
            ja.emplace_back(basis_map.get_index(state_res));
        }

        ia.emplace_back(index);
    }

    CSR_Matrix A(ia.size() - 1, basis.size(), vals, ia, ja);
    A.sort_ja();
    /*
    std::function<COMPLEX(size_t i, size_t j)> func = {
        [&basis, &op](size_t i, size_t j) {
            auto state_from = get_state_from_basis(basis, j);
            auto state_to = get_state_from_basis(basis, i);
            auto res_state = op.run(State<StateType>(*state_from));
            
            if (res_state.is_in_state(*state_to)) {
                return res_state[*state_to];
            } else {
                return COMPLEX(0, 0);
            }
        }
    };

    Matrix<COMPLEX> A(C_STYLE, dim, dim, func);
    */

    return A;
}

template<typename StateType>
inline CSR_Matrix<COMPLEX> operator_to_matrix_csr(const Operator<StateType>& op, const BasisType<StateType>& basis) {
    return operator_to_matrix_csr(op, sort_basis(basis));
}

#endif

#ifdef ENABLE_MPI
#ifdef ENABLE_CLUSTER
template<typename StateType>
BLOCKED_Matrix<COMPLEX> operator_to_matrix(ILP_TYPE ctxt, const Operator<StateType>& op, const BasisType<StateType>& basis) {
    // size_t dim = basis.size();
    // std::vector<std::shared_ptr<StateType>> basis_vec;
    // std::copy(basis.begin(), basis.end(), std::back_inserter(basis_vec));

    // BLOCKED_Matrix<COMPLEX> A(ctxt, GE, dim, dim, COMPLEX(0, 0));

    // bool is_by_cols = true;
    // auto local_size = A.local_m();
    // if (local_size == A.m()) {
    //     local_size = A.local_n();
    //     is_by_cols = false;
    // }

    // for (size_t j = 0; j < local_size; j++) {
    //     size_t global_state_index;
    //     if (is_by_cols) {
    //         global_state_index = A.get_global_col(j);
    //     } else {
    //         global_state_index = A.get_global_row(j);
    //     }

    //     //auto state_from = basis_vec[global_state_index];
    //     auto state_from = get_state_from_basis<StateType>(basis, global_state_index);
    //     auto res_state = op.run(State<StateType>(state_from));

    //     size_t index = 0;
    //     for (auto state: res_state.state_components()) {
    //         if (is_by_cols) {
    //             auto cur_global_row = get_index_state_in_basis(*state, basis);
    //             if (A.is_my_elem_row(cur_global_row)) {
    //                 A(A.get_local_row(cur_global_row), j) = res_state[index];
    //             }
    //         } else {
    //             auto cur_global_col = get_index_state_in_basis(*state, basis);
    //             if (A.is_my_elem_col(cur_global_col)) {
    //                 A(j, A.get_local_col(cur_global_col)) = std::conj(res_state[index]);
    //             }
    //         }

    //         index++;
    //     }
    // }

    // /*

    // std::function<COMPLEX(size_t i, size_t j)> func = {
    //     [&basis, &op](size_t i, size_t j) {
    //         auto state_from = get_state_from_basis(basis, j);
    //         auto state_to = get_state_from_basis(basis, i);
    //         auto res_state = op.run(State<StateType>(*state_from));
            
    //         if (res_state.is_in_state(*state_to)) {
    //             return res_state[*state_to];
    //         } else {
    //             return COMPLEX(0, 0);
    //         }
    //     }
    // };

    // BLOCKED_Matrix<COMPLEX> A(ctxt, GE, dim, dim, func);
    // */
    // return A;

    return operator_to_matrix(ctxt, op, sort_basis(basis));
}

template<typename StateType>
BLOCKED_Matrix<COMPLEX> operator_to_matrix(ILP_TYPE ctxt, const Operator<StateType>& op, const std::vector<std::shared_ptr<StateType>>& basis) {
    size_t dim = basis.size();
    State<StateType> basis_map(basis);

    BLOCKED_Matrix<COMPLEX> A(ctxt, GE, dim, dim, COMPLEX(0, 0));

    bool is_by_cols = true;
    auto local_size = A.local_m();
    if (local_size == A.m()) {
        local_size = A.local_n();
        is_by_cols = false;
    }

    for (size_t j = 0; j < local_size; j++) {
        size_t global_state_index;
        if (is_by_cols) {
            global_state_index = A.get_global_col(j);
        } else {
            global_state_index = A.get_global_row(j);
        }

        //auto state_from = basis_vec[global_state_index];
        // auto state_from = get_state_from_basis<StateType>(basis, global_state_index);
        auto state_from = basis[global_state_index];
        auto res_state = op.run(State<StateType>(state_from));

        size_t index = 0;
        res_state.set_sorted(true);
        // auto sorted_basis = res_state.get_basis();
        for (auto p: res_state.state_map()) {
            if (is_by_cols) {
                // auto cur_global_row = get_index_state_in_basis(*state, basis);
                auto cur_global_row = basis_map.get_index(p.first);
                if (A.is_my_elem_row(cur_global_row)) {
                    A(A.get_local_row(cur_global_row), j) = res_state[p.second];
                }
            } else {
                // auto cur_global_col = get_index_state_in_basis(*state, basis);
                auto cur_global_col = basis_map.get_index(p.first);
                if (A.is_my_elem_col(cur_global_col)) {
                    A(j, A.get_local_col(cur_global_col)) = std::conj(res_state[p.second]);
                }
            }

            index++;
        }
    }

    /*

    std::function<COMPLEX(size_t i, size_t j)> func = {
        [&basis, &op](size_t i, size_t j) {
            auto state_from = get_state_from_basis(basis, j);
            auto state_to = get_state_from_basis(basis, i);
            auto res_state = op.run(State<StateType>(*state_from));
            
            if (res_state.is_in_state(*state_to)) {
                return res_state[*state_to];
            } else {
                return COMPLEX(0, 0);
            }
        }
    };

    BLOCKED_Matrix<COMPLEX> A(ctxt, GE, dim, dim, func);
    */
    return A;
}
#endif
#endif

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

template<typename StateType>
inline double expectation_value(const State<StateType>& psi, const Operator<StateType>& op) {
    return std::abs(scalar_product(psi, op.run(psi)));
}



// ---------------------------- OPERATORS ---------------------------

template<typename StateType>
State<StateType> set_qudit(const std::shared_ptr<StateType>& state, ValType val, size_t qudit_index = 0, size_t group_id = 0, const std::string& info = "") {
    auto res = *state;
    if (val > state->get_max_val(qudit_index, group_id) or val < 0) {
        return State<StateType>();
    } else {
        res.set_qudit(val, qudit_index, group_id);
    }

    return State<StateType>(res);
}

template<typename StateType>
State<StateType> get_qudit(const std::shared_ptr<StateType>& state, size_t qudit_index = 0, size_t group_id = 0, const std::string& info = "") {
    auto res = State<StateType>(state);
    res[0] = state->get_qudit(qudit_index, group_id);

    return res;
}

template<typename StateType>
State<StateType> sigma_x(const std::shared_ptr<StateType>& state, size_t qudit_index = 0, size_t group_id = 0, const std::string& info = "") {
    StateType res = *state;
    auto qudit = state->get_qudit(qudit_index, group_id);
    assert(qudit == 0 or qudit == 1);

    if (qudit == 0) qudit = 1;
    else qudit = 0;

    res->set_qudit(qudit, qudit_index, group_id);

    return State<StateType>(res);
}

template<typename StateType>
State<StateType> sigma_y(const std::shared_ptr<StateType>& state, size_t qudit_index = 0, size_t group_id = 0, const std::string& info = "") {
    StateType res = *state;
    auto qudit = state->get_qudit(qudit_index, group_id);
    assert(qudit == 0 or qudit == 1);

    if (qudit == 0) { 
        qudit = 1;
    } else {
        qudit = 0;
    }

    res->set_qudit(qudit, qudit_index, group_id);

    State<StateType> stateres(res);
    stateres[0] *= COMPLEX(0, std::pow(-1, qudit + 1));

    return stateres;
}

template<typename StateType>
State<StateType> sigma_z(const std::shared_ptr<StateType>& state, size_t qudit_index = 0, size_t group_id = 0, const std::string& info = "") {
    auto res = get_qudit(state, qudit_index, group_id);
    auto qudit = state->get_qudit(qudit_index, group_id);
    assert(qudit == 0 or qudit == 1);

    if (qudit == 1) {
        res[0] *= -1;
    }

    return res;
}

template<typename StateType>
State<StateType> H(const std::shared_ptr<StateType>& state, size_t qudit_index = 0, size_t group_id = 0, bool is_sqrt_multiply = true) {
    auto res = get_qudit(state, qudit_index, group_id);
    res[0] = COMPLEX(1 / std::sqrt(2), 0) * double(is_sqrt_multiply) + double(1) * double(!is_sqrt_multiply);

    auto qudit = state->get_qudit(qudit_index, group_id);
    assert(qudit == 0 or qudit == 1);
    res[0] *= std::pow(-1, qudit);
    COMPLEX coef = COMPLEX(1 / std::sqrt(2), 0) * double(is_sqrt_multiply) + double(1) * double(!is_sqrt_multiply);

    if (qudit == 0) { 
        qudit = 1;
    } else {
        qudit = 0;
    }

    auto tmp = *state;
    tmp.set_qudit(qudit, qudit_index, group_id);

    res.insert(tmp, coef);

    return res;
}

template<typename StateType>
State<StateType> check(const std::shared_ptr<StateType>& state, ValType check_val, size_t qudit_index = 0, size_t group_id = 0, const std::string& info = "") {
    if (state->get_qudit(qudit_index, group_id) != check_val) {
        return State<StateType>();
    }

    return State<StateType>(state);
}

template<typename StateType>
State<StateType> check_func(const std::shared_ptr<StateType>& state,
                       const std::function<bool(const StateType&)>& func) {
    if (func(state)) {
        return State<StateType>();
    }

    return State<StateType>(state);
}

template<typename StateType>
State<StateType> WH(const std::shared_ptr<StateType>& st) {
    State<StateType> res(st);
    for (size_t i = 0; i < st->qudits_count(); i++) {
        State<StateType> tmp;
        // size_t index_st = 0;
        // auto basis = res.get_basis();
        res.set_sorted(true);
        for (auto p: res.state_map()) {
            // tmp += H(cur_st, i) * res[index_st++];
            tmp += H(p.first, i, 0, false) * res[p.second];
        }

        res = tmp;
    }

    res *= double(1)/std::sqrt(res.size());

    return res;
}

template<typename StateType>
void eff_WH(const StateType& st, State<StateType>& res) {
    res.insert(st);
    res[st] = COMPLEX(1, 0);
    State<StateType> tmp(res.state_components());
    tmp.set_zero();
    for (size_t i = 0; i < st.qudits_count(); i++) {
        size_t index_st = 0;
        for (auto cur_st: res.state_components()) {
            if (!is_zero(res[index_st])) {
                tmp += H(*cur_st, i) * res[index_st++];
            }
        }

        res = tmp;
        tmp.set_zero();
    }
}

State<X_State> X_OP(const std::shared_ptr<X_State>& state);
State<P_State> P_OP(const std::shared_ptr<P_State>& state);

State<TCH_State> photons_transfer(const std::shared_ptr<TCH_State>& state);
State<TCH_State> photons_count(const std::shared_ptr<TCH_State>& state);
State<TCH_State> atoms_exc_count(const std::shared_ptr<TCH_State>& state);
State<TCH_State> exc_relax_atoms(const std::shared_ptr<TCH_State>& state);

/*
State<TCH_State> a_destroy(const TCH_State& state);
State<TCH_State> a_create(const TCH_State& state);
State<TCH_State> sigma_destroy(const TCH_State& state);
State<TCH_State> sigma_create(const TCH_State& state);
State<TCH_State> photons_count(const TCH_State& state);
State<TCH_State> atoms_exc_count(const TCH_State& state);
*/

/* ------------------------- Генерация матриц операторов --------------------- */

#ifdef ENABLE_MPI
#ifdef ENABLE_CLUSTER

template<typename StateType>
BLOCKED_Matrix<COMPLEX> BLOCKED_Matrix_by_Scalar_Product(ILP_TYPE ctxt, const std::function<COMPLEX(const StateType&, const StateType&)>& func, const BasisType<StateType>& basis) {
//BLOCKED_Matrix<COMPLEX> BLOCKED_Matrix_by_Scalar_Product(ILP_TYPE ctxt, const std::function<COMPLEX(const std::shared_ptr<StateType>&, const std::shared_ptr<StateType>&)>& func, const BasisType<StateType>& basis) {
    auto matrix_func = std::function<COMPLEX(size_t i, size_t j)>{
        [&func, &basis](size_t i, size_t j) {
            return func(*get_state_from_basis(basis, i), *get_state_from_basis(basis, j));
            //return func(get_state_from_basis(basis, i), get_state_from_basis(basis, j));
        }
    };

    return BLOCKED_Matrix<COMPLEX>(ctxt, GE, basis.size(), basis.size(), matrix_func);
}

template<typename StateType>
BLOCKED_Matrix<COMPLEX> WH_Matrix(ILP_TYPE ctxt, const std::vector<std::shared_ptr<StateType>>& basis) {
    double coef = 1/std::sqrt(basis.size());
    /*
    //std::function<COMPLEX(const std::shared_ptr<StateType>&, const std::shared_ptr<StateType>&)> f = {[coef](const std::shared_ptr<StateType>& i, const std::shared_ptr<StateType>& j) {
    std::function<COMPLEX(const StateType&, const StateType&)> f = {[coef](const StateType& i, const StateType& j) {
        return coef*(-1 + 2 * ((scalar_product(i.ref_qudits(), j.ref_qudits())) + 1) % 2);
        //return 1/sqrt_n*(-1 + 2 * ((scalar_product(i->qudits(), j->qudits()) + 1) % 2));
        
    }};
    */
    // std::vector<std::shared_ptr<StateType>> states;
    // std::copy(basis.begin(), basis.end(), std::back_inserter(states));

    auto matrix_func = std::function<COMPLEX(size_t i, size_t j)>{
        [coef, &basis](size_t i, size_t j) {
            //return coef*(-1 + 2 * ((scalar_product(get_state_from_basis(basis, i)->qudits_data(), get_state_from_basis(basis, j)->qudits_data(), basis.size())) + 1) % 2);
            return coef*(-1 + 2 * (((and_sum(basis[i]->qudits_data(), basis[j]->qudits_data(), basis.size())) + 1) % 2));
            //return func(get_state_from_basis(basis, i), get_state_from_basis(basis, j));
        }
    };

    //return BLOCKED_Matrix_by_Scalar_Product(ctxt, f, basis);
    return BLOCKED_Matrix<COMPLEX>(ctxt, GE, basis.size(), basis.size(), matrix_func);
}

#endif
#endif

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
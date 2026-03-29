#pragma once


// NEED TESTING AND IMPROVEMENTS


#include "state.hpp"
#include "functions.hpp"

namespace QComputations {

    class X_State : public Basis_State {
        public:
            X_State(double a, double b, size_t N) : N_(N), h_((b - a) / N), a_(a), b_(b), Basis_State(1, N) {}

            inline void set_x_i(size_t index) { this->set_qudit(index); }
            inline double get_x() const { return (a_ + h_ * this->get_qudit(0)); }
            inline void step_forward() { this->set_qudit(qudits_[0] + 1); }
            inline void step_backward() { this->set_qudit(qudits_[0] - 1); }

            static std::vector<std::shared_ptr<X_State>> make_basis(double a, double b, size_t N) {
                std::vector<std::shared_ptr<X_State>> basis;
                for (size_t i = 0; i <= N; ++i) {
                    auto state = std::make_shared<X_State>(a, b, N);
                    state->set_x_i(i);
                    basis.emplace_back(state);
                }
                return basis;
            }

            inline double a() const { return a_; }
            inline double b() const { return b_; }
            inline double h() const { return h_; }
            inline size_t N() const { return N_; }

            inline bool operator<(const Basis_State& other) const override { return this->qudits_ref()[0] > other.qudits_ref()[0]; }
        private:
            size_t N_;
            double h_, a_, b_;
    };
    
    class P_State : public Basis_State {
        public:
            P_State(double a, double b, size_t N) : N_(N), h_((b - a) / N), a_(a), b_(b), Basis_State(1, N) {}

            inline void set_p_i(size_t index) { this->set_qudit(index); }
            inline double get_p() const { return (a_ + h_ * this->get_qudit(0)); }
            inline void step_forward() { this->set_qudit(qudits_[0] + 1); }
            inline void step_backward() { this->set_qudit(qudits_[0] - 1); }
        
            static std::vector<std::shared_ptr<P_State>> make_basis(double a, double b, size_t N) {
                std::vector<std::shared_ptr<P_State>> basis;
                for (size_t i = 0; i <= N; ++i) {
                    auto state = std::make_shared<P_State>(a, b, N);
                    state->set_p_i(i);
                    basis.emplace_back(state);
                }
                return basis;
            }

            inline double a() const { return a_; }
            inline double b() const { return b_; }
            inline double h() const { return h_; }
            inline size_t N() const { return N_; }

            inline bool operator<(const Basis_State& other) const override { return this->qudits_ref()[0] > other.qudits_ref()[0]; }
        private:
            size_t N_;
            double h_, a_, b_;
    };

    // Needs rethinking
    // class K_State : public Basis_State {
    //     public:
    //         K_State(double a, double b, size_t N) : N_(N + 1), h_((b - a) / N), a_(a), b_(b), Basis_State(1, N) {}

    //         inline void set_k_i(size_t index) { this->set_qudit(index); }
    //         inline double get_k() const { return (a_ + h_ * this->get_qudit(0)); }
    //         inline void step_forward() { this->set_qudit(qudits_[0] + 1); }
    //         inline void step_backward() { this->set_qudit(qudits_[0] - 1); }
    //     private:
    //         size_t N_, h_, a_, b_;
    // }

    template<typename StateTypeFrom, typename StateTypeTo>
    State<StateTypeTo> fft_forward(const State<StateTypeFrom>& state_from) {
        auto basis_from = state_from.get_basis();
        size_t M = basis_from.size();
        double dx = basis_from[0]->h();

        std::vector<COMPLEX> amplitudes(M);
        for (size_t i = 0; i < M; ++i) {
            amplitudes[i] = state_from[basis_from[i]];
        }

        amplitudes = fftshift(amplitudes);
        amplitudes = fft_forward(amplitudes);
        amplitudes = fftshift(amplitudes);

        double dp = 2.0 * M_PI / (M * dx);
        double p_min = -dp * M / 2;
        double p_max = p_min + (M - 1) * dp;

        auto basis_to = StateTypeTo::make_basis(p_min, p_max, M - 1);
        State<StateTypeTo> state_to(basis_to);
        for (size_t k = 0; k < M; ++k) state_to[basis_to[k]] = amplitudes[k];
        return state_to;
    }

    template<typename StateTypeFrom, typename StateTypeTo>
    State<StateTypeTo> fft_backward(const State<StateTypeFrom>& state_from) {
        // Получаем базис исходного состояния (импульсного)
        auto basis_from = state_from.get_basis();
        size_t M = basis_from.size();

        // Шаг по импульсу (предполагается, что у StateTypeFrom есть метод h())
        double dp = basis_from[0]->h();

        // Извлекаем амплитуды в порядке исходного базиса (от p_min до p_max)
        std::vector<COMPLEX> amplitudes(M);
        for (size_t i = 0; i < M; ++i) {
            amplitudes[i] = state_from[basis_from[i]];
        }

        // Обратное преобразование с учётом сдвига:
        // 1. Обратный сдвиг (приводим к стандартному порядку: нулевая частота в начале)
        amplitudes = fftshift(amplitudes);
        // 2. Обратное БПФ (с нормировкой 1/√M)
        amplitudes = fft_backward(amplitudes);
        // 3. Сдвиг результата, чтобы получить координаты от x_min до x_max
        amplitudes = fftshift(amplitudes);

        double dx = 2.0 * M_PI / (M * dp);
        double x_min = -dx * M / 2;
        double x_max = x_min + (M - 1) * dx;

        // Целевой базис (координатный) уже передан пользователем
        auto basis_to = StateTypeTo::make_basis(x_min, x_max, M - 1);
        State<StateTypeTo> state_to(basis_to);
        for (size_t k = 0; k < M; ++k) {
            state_to[basis_to[k]] = amplitudes[k];
        }
        return state_to;
    }

    inline State<P_State> fft_x_to_p(const State<X_State>& psi_x) {
        return fft_forward<X_State, P_State>(psi_x);
    }

    inline State<X_State> fft_p_to_x(const State<P_State>& psi_p) {
        return fft_backward<P_State, X_State>(psi_p);
    }
    
}
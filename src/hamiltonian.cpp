#include <vector>
#include <functional>
#include <complex>
#include <iostream>
#include <iomanip>
#include "hamiltonian.hpp"
#include "config.hpp"
#include "additional_operators.hpp"
#include "graph.hpp"
#include "functions.hpp"

namespace {
    typedef std::complex<double> COMPLEX;

    Matrix<COMPLEX> tensor_multiply(const Matrix<COMPLEX>& A, const Matrix<COMPLEX>& B) {
        auto n = A.size() * B.size();
        Matrix<COMPLEX> C(n, n, 0);

        for (size_t i_a = 0; i_a < A.size(); i_a++) {
            for (size_t j_a = 0; j_a < A.size(); j_a++) {
                for (size_t i_b = 0; i_b < B.size(); i_b++) {
                    for (size_t j_b = 0; j_b < B.size(); j_b++) {
                        size_t i = i_a * B.size() + i_b;
                        size_t j = j_a * B.size() + j_b;

                        C[i][j] = A[i_a][j_a] * B[i_b][j_b];
                    }
                }
            }
        }

        return C;
    }

    /*
    matrix operator+=(Matrix<COMPLEX>& A, const Matrix<COMPLEX>& B) {
        for (int i = 0; i < A.size(); i++) {
            for (int j = 0; j < A.size(); j++) {
                A[i][j] += B[i][j];
            }
        }

        return A;
    }

    Matrix<COMPLEX> operator * (Matrix<COMPLEX>& A, const std::complex<double>& c) {
        for (int i = 0; i < A.size(); i++) {
            for (int j = 0; j < A.size(); j++) {
                A[i][j] *= c;
            }
        }

        return A;
    }
    */

    Matrix<COMPLEX> a_destroy(size_t n) {
        size_t size = n + 1;

        Matrix<COMPLEX> a(size, size, 0);

        int j = 1;
        for (int i = 0; i < size - 1; i++) {
            a[i][j] = std::sqrt(j);
            j++;
        }

        return a;
    }

    Matrix<COMPLEX> E_photons(int n) {
        Matrix<COMPLEX> E(n, n, 0);
        for (int i = 0; i < n; i++) {
            E[i][i] = i * config::h * config::w;
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
        Matrix<COMPLEX> sum_sigma(size, size, 0);

        for (int i = 0; i < n; i++) {
            size_t left_size = std::pow(2, i);
            size_t right_size = std::pow(2, n - i - 1);

            Matrix<COMPLEX> eye_left(left_size, left_size, 0);
            Matrix<COMPLEX> eye_right(right_size, right_size, 0);

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

    /*
    Matrix transpose(const matrix& A) {
        matrix B(A.size(), vec_complex(A.size()));

        for (int i = 0; i < A.size(); i++) {
            for (int j = 0; j < A.size(); j++) {
                B[i][j] = A[j][i];
            }
        }

        return B;
    }
    */
}

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

H_by_func::H_by_func(size_t n, std::function<COMPLEX(size_t, size_t)> func) : n_(n), func_(func) {
    auto size = std::pow(2, n);
    H_ = Matrix<COMPLEX>(size, size);
    for (size_t i = 0; i < this->size(); i++) {
        for (size_t j = 0; j < this->size(); j++) {
            H_[i][j] = func(i, j);
        }
    }
}

H_TC::H_TC(size_t n, size_t m, const State& init_state): n_(n), m_(m) {
    auto size_m_ = std::pow(2, m_);
    State_Graph graph(init_state, config::LOSS_PHOTONS);
    basis_ = graph.get_basis();
    auto size_H_ = basis_.size();

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

        Matrix<COMPLEX> eye_left(size_left, size_left, 0);
        Matrix<COMPLEX> eye_right(size_right, size_right, 0);

        for (int i = 0; i < size_left; i++) eye_left[i][i] = 1;
        for (int i = 0; i < size_right; i++) eye_right[i][i] = 1;

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

/*
matrix H_TC::Reduce_H(const State& init_state) {
    if (init_state_ == init_state) {
        return Reduced_H_;
    }
    init_state_ = init_state;
    is_reduced_ = true;
    is_reduced_eigen_ = false;
    State_Graph graph(init_state, config::LOSS_PHOTONS);

    bases_ = graph.get_bases();

    Reduced_H_ = matrix(bases_.size(), vec_complex(bases_.size()));

    size_t index_i = 0;
    size_t index_j = 0;
    for (const auto& basis_row: bases_) {
        for (const auto& basis_col: bases_) {
            size_t i = basis_row.get_index();
            size_t j = basis_col.get_index();

            Reduced_H_[index_i][index_j] = H_[i][j];
            index_j++;
        }

        index_j = 0;
        index_i++;
    }

    return Reduced_H_;
}

std::pair<std::vector<double>, matrix> H_TC::reduced_eigen() {
    if (is_reduced_eigen_) {
        return std::make_pair(reduced_eigenvalues_, reduced_eigenvectors_);
    }
    auto res = jacobi(Reduced_H_);

    is_reduced_eigen_ = true;
    reduced_eigenvectors_ = res.second;
    reduced_eigenvalues_ = res.first;

    return std::make_pair(reduced_eigenvalues_, reduced_eigenvectors_);
}

void Hamiltonian::show_reduced() const {
    for (size_t i = 0; i < bases_.size(); i++) {
        for (size_t j = 0; j < bases_.size(); j++) {
            std::cout << std::setw(config::WIDTH) << Reduced_H_[i][j] << " ";
        }

        std::cout << std::endl;
    }
}
*/
H_TCH::H_TCH(int n, int n_pol) {}

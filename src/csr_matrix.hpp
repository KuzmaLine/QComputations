#pragma once
#include <iostream>
#include <complex>
#include <vector>

#ifndef DEFINED_MATRIX
#define DEFINED_MATRIX
#include "matrix.hpp"
#endif

#include <functional>
#include <iomanip>
#include "config.hpp"
#include "additional_operators.hpp"

namespace {
    template <typename T>
    bool is_close(T a, T b) {
        return (std::abs(a - b) < config::eps);
    }

    size_t get_index_value(size_t row_index, const std::vector<size_t>& row_index_pointers) {
        size_t index_value = 0;
        for (size_t k = 0; k < row_index; k++) {
            index_value = row_index_pointers[k + 1];
        }

        return index_value;
    }

    template <typename T>
    void insert_value_csr(size_t i, size_t j, T value,
                          std::vector<T>& values, std::vector<size_t>& col_indices,
                          std::vector<size_t>& row_index_pointers) {
        auto index_value = get_index_value(i, row_index_pointers);
        
        //std::cout << index_value << std::endl;
        if (index_value == values.size()) {
            values.emplace_back(value);
            col_indices.emplace_back(j);
        } else {
            for (size_t k = 0; k < row_index_pointers[i + 1] - row_index_pointers[i]; k++) {
                if (col_indices[index_value] < j) {
                    index_value++;
                } else {
                    break;
                }
            }

            values.insert(std::next(values.begin(), index_value), value);
            col_indices.insert(std::next(col_indices.begin(), index_value), j);
        }

        //std::cout << "HERE\n";

        for (size_t k = i + 1; k < row_index_pointers.size(); k++) {
            row_index_pointers[k]++;
        }
    }
}

template<typename T> class CSR_Matrix {
    public:
        explicit CSR_Matrix() = default;
        explicit CSR_Matrix(const Matrix<T>& A, T default_value = T(0));
        explicit CSR_Matrix(std::function<T(size_t, size_t)> func, T default_value = T(0));
        explicit CSR_Matrix(const CSR_Matrix<T>& A) = default;

        void insert_value(size_t i, size_t j, T value);
        const T operator()(size_t i, size_t j) const;
        T operator()(size_t i, size_t j);
        void delete_value(size_t i, size_t j);
        bool is_contain(size_t i, size_t j) const;

        void show() const;
    private:
        size_t n_;
        size_t m_;
        std::vector<T> values_;
        std::vector<size_t> col_indices_;
        std::vector<size_t> row_index_pointers_;
        T default_value_;
};

template<typename T>
CSR_Matrix<T>::CSR_Matrix(const Matrix<T>& A, T default_value) : n_(A.n()), m_(A.m()), row_index_pointers_(n_ + 1, 0), default_value_(default_value) {
    for (size_t i = 0; i < n_; i++) {
        for (size_t j = 0; j < m_; j++) {
            if (!is_close(A.elem(i, j), default_value)) {
                //std::cout << "NEW_ELEM: " << i << " " << j << " " << A.elem(i, j) << std::endl;
                //std::cout << values_ << std::endl << col_indices_ << std::endl << row_index_pointers_ << std::endl;
                insert_value_csr(i, j, A.elem(i, j), values_, col_indices_, row_index_pointers_);
            }
        }
    }
}

template<typename T>
const T CSR_Matrix<T>::operator()(size_t i, size_t j) const {
    size_t index_value = get_index_value(i, row_index_pointers_);

    for (size_t k = 0; k < row_index_pointers_[i + 1] - row_index_pointers_[i]; k++) {
        if (col_indices_[index_value] == j) { 
            return values_[index_value];
        }

        index_value++;
    }

    return default_value_;
}

template<typename T>
void CSR_Matrix<T>::show() const {
    for (size_t i = 0; i < n_; i++) {
        for (size_t j = 0; j < m_; j++) {
            std::cout << std::setw(config::WIDTH) << (*this)(i, j) << " ";
        }
        std::cout << std::endl;
    }
}
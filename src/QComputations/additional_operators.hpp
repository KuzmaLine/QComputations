#pragma once
#include <vector>
#include <complex>
#include <iomanip>
#include <map>
#include <iostream>
#include "state.hpp"

namespace {
    using COMPLEX = std::complex<double>;
    using E_LEVEL = int;
    using vec_levels = std::vector<E_LEVEL>;
}

namespace QComputations {

// Additional operators for vector, Matrix, States and so on

Matrix<COMPLEX> operator* (const Matrix<COMPLEX>& A, const Matrix<double>& B);

/// ################################ std::vector ###################################

// vector * num
template<typename T>
std::vector<T> operator*(const std::vector<T>& v, T num) {
    std::vector<T> answer(v.size());

    for (size_t i = 0; i < v.size(); i++) {
        answer[i] = v[i] * num;
    }

    return answer;
}

// std::cout << vector
template<typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
    for (const auto& val: v) {
        out << val << " ";
    }

    return out;
}

// num * vector
template<typename T>
std::vector<T> operator*(T num, const std::vector<T>& v) { return v * num; }

// a[i] * b[i] -> vector
template<typename T>
std::vector<T> operator*(const std::vector<T>& a, const std::vector<T>& b) {
    std::vector<T> answer(a.size());

    for (size_t i = 0; i < a.size(); i++) {
        answer[i] = a[i] * b[i];
    }

    return answer;
}

//   -------------------------------------- <a|b> ------------------------
COMPLEX operator | (const std::vector<COMPLEX>& a, const std::vector<COMPLEX>& b);

// vector / num
template<typename T>
std::vector<T> operator/(const std::vector<T>& v, T num) {
    std::vector<T> answer(v.size());

    for (size_t i = 0; i < v.size(); i++) {
        answer[i] = v[i] / num;
    }

    return answer;
}

/*
template<typename T>
std::vector<T> operator/(T num, const std::vector<T>& v) { return v * num; }
*/

// vector - vector
template<typename T>
std::vector<T> operator-(const std::vector<T>& a, const std::vector<T>& b) {
    size_t m = a.size();
    std::vector<T> res(m);

    for (size_t i = 0; i < m; i++) {
        res[i] = a[i] - b[i];
    }

    return res;
}

} // namespace QComputations

// hash functions for Cavity_State and State (cavity_state.hpp and state.hpp)

template<>
struct std::hash<std::vector<COMPLEX>> {
    size_t operator()(const std::vector<COMPLEX>& v) const {
        std::hash<double> real_hash, imag_hash;
        size_t h = 0;
        for (const auto& c : v) {
            h ^= real_hash(c.real()) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= imag_hash(c.imag()) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }

};

template<> struct std::hash<std::vector<int>> {
    size_t operator()(const std::vector<int>& vec) const {
        size_t hash = 0;
        for (const auto& val : vec) {
            hash ^= std::hash<int>()(val) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

template <>
struct std::hash<std::map<int, std::vector<int>>> {
  std::size_t operator()(const std::map<int, std::vector<int>>& m) const {
    std::size_t seed = m.size();
    for (const auto& kv : m) {
      std::size_t key_hash = std::hash<int>()(kv.first);
      std::size_t val_hash = std::hash<std::vector<int>>()(kv.second);
      seed ^= key_hash + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      seed ^= val_hash + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

//bool operator()(const QComputations::Basis_State* a, const QComputations::Basis_State* b) {
//    return (a->Basis_State::operator<(*b));
//}

/*
template<>
struct std::hash<QComputations::Cavity_State> {
    size_t operator()(const QComputations::Cavity_State& state) const {
        return state.hash();
    }
};

template<>
struct std::hash<std::pair<QComputations::Cavity_State, QComputations::Cavity_State>> {
    size_t operator()(const std::pair<QComputations::Cavity_State, QComputations::Cavity_State>& p) const {
        std::hash<QComputations::Cavity_State> state_hash;
        size_t h1 = state_hash(p.first);
        size_t h2 = state_hash(p.second);
        return h1 ^ (h2 << 1);
    }
};


template<>
struct std::hash<QComputations::State> {
    size_t operator()(const QComputations::State& state) const {
        return state.hash();
    }
};

template<>
struct std::hash<std::pair<QComputations::State, QComputations::State>> {
    size_t operator()(const std::pair<QComputations::State, QComputations::State>& p) const {
        std::hash<QComputations::State> state_hash;
        size_t h1 = state_hash(p.first);
        size_t h2 = state_hash(p.second);
        return h1 ^ (h2 << 1);
    }
};
*/
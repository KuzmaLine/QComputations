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

template<typename T>
std::vector<T> operator*(const std::vector<T>& v, T num) {
    std::vector<T> answer(v.size());

    for (size_t i = 0; i < v.size(); i++) {
        answer[i] = v[i] * num;
    }

    return answer;
}

template<typename T>
std::vector<T> operator*(const std::vector<T>& a, const std::vector<T>& b) {
    std::vector<T> answer(a.size());

    for (size_t i = 0; i < a.size(); i++) {
        answer[i] = a[i] * b[i];
    }

    return answer;
}

template<typename T>
std::vector<T> operator*(T num, const std::vector<T>& v) {
    return v * num;
}

template<typename T>
std::vector<T> operator/(const std::vector<T>& v, T num) {
    std::vector<T> answer(v.size());

    for (size_t i = 0; i < v.size(); i++) {
        answer[i] = v[i] / num;
    }

    return answer;
}


template<typename T>
void show_vector(const std::vector<T>& v) {
    for (const auto& num: v) {
        std::cout << std::setw(15) << num << " ";
    }

    std::cout << std::endl;
}

template<typename T, typename R>
std::vector<T> operator*(const std::vector<std::vector<T>>& A, const std::vector<R>& v) {
    size_t m = A.size();
    std::vector<T> res(m, T(0));

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < m; j++) {
            res[i] += A[i][j] * v[j];
        }
    }

    return res;
}

template<typename T>
std::vector<T> operator-(const std::vector<T>& a, const std::vector<T>& b) {
    size_t m = a.size();
    std::vector<T> res(m);

    for (size_t i = 0; i < m; i++) {
        res[i] = a[i] - b[i];
    }

    return res;
}

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


template<>
struct std::hash<State> {
    size_t operator()(const State& state) const {
        return state.hash();
    }

};

template<>
struct std::hash<std::pair<State, State>> {
    size_t operator()(const std::pair<State, State>& p) const {
        std::hash<State> state_hash;
        size_t h1 = state_hash(p.first);
        size_t h2 = state_hash(p.second);
        return h1 ^ (h2 << 1);
    }
};

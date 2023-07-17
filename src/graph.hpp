#pragma once
#include <iostream>
#include <unordered_map>
#include <queue>
#include <unordered_set>
#include <set>
#include "basis.hpp"
#include "additional_operators.hpp"


class Basis_Graph {
    public:
        explicit Basis_Graph(const Basis& init_state, bool with_loss_photons = false);
        void show() const;

        std::set<Basis> get_basis() const { return basis_; }
    private:
        std::set<Basis> basis_;
        std::queue<Basis> basis_queue_;
        std::unordered_map<Basis, std::unordered_set<Basis>> to_;
        std::unordered_map<Basis, std::unordered_set<Basis>> from_;
};

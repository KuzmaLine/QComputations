#include "additional_operators.hpp"



/*
void show_matrix(const matrix& A) {
    for (const auto& v: A) {
        for (const auto& num: v) {
            std::cout << std::setw(25) << num << " ";
        }
        std::cout << std::endl;
    }
}
*/

void show_vector (const std::vector<COMPLEX>& v) {
    for (const auto& num: v) {
        std::cout << std::setw(15) << num << " ";
    }

    std::cout << std::endl;
}

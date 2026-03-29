// ! NOT FINISHED !!!!!!!!!!!!!!!!!!!!!

#include "QComputations_SINGLE.hpp"

using namespace QComputations;

void test_basis_creation() {
    double a = -5.0, b = 5.0;
    size_t N = 10; // количество интервалов
    auto basis_x = X_State::make_basis(a, b, N);
    // show_basis(basis_x);
    // std::cout << basis_x[0]->a() << " " << basis_x[0]->h() << " " << basis_x[0]->get_qudit(0) << std::endl;

    assert(basis_x.size() == N + 1);
    double h = (b - a) / N;
    for (size_t i = 0; i <= N; ++i) {
        // std::cout << basis_x[i]->get_x() << " " << a + i * h << std::endl;
        assert(std::abs(basis_x[i]->get_x() - (a + i * h)) < 1e-12);
        assert(basis_x[i]->a() == a);
        assert(basis_x[i]->b() == b);
        assert(basis_x[i]->N() == N);
        assert(std::abs(basis_x[i]->h() - h) < 1e-12);
    }

    auto basis_p = P_State::make_basis(a, b, N);
    assert(basis_p.size() == N + 1);
    for (size_t i = 0; i <= N; ++i) {
        assert(std::abs(basis_p[i]->get_p() - (a + i * h)) < 1e-12);
    }

    std::cout << "test_basis_creation passed\n";
}

void test_fourier_unitarity() {
    double a = -10.0, b = 10.0;
    size_t N = 511; // должно быть степенью двойки для FFT
    auto basis_x = X_State::make_basis(a, b, N);

    // Случайное состояние
    State<X_State> psi_x(basis_x);
    for (size_t i = 0; i <= N; ++i) {
        psi_x[basis_x[i]] = COMPLEX(rand() / (RAND_MAX + 1.0), rand() / (RAND_MAX + 1.0));
    }
    psi_x.normalize();

    // std::cout << psi_x.to_string() <<  " " << psi_x(0)->a() << " " << psi_x(0)->b() << " " << psi_x(0)->h() << std::endl;

    double norm_before = 0.0;
    for (size_t i = 0; i <= N; ++i) norm_before += std::norm(psi_x[basis_x[i]]);
    assert(std::abs(norm_before - 1.0) < 1e-10);

    // Прямое преобразование
    auto psi_p = fft_x_to_p(psi_x);
    auto psi_x_back = fft_p_to_x(psi_p);

        // std::cout << psi_x_back.to_string() << " " << psi_x_back(0)->a() << " " << psi_x_back(0)->b() << " " << psi_x_back(0)->h() << std::endl;
    // Проверка нормы
    double norm_after = 0.0;
    for (size_t i = 0; i <= N; ++i) norm_after += std::norm(psi_x_back[basis_x[i]]);
    // std::cout << norm_before << " " << norm_after << std::endl;
    assert(std::abs(norm_after - 1.0) < 1e-10);

    // Проверка близости исходного и восстановленного состояний
    double diff = 0.0;
    for (size_t i = 0; i <= N; ++i) {
        diff += std::norm(psi_x[basis_x[i]] - psi_x_back[basis_x[i]]);
    }
    assert(diff < 1e-10);

    std::cout << "test_fourier_unitarity passed\n";
}

void test_vacuum_fourier() {
    double a = -1.0, b = 1.0;
    size_t N = 127; // 128 точек
    auto basis_x = X_State::make_basis(a, b, N);
    State<X_State> vacuum_x(basis_x);
    double sigma = 1.0;
    for (size_t i = 0; i <= N; ++i) {
        double x = basis_x[i]->get_x();
        vacuum_x[basis_x[i]] = std::exp(-0.5 * x * x / (sigma * sigma));
    }
    vacuum_x.normalize();

    // Проверка нормы в координатном представлении
    double norm_x = 0.0;
    for (size_t i = 0; i <= N; ++i) norm_x += std::norm(vacuum_x[basis_x[i]]);
    assert(std::abs(norm_x - 1.0) < 1e-10);

    auto vacuum_p = fft_x_to_p(vacuum_x);

    // Проверка нормы в импульсном представлении
    auto basis_p = vacuum_p.get_basis();
    double norm_p = 0.0;
    for (size_t i = 0; i < basis_p.size(); ++i) norm_p += std::norm(vacuum_p[basis_p[i]]);
    assert(std::abs(norm_p - 1.0) < 1e-10);

    // Пик распределения должен быть вблизи p = 0
    double max_amp = 0.0;
    size_t max_idx = 0;
    for (size_t i = 0; i < basis_p.size(); ++i) {
        double amp = std::norm(vacuum_p[basis_p[i]]);
        if (amp > max_amp) {
            max_amp = amp;
            max_idx = i;
        }
    }
    double p_max = basis_p[max_idx]->get_p();
    assert(std::abs(p_max) < 0.1);

    std::cout << "test_vacuum_fourier passed\n";
}

int main() {
    test_basis_creation();
    test_fourier_unitarity();
    test_vacuum_fourier();
    std::cout << "All tests passed.\n";
    return 0;
}
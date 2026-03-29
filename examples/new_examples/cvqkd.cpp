#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Параметры протокола
    const double V_A = 1.0;          // дисперсия модуляции Алисы
    const double T = 0.5;             // пропускание канала
    const double xi = 0.01;           // избыточный шум (в ед. вакуумного)
    const double v_el = 0.001;        // электронный шум детектора
    const size_t total_pulses = 10000000;

    // Распределение импульсов по процессам
    size_t local_pulses = total_pulses / world_size;
    size_t remainder = total_pulses % world_size;
    if (rank < remainder) local_pulses++;

    // Генераторы случайных чисел
    std::random_device rd;
    std::mt19937 gen(rd() + rank);
    std::normal_distribution<double> mod_dist(0.0, std::sqrt(V_A)); // для амплитуд
    std::uniform_int_distribution<int> basis_dist(0, 1);
    double meas_var = 1.0 + xi + v_el; // дисперсия шума измерения
    std::normal_distribution<double> meas_noise(0.0, std::sqrt(meas_var));

    // Локальные данные
    std::vector<double> alice_x(local_pulses);
    std::vector<double> alice_p(local_pulses);
    std::vector<int> alice_basis(local_pulses);
    std::vector<double> bob_x(local_pulses);
    std::vector<double> bob_p(local_pulses);
    std::vector<int> bob_basis(local_pulses);

    for (size_t i = 0; i < local_pulses; ++i) {
        int basis = basis_dist(gen);
        double x0, p0;
        if (basis == 0) {
            x0 = mod_dist(gen);
            p0 = 0.0;
        } else {
            x0 = 0.0;
            p0 = mod_dist(gen);
        }
        alice_basis[i] = basis;
        alice_x[i] = x0;
        alice_p[i] = p0;

        // Ожидаемое значение у Боба после канала
        double expect = std::sqrt(T) * (basis == 0 ? x0 : p0);
        // Результат измерения с шумом
        double measurement = expect + meas_noise(gen);

        bob_basis[i] = basis;
        if (basis == 0) {
            bob_x[i] = measurement;
            bob_p[i] = 0.0;
        } else {
            bob_p[i] = measurement;
            bob_x[i] = 0.0;
        }
    }

    // Сбор данных на корневом процессе
    std::vector<double> global_alice_x, global_alice_p, global_bob_x, global_bob_p;
    std::vector<int> global_alice_basis, global_bob_basis;
    if (rank == 0) {
        global_alice_x.resize(total_pulses);
        global_alice_p.resize(total_pulses);
        global_bob_x.resize(total_pulses);
        global_bob_p.resize(total_pulses);
        global_alice_basis.resize(total_pulses);
        global_bob_basis.resize(total_pulses);
    }

    MPI_Gather(alice_x.data(), local_pulses, MPI_DOUBLE,
               global_alice_x.data(), local_pulses, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(alice_p.data(), local_pulses, MPI_DOUBLE,
               global_alice_p.data(), local_pulses, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(bob_x.data(), local_pulses, MPI_DOUBLE,
               global_bob_x.data(), local_pulses, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(bob_p.data(), local_pulses, MPI_DOUBLE,
               global_bob_p.data(), local_pulses, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(alice_basis.data(), local_pulses, MPI_INT,
               global_alice_basis.data(), local_pulses, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(bob_basis.data(), local_pulses, MPI_INT,
               global_bob_basis.data(), local_pulses, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Разделяем данные по совпавшим базисам
        std::vector<double> matched_x, matched_p;
        for (size_t i = 0; i < total_pulses; ++i) {
            if (global_alice_basis[i] == global_bob_basis[i]) {
                if (global_alice_basis[i] == 0) {
                    matched_x.push_back(global_alice_x[i]);
                    matched_x.push_back(global_bob_x[i]);
                } else {
                    matched_p.push_back(global_alice_p[i]);
                    matched_p.push_back(global_bob_p[i]);
                }
            }
        }

        size_t count_x = matched_x.size() / 2;
        size_t count_p = matched_p.size() / 2;

        // Вычисление статистик для X
        double sum_xA = 0.0, sum_xB = 0.0, sum_xA2 = 0.0, sum_xB2 = 0.0, sum_xAxB = 0.0;
        for (size_t j = 0; j < count_x; ++j) {
            double a = matched_x[2*j];
            double b = matched_x[2*j+1];
            sum_xA += a;
            sum_xB += b;
            sum_xA2 += a * a;
            sum_xB2 += b * b;
            sum_xAxB += a * b;
        }
        double mean_xA = sum_xA / count_x;
        double mean_xB = sum_xB / count_x;
        double var_xA = (sum_xA2 / count_x - mean_xA * mean_xA) * count_x / (count_x - 1);
        double var_xB = (sum_xB2 / count_x - mean_xB * mean_xB) * count_x / (count_x - 1);
        double cov_x = (sum_xAxB / count_x - mean_xA * mean_xB) * count_x / (count_x - 1);

        // Оценка SNR и взаимной информации
        double SNR_x = cov_x * cov_x / (var_xA * (var_xB - cov_x * cov_x / var_xA));
        double I_AB_x = 0.5 * std::log2(1 + SNR_x);

        // Теоретическое значение для проверки
        double SNR_theory = T * V_A / (1.0 + xi + v_el);
        double I_theory = 0.5 * std::log2(1 + SNR_theory);

        std::cout << "CV-QKD Simulation Results\n";
        std::cout << "Total pulses: " << total_pulses << "\n";
        std::cout << "Matched X events: " << count_x << "\n";
        std::cout << "Matched P events: " << count_p << "\n";
        std::cout << "Estimated V_A (X): " << var_xA << " (theoretical " << V_A << ")\n";
        std::cout << "Estimated V_B (X): " << var_xB << " (theoretical " << T*V_A + 1 + xi + v_el << ")\n";
        std::cout << "Estimated covariance: " << cov_x << " (theoretical " << std::sqrt(T)*V_A << ")\n";
        std::cout << "SNR (estimated): " << SNR_x << "\n";
        std::cout << "Mutual information I_AB (estimated): " << I_AB_x << " bits/pulse\n";
        std::cout << "Theoretical I_AB: " << I_theory << " bits/pulse\n";
    }

    MPI_Finalize();
    return 0;
}
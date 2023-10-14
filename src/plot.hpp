#pragma once
#include "dynamic.hpp"
#include "state.hpp"
#include <string>
#include <map>
#include <algorithm>

#ifdef ENABLE_MATPLOTLIB

#include "matplotlibcpp.hpp"

namespace QComputations {

// NEED REWORK

// !!!!!!!!!!!!!! ЕСТЬ ОДНА ЗАЛУПА. ПРИ ИСПОЛЬЗОВАНИИ Parallel_QME команду make_figure 
// !!!!!!!!!!!!!! использовать только ДО ВЫЗОВА Parallel_QME, иначе free invalid pointer по причине - хз,
// !!!!!!!!!!!!!! интерпретатор питона тупа шлёт нахер.
namespace matplotlib {
    void make_figure(size_t x = 0, size_t y = 0, size_t dpi = QConfig::instance().dpi());
    void probs_to_plot(const Evolution::Probs& probs,
                       const std::vector<double>& time_vec,
                       const std::set<State>& basis,
                       std::vector<std::map<std::string, std::string>> keywords = {});
    void probs_to_plot(const Evolution::Probs& probs,
                       const std::vector<double>& time_vec,
                       const std::vector<std::string>& basis_str,
                       std::vector<std::map<std::string, std::string>> keywords = {});
 #ifdef ENABLE_MPI
 #ifdef ENABLE_CLUSTER
    void probs_to_plot(const Evolution::BLOCKED_Probs& probs,
                       const std::vector<double>& time_vec,
                       const std::set<State>& basis,
                       std::vector<std::map<std::string, std::string>> keywords = {});
    void probs_to_plot(const Evolution::BLOCKED_Probs& probs,
                       const std::vector<double>& time_vec,
                       const std::vector<std::string>& basis_str,
                       std::vector<std::map<std::string, std::string>> keywords = {});
    void probs_in_cavity_to_plot(const Evolution::BLOCKED_Probs& probs,
                                 const std::vector<double>& time_vec,
                                 const std::set<State>& basis,
                                 size_t cavity_id,
                                 std::vector<std::map<std::string, std::string>> keywords = {});
 #endif
 #endif
    void rho_probs_to_plot(const Evolution::Probs& probs,
                           const std::vector<double>& time_vec,
                           const std::set<State>& basis,
                           std::vector<std::map<std::string, std::string>> keywords = {});
    void rho_diag_to_plot(const Evolution::Probs& probs,
                          const std::vector<double>& time_vec,
                          const std::set<State>& basis,
                          std::vector<std::map<std::string, std::string>> keywords = {});
    void rho_subdiag_to_plot(const Evolution::Probs& probs,
                             const std::vector<double>& time_vec,
                             const std::set<State>& basis,
                             std::vector<std::map<std::string, std::string>> keywords = {});
    void probs_in_cavity_to_plot(const Evolution::Probs& probs,
                                const std::vector<double>& time_vec,
                                const std::set<State>& basis,
                                size_t cavity_id,
                                std::vector<std::map<std::string, std::string>> keywords = {});

    void show(bool is_block = true);
    void savefig(const std::string& filename, size_t dpi = QConfig::instance().dpi());
    void grid(bool is_enable = true);
}

} // namespace QComputations

#endif

namespace QComputations {

namespace plotly {

}

namespace matlab {

}

} // namespace QComputations

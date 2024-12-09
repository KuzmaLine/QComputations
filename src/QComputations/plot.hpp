#pragma once
#include <algorithm>
#include <map>
#include <string>

#include "dynamic.hpp"
#include "state.hpp"

namespace QComputations {

#ifndef ENABLE_CLUSTER

void hamiltonian_to_file(const std::string& filename, const Hamiltonian& H, std::string dir = "");

void basis_to_file(const std::string& filename, const BasisType<Basis_State>& basis, std::string dir = "");

void time_vec_to_file(const std::string& filename, const std::vector<double>& time_vec, std::string dir = "");

void probs_to_file(const std::string& filename, const Probs& probs, std::string dir = "");

void make_probs_files(const Hamiltonian& H,
                      const Probs& probs,
                      const std::vector<double>& time_vec,
                      const BasisType<Basis_State>& basis,
                      std::string dir = "");

#endif

#ifdef ENABLE_MPI
#ifdef ENABLE_CLUSTER

void hamiltonian_to_file(const std::string& filename,
                         const BLOCKED_Hamiltonian& H,
                         std::string dir = "",
                         ILP_TYPE main_rank = 0);
void probs_to_file(const std::string& filename, const BLOCKED_Probs& probs, std::string dir = "");
void plot_from_files(const std::string& plotname,
                     std::string dir,
                     const std::string& python_script_path = QConfig::instance().python_script_path());
void basis_to_file(const std::string& filename,
                   const BasisType<Basis_State>& basis,
                   std::string dir = "",
                   ILP_TYPE main_rank = 0);
void time_vec_to_file(const std::string& filename,
                      const std::vector<double>& time_vec,
                      std::string dir = "",
                      ILP_TYPE main_rank = 0);
void probs_to_file(const std::string& filename, const Probs& probs, std::string dir = "", ILP_TYPE main_rank = 0);
void hamiltonian_to_file(const std::string& filename,
                         const Hamiltonian& H,
                         std::string dir = "",
                         ILP_TYPE main_rank = 0);

void make_probs_files(const Hamiltonian& H,
                      const Probs& probs,
                      const std::vector<double>& time_vec,
                      const BasisType<Basis_State>& basis,
                      std::string dir = "",
                      ILP_TYPE main_rank = 0);

void make_probs_files(const BLOCKED_Hamiltonian& H,
                      const BLOCKED_Probs& probs,
                      const std::vector<double>& time_vec,
                      const BasisType<Basis_State>& basis,
                      std::string dir = "",
                      ILP_TYPE main_rank = 0);

void make_plot(const std::string& plotname,
               const BLOCKED_Hamiltonian& H,
               const BLOCKED_Probs& probs,
               const std::vector<double>& time_vec,
               const BasisType<Basis_State>& basis,
               std::string dir);

#endif
#endif

#ifdef ENABLE_MATPLOTLIB

namespace matplotlib {
void make_figure(size_t x = 0, size_t y = 0, size_t dpi = QConfig::instance().dpi());
void probs_to_plot(const Probs& probs,
                   const std::vector<double>& time_vec,
                   const BasisType<Basis_State>& basis,
                   std::vector<std::map<std::string, std::string>> keywords = {});
void probs_to_plot(const Probs& probs,
                   const std::vector<double>& time_vec,
                   const std::vector<std::string>& basis_str,
                   std::vector<std::map<std::string, std::string>> keywords = {});
template <typename T, typename V>
void plot(const std::vector<T>& x, const std::vector<V>& y, std::map<std::string, std::string> keywords = {});
void title(const std::string& name);
void xlabel(const std::string& name);
void ylabel(const std::string& name);
void zlabel(const std::string& name);

void surface(const std::vector<std::vector<double>>& x,
             const std::vector<std::vector<double>>& y,
             const std::vector<std::vector<double>>& z,
             std::map<std::string, std::string> keywords = {});
#ifdef ENABLE_MPI
#ifdef ENABLE_CLUSTER
void probs_to_plot(const BLOCKED_Probs& probs,
                   const std::vector<double>& time_vec,
                   const BasisType<Basis_State>& basis,
                   std::vector<std::map<std::string, std::string>> keywords = {});
void probs_to_plot(const BLOCKED_Probs& probs,
                   const std::vector<double>& time_vec,
                   const std::vector<std::string>& basis_str,
                   std::vector<std::map<std::string, std::string>> keywords = {});
void probs_in_cavity_to_plot(const BLOCKED_Probs& probs,
                             const std::vector<double>& time_vec,
                             const BasisType<Basis_State>& basis,
                             size_t cavity_id,
                             std::vector<std::map<std::string, std::string>> keywords = {});
#endif
#endif
void rho_probs_to_plot(const Probs& probs,
                       const std::vector<double>& time_vec,
                       const BasisType<Basis_State>& basis,
                       std::vector<std::map<std::string, std::string>> keywords = {});
void rho_diag_to_plot(const Probs& probs,
                      const std::vector<double>& time_vec,
                      const BasisType<Basis_State>& basis,
                      std::vector<std::map<std::string, std::string>> keywords = {});
void rho_subdiag_to_plot(const Probs& probs,
                         const std::vector<double>& time_vec,
                         const BasisType<Basis_State>& basis,
                         std::vector<std::map<std::string, std::string>> keywords = {});
void probs_in_cavity_to_plot(const Probs& probs,
                             const std::vector<double>& time_vec,
                             const BasisType<Basis_State>& basis,
                             size_t cavity_id,
                             std::vector<std::map<std::string, std::string>> keywords = {});

void show(bool is_block = true);
void savefig(const std::string& filename, size_t dpi = QConfig::instance().dpi());
void grid(bool is_enable = true);
void legend();
}  // namespace matplotlib

#endif

namespace plotly {}

namespace matlab {}

}  // namespace QComputations

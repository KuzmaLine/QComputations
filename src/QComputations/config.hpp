#pragma once
#include <complex>
#include <iostream>

namespace QComputations {

// Рудимент - УБРАТЬ (Осторожно с private переменной QConfig)
enum MULTIPLY_ALGS { COMMON_MODE = 0 };

enum FUNCTION_QME { RUNGE_KUTT_4 = 121, RUNGE_KUTT_2 = 122 };
namespace {
const std::string angle_bracket_right = "\u29FD";
enum FIG_PARAMS { FIG_WIDTH = 19, FIG_HEIGHT = 10, DPI = 80 };
constexpr double h_default = 1;
constexpr double w_default = 1;
constexpr double g_default = 0.01;
constexpr double waveguides_length_default = 0;
constexpr double waveguides_amplitude_default = 0;
constexpr int max_photons_default = 1;

constexpr double eps_default = 1e-12;
constexpr int width_default = 15;

constexpr int csv_max_number_size_default = 21;
constexpr int csv_num_accuracy_default = 16;

const std::string python_script_path_default = "seaborn_plot.py";
constexpr FUNCTION_QME qme_algorithm_default = RUNGE_KUTT_2;
constexpr size_t exp_accuracy_default = 10;
}  // namespace

class QConfig {
   public:
    QConfig(const QConfig&) = delete;
    void operator=(const QConfig&) = delete;

    static QConfig& instance() {
        static QConfig instance;
        return instance;
    }

    void set_h(double h) { h_ = h; }
    void set_max_photons(int max_photons) { max_photons_ = max_photons; }
    void set_w(double w) { w_ = w; }
    void set_g(double g) { g_ = g; }
    void set_fig_width(int fig_width) { fig_width_ = fig_width; }
    void set_fig_height(int fig_height) { fig_height_ = fig_height; }
    void set_dpi(int dpi) { dpi_ = dpi; }
    void set_multiply_mode(MULTIPLY_ALGS MULTIPLY_MODE) { MULTIPLY_MODE_ = MULTIPLY_MODE; }
    void set_eps(double eps) { eps_ = eps; }
    void set_width(int width) { width_ = width; }
    void set_waveguides_length(double waveguides_length) { wavegiudes_length_ = waveguides_length; }
    void set_csv_max_number_size(int csv_max_number_size) { csv_max_number_size_ = csv_max_number_size; }
    void set_csv_num_accuracy(int csv_num_accuracy) { csv_num_accuracy_ = csv_num_accuracy; }
    void set_qme_algorithm(const FUNCTION_QME alg) { qme_algorithm_ = alg; }
    void set_exp_accuracy(const size_t exp_accuracy) { exp_accuracy_ = exp_accuracy; }

    double h() const { return h_; }
    int max_photons() const { return max_photons_; }
    double w() const { return w_; }
    double g() const { return g_; }
    int fig_width() const { return fig_width_; }
    int fig_height() const { return fig_height_; }
    int dpi() const { return dpi_; }
    MULTIPLY_ALGS MULTIPLY_MODE() const { return MULTIPLY_MODE_; }
    double eps() const { return eps_; }
    int width() const { return width_; }
    double waveguides_length() const { return wavegiudes_length_; }
    double waveguides_amplitude() const { return wavegiudes_amplitude_; }
    int csv_max_number_size() const { return csv_max_number_size_; }
    int csv_num_accuracy() const { return csv_num_accuracy_; }
    std::string python_script_path() const { return python_script_path_; }
    FUNCTION_QME qme_algorithm() const { return qme_algorithm_; }
    size_t exp_accuracy() const { return exp_accuracy_; }

    void show() const {
        std::cout << "CONFIG PARAMS: " << std::endl;
        std::cout << " h - " << h_ << std::endl;
        std::cout << " w - " << w_ << std::endl;
        std::cout << " g - " << g_ << std::endl;
        std::cout << " eps - " << eps_ << std::endl;
        std::cout << " fig_width - " << fig_width_ << std::endl;
        std::cout << " fig_height - " << fig_height_ << std::endl;
        std::cout << " dpi - " << dpi_ << std::endl;
        std::cout << " print width - " << width_ << std::endl;
        std::cout << " MULTIPLY_MODE - " << (MULTIPLY_MODE_ == COMMON_MODE ? "COMMON_MODE" : "") << std::endl;
    }

   private:
    QConfig() {}
    ~QConfig() {}
    MULTIPLY_ALGS MULTIPLY_MODE_ = COMMON_MODE;
    int fig_width_ = int(FIG_WIDTH);
    int fig_height_ = int(FIG_HEIGHT);
    int dpi_ = int(DPI);

    int csv_max_number_size_ = csv_max_number_size_default;
    int csv_num_accuracy_ = csv_num_accuracy_default;

    int width_ = width_default;

    double eps_ = eps_default;

    int max_photons_ = max_photons_default;
    double wavegiudes_length_ = waveguides_length_default;
    double wavegiudes_amplitude_ = waveguides_amplitude_default;

    double h_ = h_default;
    double w_ = w_default;
    double g_ = g_default;

    size_t exp_accuracy_ = exp_accuracy_default;

    std::string python_script_path_ = python_script_path_default;
    FUNCTION_QME qme_algorithm_ = qme_algorithm_default;
};

}  // namespace QComputations
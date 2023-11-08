#pragma once
#include <complex>
#include <iostream>

namespace QComputations {

enum MULTIPLY_ALGS {COMMON_MODE = 0, CANNON_MODE = 1, DIM_MODE = 2, P_GEMM_MODE = 3};
namespace {
    const std::string angle_bracket_right = "\u29FD";
    enum FIG_PARAMS {FIG_WIDTH = 960, FIG_HEIGHT = 540, DPI = 80};
    constexpr double h_default = 1;
    constexpr double w_default = 1;
    constexpr double g_default = 0.01;
    constexpr double waveguides_length_default = 0;
    constexpr double waveguides_amplitude_default = 0;
    constexpr int E_LEVELS_COUNT_DEFAULT = 2;

    constexpr double eps_default = 1e-12;
    constexpr int width_default = 15;

    constexpr int csv_max_number_size_default = 21;
    constexpr int csv_num_accuracy_default = 18;

    constexpr bool is_freq_display_default = true;
    constexpr bool is_sequence_default = false;
    const std::string state_format_default = "|$N$W$!;$M>";
    const std::string state_delimeter_default = ",";
    const std::string excitation_state_format_default = "$S|$N>$W$!{$M" + angle_bracket_right;
}

class QConfig {
    public:
        QConfig(const QConfig&) = delete;
        void operator=(const QConfig&) = delete;

        static QConfig& instance() {
            static QConfig instance;
            return instance;
        }

        void set_h(double h) { h_ = h; }
        void set_w(double w) { w_ = w; }
        void set_g(double g) { g_ = g; }
        void set_fig_width(int fig_width) { fig_width_ = fig_width; }
        void set_fig_height(int fig_height) { fig_height_ = fig_height; }
        void set_dpi(int dpi) { dpi_ = dpi; }
        void set_multiply_mode(MULTIPLY_ALGS MULTIPLY_MODE) { MULTIPLY_MODE_ = MULTIPLY_MODE; }
        void set_eps(double eps) { eps_ = eps; }
        void set_width(int width) { width_ = width; }
        void set_E_LEVELS_COUNT(int E_LEVELS_COUNT) { E_LEVELS_COUNT_ = E_LEVELS_COUNT; }
        void set_waveguides_length(double waveguides_length) { wavegiudes_length_ = waveguides_length; }
        void set_csv_max_number_size(int csv_max_number_size) { csv_max_number_size_ = csv_max_number_size; }
        void set_csv_num_accuracy(int csv_num_accuracy) { csv_num_accuracy_ = csv_num_accuracy; }
        void set_is_sequence_state(bool is_sequence_state) { is_sequence_ = is_sequence_state; }
        void set_is_freq_display(bool is_freq_display) { is_freq_display_ = is_freq_display; }
        void set_state_format(const std::string& state_format) { state_format_ = state_format; }
        void set_state_delimeter(const std::string& state_delimeter) { state_delimeter_ = state_delimeter; }
        void set_excitation_state_format(const std::string& excitation_state_format) { excitation_state_format_ = excitation_state_format; }

        double h() const { return h_; }
        double w() const { return w_; }
        double g() const { return g_; }
        int fig_width() const { return fig_width_; }
        int fig_height() const { return fig_height_; }
        int dpi() const { return dpi_; }
        MULTIPLY_ALGS MULTIPLY_MODE() const { return MULTIPLY_MODE_; }
        double eps() const { return eps_; }
        int width() const { return width_; }
        int E_LEVELS_COUNT() const { return E_LEVELS_COUNT_; }
        double waveguides_length() const { return wavegiudes_length_; }
        double waveguides_amplitude() const { return wavegiudes_amplitude_; }
        int csv_max_number_size() const { return csv_max_number_size_; }
        int csv_num_accuracy() const { return csv_num_accuracy_; }
        bool is_sequence_state() const { return is_sequence_; }
        bool is_freq_display() const { return is_freq_display_; }
        std::string state_format() const { return state_format_; }
        std::string state_delimeter() const { return state_delimeter_; }
        std::string excitation_state_format() const { return excitation_state_format_; }

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
        
        int E_LEVELS_COUNT_ = E_LEVELS_COUNT_DEFAULT;
        double wavegiudes_length_ = waveguides_length_default;
        double wavegiudes_amplitude_ = waveguides_amplitude_default;
    
        double h_ = h_default;
        double w_ = w_default;
        double g_ = g_default;

        bool is_sequence_ = is_sequence_default;
        bool is_freq_display_ = is_freq_display_default;
        std::string state_format_ = state_format_default;
        std::string state_delimeter_ = state_delimeter_default;
        std::string excitation_state_format_ = excitation_state_format_default;
};

} // namespace QComputations
#pragma once
#include <complex>


namespace QComputations {

/*
namespace config {
//  Quantum parametrs
    constexpr double h = 1;
    constexpr double w = 1;
    constexpr double g = 0.01;

// matplotlib figure() parametrs
    constexpr double fig_width = 960;
    constexpr double fig_height = 540;
    constexpr size_t dpi = 80;

// printing parametrs
    constexpr int WIDTH = 15;

// is_zero
    constexpr double eps = 10e-12;

    constexpr int COMMON_MODE = 0;
    constexpr int CANNON_MODE = 1;
    constexpr int DIM_MODE = 2;
    constexpr int P_GEMM_MODE = 3;

#ifdef ENABLE_CLUSTER
    static int MULTIPLY_MODE = COMMON_MODE;
#else
    constexpr int MULTIPLY_MODE = COMMON_MODE;
#endif
}
*/

enum MULTIPLY_ALGS {COMMON_MODE = 0, CANNON_MODE = 1, DIM_MODE = 2, P_GEMM_MODE = 3};
namespace {
    enum FIG_PARAMS {FIG_WIDTH = 960, FIG_HEIGHT = 540, DPI = 80};
    constexpr double h_default = 1;
    constexpr double w_default = 1;
    constexpr double g_default = 0.01;
    constexpr double waveguides_length_default = 0;
    constexpr double waveguides_amplitude_default = 0;
    constexpr int E_LEVELS_COUNT_DEFAULT = 2;
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
        
        int width_ = 15;

        double eps_ = 1e-12;
        
        int E_LEVELS_COUNT_ = E_LEVELS_COUNT_DEFAULT;
        double wavegiudes_length_ = waveguides_length_default;
        double wavegiudes_amplitude_ = waveguides_amplitude_default;
        double h_ = h_default;
        double w_ = w_default;
        double g_ = g_default;
};

} // namespace QComputations
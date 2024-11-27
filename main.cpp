#include <cstdio>
#include <vector>
#include <cmath>

#include "librosa/librosa.h"

typedef std::vector<std::vector<std::complex<float>>> FFT_RESULT;

FFT_RESULT transpose_f(const FFT_RESULT& f) {
    FFT_RESULT t(f[0].size(), std::vector<std::complex<float>>(f.size()));
    for (size_t i = 0; i < t.size(); i++) {
        for (size_t j = 0; j < t[0].size(); j++) {
            t[i][j] = f[j][i];
        }
    }
    return t;
}

FFT_RESULT normalize(FFT_RESULT& f) {
    int frame_length = f[0].size();
    float norm_ratio = 1.0f / sqrtf(frame_length);
    for (size_t i = 0; i < f.size(); i++) {
        for (size_t n = 0; n < f[0].size(); n++) {
            f[i][n] *= norm_ratio;
        }
    }
    return f;
}

void save_fft_result(const FFT_RESULT& f, const char* filename) {
    FILE* fp = fopen(filename, "wb");
    for (size_t i = 0; i < f.size(); i++) {
        for (size_t j = 0; j < f[0].size(); j++) {
            float real = f[i][j].real();
            float imag = f[i][j].imag();
            fwrite(&real, sizeof(float), 1, fp);
            fwrite(&imag, sizeof(float), 1, fp);
        }
    }
    fclose(fp);
}

void save_ifft_result(const std::vector<float>& inv_f, const char* filename) {
    FILE* fp = fopen(filename, "wb");
    fwrite(inv_f.data(), sizeof(float), inv_f.size(), fp);
    fclose(fp);
}

int main(int argc, char** argv) {
    int n_fft = 256;
    int hop_length = n_fft / 4;
    bool normalized = true;

    std::vector<float> x(4096);
    FILE* fp = fopen("data.bin", "rb");
    fread(x.data(), sizeof(float), x.size(), fp);
    fclose(fp);

    auto f = transpose_f(librosa::Feature::stft(x, n_fft, hop_length, "hann", true, "reflect", normalized));
    // normalize(f);
    printf("f.shape = %ld %ld 2\n", f.size(), f[0].size());
    save_fft_result(f, "cpp_f.bin");

    auto inv_f = librosa::Feature::istft(f, n_fft, hop_length, "hann", true, "reflect", normalized);
    save_ifft_result(inv_f, "cpp_inv_f.bin");
    return 0;
}
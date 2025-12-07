#include <cstdio>
#include <vector>
#include <cmath>

#include "librosa/librosa.h"

typedef std::vector<std::vector<std::complex<float>>> FFT_RESULT;

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

void load_fft_result(std::vector<FFT_RESULT>& f, const char* filename, int B, int Fr, int T) {
    FILE* fp = fopen(filename, "rb");
    for (int b = 0; b < B; b++) {
        FFT_RESULT res(Fr, std::vector<std::complex<float>>(T));
        for (int i = 0; i < Fr; i++) {
            for (int n = 0; n < T; n++) {
                float real, imag;
                fread(&real, sizeof(float), 1, fp);
                fread(&imag, sizeof(float), 1, fp);
                std::complex<float> val(real, imag);
                res[i][n] = val;
            }
        }
        f.push_back(res);
    }
    fclose(fp);
}

int main(int argc, char** argv) {
    int n_fft = 256;
    int hop_length = n_fft / 4;
    bool normalized = true;

    std::vector<float> x;
    FILE* fp = fopen("data.bin", "rb");
    int size = 0;
    fseek(fp, 0, SEEK_END);
    size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    x.resize(size / sizeof(float));
    fread(x.data(), sizeof(float), x.size(), fp);
    fclose(fp);

    auto f = librosa::Feature::stft(x, n_fft, hop_length, "hann", true, "reflect", normalized);
    save_fft_result(f, "cpp_f.bin");

    auto inv_f = librosa::Feature::istft(f, n_fft, hop_length, "hann", true, "reflect", normalized);
    save_ifft_result(inv_f, "cpp_inv_f.bin");

    return 0;
}
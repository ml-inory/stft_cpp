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
    // int n_fft = 256;
    // int hop_length = n_fft / 4;
    // bool normalized = true;

    // std::vector<float> x(4096);
    // FILE* fp = fopen("data.bin", "rb");
    // fread(x.data(), sizeof(float), x.size(), fp);
    // fclose(fp);

    // auto f = librosa::Feature::stft(x, n_fft, hop_length, "hann", true, "reflect", normalized);
    // // normalize(f);
    // printf("f.shape = %ld %ld 2\n", f.size(), f[0].size());
    // save_fft_result(f, "cpp_f.bin");

    // auto inv_f = librosa::Feature::istft(f, n_fft, hop_length, "hann", true, "reflect", normalized);
    // save_ifft_result(inv_f, "cpp_inv_f.bin");


    int Fr = 2049;
    int T = 340;
    int n_fft = 4096;
    int hop_length = n_fft / 4;
    bool normalized = true;

    std::vector<FFT_RESULT> f;
    load_fft_result(f, "batch_z.bin", 8, 2049, 340);

    // FILE* fp = fopen("batch_z.bin", "rb");
    // for (int i = 0; i < 8; i++) {
    //     std::string name = "cpp_inv_f_" + std::to_string(i) + ".bin";
    //     // FFT_RESULT res(Fr, std::vector<std::complex<float>>(T));
    //     // for (int k = 0; k < Fr; k++) {
    //     //     for (int n = 0; n < T; n++) {
    //     //         float real, imag;
    //     //         fread(&real, sizeof(float), 1, fp);
    //     //         fread(&imag, sizeof(float), 1, fp);
    //     //         std::complex<float> val(real, imag);
    //     //         res[k][n] = val;
    //     //     }
    //     // }
        
    //     // load_fft_result(f, "batch_z.bin", 8, 2049, 340);

    //     auto inv_f = librosa::Feature::istft(f[i], n_fft, hop_length, "hann", true, "reflect", normalized);
    //     save_ifft_result(inv_f, name.c_str());
    // }
    // fclose(fp);

    // load_fft_result(f, "batch_z.bin", 8, 2049, 340);
    // FFT_RESULT sub_f = f[2];
    // auto inv_f = librosa::Feature::istft(sub_f, 2 * Fr - 2, hop_length, "hann", true, "reflect", normalized);
    // save_ifft_result(inv_f, "cpp_inv_f.bin");
    

    for (int i = 0; i < 8; i++) {
        std::string name = "cpp_inv_f_" + std::to_string(i) + ".bin";
        auto inv_f = librosa::Feature::istft(f[i], n_fft, hop_length, "hann", true, "reflect", normalized);
        save_ifft_result(inv_f, name.c_str());
    }

    // // printf("f.size() = %ld\n", f.size());
    // for (int i = 0; i < 8; i++) {
    //     // printf("f[%d].size() = %ld\n", i, f[i].size());
    //     // printf("f[%d][0].size() = %ld\n", i, f[i][0].size());
    //     std::string name = "f_" + std::to_string(i) + ".bin";
    //     save_fft_result(f[i], name.c_str());
    // }
    
    // for (int i = 0; i < 8; i++) {
    //     load_fft_result(f, "batch_z.bin", 8, 2049, 340);
    //     auto inv_f = librosa::Feature::istft(f[0], n_fft, hop_length, "hann", true, "reflect", normalized);
    //     std::string name = "reload_" + std::to_string(i) + ".bin";
    //     save_ifft_result(inv_f, name.c_str());
    // }
    

    return 0;
}
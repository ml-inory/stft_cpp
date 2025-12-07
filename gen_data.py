import numpy as np
import torch

# 创建简单的测试信号
x = np.sin(2 * np.pi * np.linspace(0, 1, 44100))[np.newaxis, :].astype(np.float32)
print(f"x.shape: {x.shape}")
x.tofile("data.bin")

n_fft = 256
hop_length = n_fft // 4

f = torch.stft(
    torch.from_numpy(x),
    n_fft,
    hop_length,
    win_length=n_fft,
    window=torch.hann_window(n_fft),
    center=True,
    pad_mode="reflect",
    normalized=True,
    return_complex=True
)

f = torch.view_as_real(f).numpy().flatten()
print(f"gt stft.shape: {f.shape}")
f.tofile("gt_stft.bin")
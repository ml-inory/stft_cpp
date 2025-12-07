import torch
import numpy as np


def cos_sim(A, B):
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    cosine_sim = dot_product / (norm_A * norm_B)
    return cosine_sim


n_fft = 256
hop_length = n_fft // 4

x = np.fromfile("data.bin", dtype=np.float32)[np.newaxis, :]
print(f"x.shape: {x.shape}")

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

inv_f = torch.istft(
    f,
    n_fft,
    hop_length,
    win_length=n_fft,
    window=torch.hann_window(n_fft),
    center=True,
    normalized=True,
    return_complex=False
)

f = torch.view_as_real(f).numpy().flatten()
inv_f = inv_f.numpy().flatten()

print(f"f.shape: {f.shape}")
print(f"inv_f.shape: {inv_f.shape}")

cpp_f = np.fromfile(f"cpp_f.bin", dtype=np.float32)
cpp_inv_f = np.fromfile(f"cpp_inv_f.bin", dtype=np.float32)

print(f"cpp_f.shape: {cpp_f.shape}")
print(f"cpp_inv_f.shape: {cpp_inv_f.shape}")

gt_f = np.fromfile(f"gt_stft.bin", dtype=np.float32)
sim = cos_sim(gt_f, f)
print(f"cos sim of gt stft: {sim}")

sim = cos_sim(cpp_f, f)
print(f"cos sim of stft: {sim}")
sim = cos_sim(cpp_inv_f, inv_f)
print(f"cos sim of istft: {sim}")

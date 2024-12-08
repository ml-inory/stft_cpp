import torch
import numpy as np
from scipy.signal import get_window
from librosa import util as librosa_util
from librosa.util import pad_center, tiny
from numpy.linalg import norm


def cos_sim(a, b):
    return (a @ b.T) / (norm(a)*norm(b))

n_fft = 256
hop_length = n_fft // 4
x = np.fromfile("data.bin", dtype=np.float32)
L = x.shape[0]
# print(f"input shape: {x.shape}")
# f = torch.stft(
#     torch.from_numpy(x),
#     n_fft,
#     hop_length,
#     win_length=n_fft,
#     window=torch.hann_window(n_fft),
#     center=True,
#     pad_mode="reflect",
#     normalized=True,
#     return_complex=False
# )
# f = f.numpy()

# pred = np.fromfile("cpp_f.bin", dtype=np.float32).reshape((n_fft // 2 + 1, 1 + L // hop_length, 2))
# np.testing.assert_allclose(pred, f, atol=1e-6)
# print("Compare pass")

def window_sumsquare(
    window,
    n_frames,
    hop_length=200,
    win_length=800,
    n_fft=800,
    dtype=np.float32,
    norm=None,
):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.
    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.
    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`
    n_frames : int > 0
        The number of analysis frames
    hop_length : int > 0
        The number of samples to advance between frames
    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.
    n_fft : int > 0
        The length of each analysis frame.
    dtype : np.dtype
        The data type of the output
    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm) ** 2
    win_sq = librosa_util.pad_center(win_sq, size=n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample : min(n, sample + n_fft)] += win_sq[
            : max(0, min(n_fft, n - sample))
        ]
    return x

# win = window_sumsquare("hann", 65, hop_length, n_fft=n_fft, win_length=n_fft)
# cpp_win = np.fromfile("cpp_win.bin", dtype=np.float32)
# np.testing.assert_allclose(cpp_win, win, atol=1e-6)


# f = torch.stft(
#     torch.from_numpy(x),
#     n_fft,
#     hop_length,
#     win_length=n_fft,
#     window=torch.hann_window(n_fft),
#     center=True,
#     pad_mode="reflect",
#     normalized=True,
#     return_complex=True
# )
Fr = 2049
n_fft = 4096
hop_length = n_fft // 4
f = np.fromfile("batch_z.bin", dtype=np.float32)
f = f.reshape((8, 2049, 340, 2))
f = torch.view_as_complex(torch.from_numpy(f))
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
inv_f = inv_f.numpy()

# pred = np.fromfile(f"cpp_inv_f.bin", dtype=np.float32)
# # print(pred)
# np.testing.assert_allclose(pred, inv_f[2], atol=1e-6)

# for i in range(8):
#     pred = np.fromfile(f"reload_{i}.bin", dtype=np.float32)
#     np.testing.assert_allclose(pred, inv_f[0], atol=1e-5)
#     # sim = cos_sim(pred, inv_f[i])
#     # assert sim > 0.9
#     print(f"[{i}] Compare pass")

for i in range(8):
    pred = np.fromfile(f"cpp_inv_f_{i}.bin", dtype=np.float32)
    # np.testing.assert_allclose(pred, inv_f[i], atol=1e-5)
    sim = cos_sim(pred, inv_f[i])
    assert sim > 0.9
    print(f"[{i}] Compare pass")

# f = np.fromfile("batch_z.bin", dtype=np.float32)
# f = f.reshape((8, -1))
# for i in range(8):
#     pred = np.fromfile(f"f_{i}.bin", dtype=np.float32)
#     np.testing.assert_allclose(pred, f[i], atol=1e-6)
#     print(f"[{i}] Compare pass")
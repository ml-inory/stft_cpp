import numpy as np

x = np.random.rand(1, 4096).astype(np.float32)
x.tofile("data.bin")
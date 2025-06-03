import sys
import os
import numpy as np
import torch
import time

sys.path.append(os.path.abspath("../build"))
import cumatmul

import time
import numpy as np
import torch

# Matrix size
N = 1024
repeats = 10
results = []

# Generate fixed inputs
A_np = np.random.rand(N, N).astype(np.float32)
B_np = np.random.rand(N, N).astype(np.float32)

# 1. NumPy CPU
cpu_times = []
for _ in range(repeats):
    start = time.time()
    _ = A_np @ B_np
    cpu_times.append(time.time() - start)
avg_cpu_time = sum(cpu_times) / repeats
cpu_gflops = 2 * (N ** 3) / (avg_cpu_time * 1e9)
results.append({
    'Method': 'NumPy CPU',
    'Matrix Size': N,
    'Time (s)': avg_cpu_time,
    'GFLOPs': cpu_gflops
})

# 2. CUDA Naive
C_naive = np.zeros_like(A_np)
torch.cuda.synchronize()
naive_times = []
for _ in range(repeats):
    torch.cuda.synchronize()
    start = time.time()
    cumatmul.naive(A_np, B_np, C_naive, N)
    torch.cuda.synchronize()
    naive_times.append(time.time() - start)
avg_naive_time = sum(naive_times) / repeats
naive_gflops = 2 * (N ** 3) / (avg_naive_time * 1e9)
results.append({
    'Method': 'CUDA Naive',
    'Matrix Size': N,
    'Time (s)': avg_naive_time,
    'GFLOPs': naive_gflops
})

# 3. CUDA Tiled
C_tiled = np.zeros_like(A_np)
torch.cuda.synchronize()
tiled_times = []
for _ in range(repeats):
    torch.cuda.synchronize()
    start = time.time()
    cumatmul.tiled(A_np, B_np, C_tiled, N)
    torch.cuda.synchronize()
    tiled_times.append(time.time() - start)
avg_tiled_time = sum(tiled_times) / repeats
tiled_gflops = 2 * (N ** 3) / (avg_tiled_time * 1e9)
results.append({
    'Method': 'CUDA Tiled',
    'Matrix Size': N,
    'Time (s)': avg_tiled_time,
    'GFLOPs': tiled_gflops
})

# 4. cuBLAS (PyTorch)
A_torch = torch.from_numpy(A_np).cuda()
B_torch = torch.from_numpy(B_np).cuda()
torch.cuda.synchronize()
cublas_times = []
for _ in range(repeats):
    torch.cuda.synchronize()
    start = time.time()
    _ = torch.matmul(A_torch, B_torch)
    torch.cuda.synchronize()
    cublas_times.append(time.time() - start)
avg_cublas_time = sum(cublas_times) / repeats
cublas_gflops = 2 * (N ** 3) / (avg_cublas_time * 1e9)
results.append({
    'Method': 'cuBLAS (PyTorch)',
    'Matrix Size': N,
    'Time (s)': avg_cublas_time,
    'GFLOPs': cublas_gflops
})

# Write to results.txt
with open("/results.txt", "w") as f:
    for r in results:
        f.write(str(r) + "\n")

print("✅ Benchmark completed and written to results.txt")
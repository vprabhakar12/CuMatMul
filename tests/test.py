import sys
import os
sys.path.append(os.path.abspath("../build"))

import numpy as np
import cumatmul
import time
import torch

N = 1024  # Matrix size
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)

runs = 10
results = []

# #NUMPY
# numpy_times = []
# for run in range(runs):
#     C = np.zeros_like(A)
#     start = time.time()
#     C = A @ B
#     elapsed = time.time() - start
#     numpy_times.append(elapsed)
# avg_numpy_time = sum(numpy_times) / runs

# print(f"Numpy MatMul Time: {avg_numpy_time:.4f} seconds")

# NAIVE GPU
naive_times = []
for run in range(runs):
    C = np.zeros_like(A)
    start = time.time()
    cumatmul.naive(A, B, C, N)
    elapsed = time.time() - start
    naive_times.append(elapsed)
avg_naive_time = sum(naive_times) / runs
naive_gflops = 2 * (N ** 3) / (avg_naive_time * 1e9)
results.append({
    'Method': 'CUDA Naive',
    'Matrix Size': N,
    'Time (s)': avg_naive_time,
    'GFLOPs': naive_gflops
})

print(f"CUDA Naive MatMul Time: {avg_naive_time:.4f} seconds")

# TILED SHARED MEM
tiled_times = []
for i in range(runs):
    C_tiled = np.zeros_like(A)
    start = time.time()
    cumatmul.tiled(A, B, C_tiled, N)
    elapsed_tiled = time.time() - start
    tiled_times.append(elapsed_tiled)
avg_tiled_time = sum(tiled_times) / runs
tiled_gflops = 2 * (N ** 3) / (avg_tiled_time * 1e9)
results.append({
    'Method': 'CUDA Tiled',
    'Matrix Size': N,
    'Time (s)': avg_tiled_time,
    'GFLOPs': tiled_gflops
})

print(f"CUDA Tiled MatMul Time: {avg_tiled_time:.4f} seconds")


# CUBLAS
cublas_times = []
A_torch = torch.from_numpy(A).cuda()
B_torch = torch.from_numpy(B).cuda()
torch.cuda.synchronize()
for i in range(runs):
    torch.cuda.synchronize()
    start = time.time()
    _ = torch.matmul(A_torch, B_torch)
    torch.cuda.synchronize()
    elapsed_cublas = time.time() - start
    cublas_times.append(elapsed_cublas)
avg_cublas_time = sum(cublas_times) / runs
cublas_gflops = 2 * (N ** 3) / (avg_cublas_time * 1e9)
results.append({
    'Method': 'cuBLAS (PyTorch)',
    'Matrix Size': N,
    'Time (s)': avg_cublas_time,
    'GFLOPs': cublas_gflops
})

print(f"cuBLAS MatMul Time: {avg_cublas_time:.4f} seconds")

# Write to results.txt
with open("results.txt", "w") as f:
    for r in results:
        f.write(str(r) + "\n")

# print(results)

print("✅ Benchmark completed and written to results.txt")

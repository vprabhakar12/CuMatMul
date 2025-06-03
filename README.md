# CuMatMul - Fast Matrix Multiplication Library 🔢❌➕🔢
CuMatMul is a high-performance matrix multiplication benchmarking suite implemented in CUDA. It includes:

- Naive global memory kernel
- GMEM Coalesced kernel
- Tiled shared memory kernel
- cuBLAS GPU baseline (via PyTorch)

The goal is to understand and optimize GPU kernel performance by comparing different strategies using GFLOPs/s.

## ⚙️ Build Instructions (Windows)
> Prerequisites:
> - CUDA Toolkit (>= 11.0)
> - Visual Studio (with C++ build tools)
> - Python (>= 3.8)
> - pybind11

Use the following command to install the requirements ```pip install -r ./requirements.txt```

In x64 Native Tools Command Prompt:
```
cmake -S . -B build -G "NMake Makefiles"
cmake --build build
```

This builds a Python-accessible module: cumatmul.pyd

## 🧪 Run Benchmark
In the test directory
```python test.py```

This runs matrix multiplication (A: 1024×2048, B: 2048x1024, C= A x B) using:
- Naive CUDA kernel
- GMEM Coalesced kernel
- Tiled shared memory CUDA kernel
- cuBLAS (via PyTorch)

Output is written to:
results.txt

## 📈 Visualize in Jupyter Notebook

pip install notebook matplotlib seaborn pandas
jupyter notebook benchmarks.ipynb

> Use benchmarks.ipynb to generate bar plots from results.txt.

## 📊 Results
![Kernel Comparision_2](https://github.com/user-attachments/assets/787a8b40-eb67-4bfd-84cd-7a0ef217652e)


## 🔁 Ongoing
Kernel Development:
- Warp Level Parallelism using fast register memory

Hyper Parameters:
- Tile size
- Grid launch

## 🙌 References
- NVIDIA CUDA Programming Guide (https://docs.nvidia.com/cuda/)
- Efficient GEMM Kernel Design (https://siboehm.com/articles/22/CUDA-MMM)


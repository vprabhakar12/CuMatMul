#include <cuda_runtime.h>
#include <iostream>
// using namespace std;

#define BLOCK_SIZE 32

__global__ void matmul_naive(const float* A, const float* B, float* C,
                             int M, int K, int N) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;  // x = row index (first dimension of output C)
    const int y = blockIdx.y * blockDim.y + threadIdx.y;  // y = col index (second dimension of output C)

    if (x < M && y < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = sum;
    }
    // C[0] = 5;
}

extern "C" void launch_naive(float* A, float* B, float* C, int M, int K, int N) {
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((M + BLOCK_SIZE-1) / BLOCK_SIZE, (N + BLOCK_SIZE-1) / BLOCK_SIZE);

    matmul_naive<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N);

    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    // std::cerr <<" value " << *C << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

__global__ void matmul_gmemcoal(const float* A, const float* B, float* C,
                             int M, int K, int N) {
    const int x = blockIdx.x * BLOCK_SIZE + threadIdx.x/BLOCK_SIZE;  // x = col index (2nd dimension of output C)
    const int y = blockIdx.y * BLOCK_SIZE + threadIdx.x%BLOCK_SIZE;  // y = row index (1st dimension of output C)

    if (x < M && y < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = sum;
    }
    // C[0] = 5;
}

extern "C" void launch_gmemcoal(float* A, float* B, float* C, int M, int K, int N) {
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE * BLOCK_SIZE);
    dim3 gridSize((M + BLOCK_SIZE-1) / BLOCK_SIZE, (N + BLOCK_SIZE-1) / BLOCK_SIZE);

    matmul_gmemcoal<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N);

    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

#define TILE_SIZE 16

__global__ void matmul_tiled(const float* A, const float* B, float* C, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        if (row < N && tile * TILE_SIZE + threadIdx.x < N)
            tileA[threadIdx.y][threadIdx.x] = A[row * N + tile * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && tile * TILE_SIZE + threadIdx.y < N)
            tileB[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

extern "C" void launch_tiled(float* A, float* B, float* C, int N) {
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    matmul_tiled<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
#include <iostream>
#include <chrono>
#include <fstream>
#include <cuda_runtime.h>

// CUDA 核心函數
__global__ void matrixMulKernel(float* A, float* B, float* C, int N) {
    // 計算當前線程的行和列索引
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 確保線程在矩陣範圍內
    if (row < N && col < N) {
        float value = 0;
        for (int k = 0; k < N; ++k) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

// 主機端函數
void matrixMultiplyCUDA(float* A, float* B, float* C, int N) {
    // 設置 CUDA 執行配置
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 計算內存大小
    size_t size = N * N * sizeof(float);

    // 在設備上分配內存
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // 將數據從主機複製到設備
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // 啟動 CUDA 核心函數
    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // 確保 CUDA 核心函數完成執行
    cudaDeviceSynchronize();

    // 將結果從設備複製回主機
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // 釋放設備內存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int N = 2000; // Example size of the matrix
    size_t size = N * N * sizeof(float);

    // Allocate memory for matrices
    float* A = (float*)malloc(size);
    float* B = (float*)malloc(size);
    float* C = (float*)malloc(size);

    // Initialize matrices with some values
    for (int i = 0; i < N * N; ++i) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(i);
    }

    // Measure the time taken for the CPU matrix multiplication
    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiplyCPU(A, B, C, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> durationCPU = end - start;

    // Write the last row of the result matrix to a file
    std::ofstream outputFile("cpu_output.txt");
    if (outputFile.is_open()) {
        for (int i = 0; i < N; ++i) {
            outputFile << C[(N-1) * N + i] << " "; // Write the last row
        }
        outputFile << "\n";
        outputFile.close();
    } else {
        std::cerr << "Unable to open file for writing\n";
    }

    std::cout << "CPU Matrix multiplication took " << durationCPU.count() << " ms\n";

    // Measure the time taken for the CUDA matrix multiplication
    start = std::chrono::high_resolution_clock::now();
    matrixMultiplyCUDA(A, B, C, N);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> durationCUDA = end - start;

    // Write the last row of the result matrix to a file
    outputFile.open("cuda_output.txt");
    if (outputFile.is_open()) {
        for (int i = 0; i < N; ++i) {
            outputFile << C[(N-1) * N + i] << " "; // Write the last row
        }
        outputFile << "\n";
        outputFile.close();
    } else {
        std::cerr << "Unable to open file for writing\n";
    }

    std::cout << "CUDA Matrix multiplication took " << durationCUDA.count() << " ms\n";

    // Free memory
    free(A);
    free(B);
    free(C);

    return 0;
}

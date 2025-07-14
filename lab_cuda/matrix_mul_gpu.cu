#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>

//TODO: CUDA kernel for matrix multiplication
__global__ void matrixMulKernel(float* A, float* B, float* C, int N) {
    //hint: implement the matrix multiplication
}

void matrixMultiplyGPU(float* h_A, float* h_B, float* h_C, int N) {
    // The matrix size
    size_t size = N * N * sizeof(float);

    //TODO: Allocate device(GPU) memory
    //hint: cudaMalloc(??, ??)
    float *d_A, *d_B, *d_C;

    //TODO: Copy matrices from host(CPU) to device(GPU)
    //hint: cudaMemcpy(??, ??, ??, ??)
    

    //TODO: Define block and grid dimensions
    //hint: dim3 threadsPerBlock(??, ??)
    //hint: dim3 numBlocks(??, ??)

    //TODO: Launch the matrix multiplication kernel
    //hint: matrixMulKernel<<<??, ??>>>(?, ?, ?, ?)

    //TODO: Copy result from device to host
    //hint: cudaMemcpy(??, ??, ??, ??);
    

    //TODO: Free device memory
    //hint: cudaFree()
}

int main() {
    int N = 2000; // Example size of the matrix
    size_t size = N * N * sizeof(float);

    // Allocate host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Initialize matrices with some values
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i);
    }

    // Perform matrix multiplication
    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiplyGPU(h_A, h_B, h_C, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;

    // Write the last row of the result matrix to a file
    std::ofstream outputFile("gpu_output.txt");
    if (outputFile.is_open()) {
        outputFile << h_C[0] << " ";
        outputFile << "\n";
        outputFile.close();
    } else {
        std::cerr << "Unable to open file for writing\n";
    }

    std::cout << "finish" << std::endl;
    std::cout << "GPU Matrix multiplication took " << duration.count() << " ms\n";

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}


// nvc++ matrix_mul_gpu.cu -o matrix_mul_gpu
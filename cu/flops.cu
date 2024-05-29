#include <iostream>
#include <cuda_runtime.h>
#include "cuda_helper.h"

#define BENCHMARK_GPU_ITERS 10000000


__global__ void flops_kernel(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = 0.0f;
        float mul = 1.0f;
        for (int i = 0; i < BENCHMARK_GPU_ITERS; ++i) {
            mul *= 1.00001f;
            sum += mul;
        }
        a[idx] = sum + b[idx] * c[idx];
    }
}


void calculateTheoreticalFLOPS(int deviceId) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);

    int coresPerSM = ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);

    int numSMs = deviceProp.multiProcessorCount;
    int totalCores = numSMs * coresPerSM;

    float clockRateGHz = deviceProp.clockRate * 1e-6f;
    float theoreticalFLOPS = totalCores * clockRateGHz * 2.0f;

    std::cout << "Device " << deviceId << ": " << deviceProp.name << "\n";
    std::cout << "  CUDA Cores: " << totalCores << "\n";
    std::cout << "  GPU Clock Rate: " << clockRateGHz << " GHz\n";
    std::cout << "  Theoretical Single-Precision FLOPS: " << theoreticalFLOPS << " GFLOPS\n";
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found.\n";
        return 1;
    }

    for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
        calculateTheoreticalFLOPS(deviceId);
    }


    int n = 1 << 20; // 1 million elements
    size_t size = n * sizeof(float);

    // Allocate memory on the host
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < n; ++i) {
        h_a[i] = 0.0f;
        h_b[i] = 2.0f;
        h_c[i] = 3.0f;
    }

    // Allocate memory on the device
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, size, cudaMemcpyHostToDevice);

    // Set up execution configuration
    int threadsPerBlock = 512;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Start timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Launch the kernel
    flops_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // Stop timing
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Calculate FLOPS
    // Each iteration in the loop does 2 operations (one multiply and one add)
    int operationsPerElement = BENCHMARK_GPU_ITERS * 2; // 100000 iterations, 2 operations each
    long long totalOperations = (long long)n * operationsPerElement;
    float flops = (totalOperations / (elapsedTime / 1000.0f)) / 1e9; // GFLOPS

    std::cout << "Elapsed time: " << elapsedTime << " ms\n";
    std::cout << "GFLOPS: " << flops << "\n";

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}



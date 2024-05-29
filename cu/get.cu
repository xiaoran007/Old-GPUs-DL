#include <iostream>
#include <cuda_runtime.h>
#include "cuda_helper.h"

void printDeviceProperties(int device) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    std::cout << "Device Number: " << device << std::endl;
    std::cout << "  Device name: " << prop.name << std::endl;
    std::cout << "  Memory Clock Rate (MHz): " << prop.memoryClockRate / 1000 << std::endl;
    std::cout << "  Memory Bus Width (bits): " << prop.memoryBusWidth << std::endl;
    std::cout << "  Peak Memory Bandwidth (GB/s): " << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 << std::endl;
    std::cout << "  Total Global Memory (MB): " << prop.totalGlobalMem / 1024 / 1024 << std::endl;
    std::cout << "  Shared Memory per Block (bytes): " << prop.sharedMemPerBlock << std::endl;
    std::cout << "  Number of SMs: " << prop.multiProcessorCount << std::endl;
    std::cout << "  Number of CUDA cores: " << ConvertSMVer2Cores(prop.major, prop.minor) * prop.multiProcessorCount << std::endl;
    std::cout << "  Compute Mode: " << prop.computeMode << std::endl;
    std::cout << "  Warp Size: " << prop.warpSize << std::endl;
    std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "  Max Threads Dimension: (" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << std::endl;
    std::cout << "  Max Grid Size: (" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << std::endl;
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "-------------------------------------------" << std::endl;
}

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cout << "No CUDA devices found." << std::endl;
    } else {
        std::cout << "Found " << deviceCount << " CUDA device(s)." << std::endl;
        std::cout << "-------------------------------------------" << std::endl;
        for (int device = 0; device < deviceCount; ++device) {
            printDeviceProperties(device);
        }
    }

    return 0;
}

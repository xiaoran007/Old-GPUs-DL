#include <iostream>
#include <cuda_runtime.h>

void getPcieInfo(int deviceId) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);

    int pciBusId, pciDeviceId, pciDomainId;
    cudaDeviceGetAttribute(&pciBusId, cudaDevAttrPciBusId, deviceId);
    cudaDeviceGetAttribute(&pciDeviceId, cudaDevAttrPciDeviceId, deviceId);
    cudaDeviceGetAttribute(&pciDomainId, cudaDevAttrPciDomainId, deviceId);

    int pciLinkGeneration;
    cudaDeviceGetAttribute(&pciLinkGeneration, cudaDevAttrPciBusId, deviceId);

    int pciLinkWidth;
    cudaDeviceGetAttribute(&pciLinkWidth, cudaDevAttrPciBusId, deviceId);

    std::cout << "Device " << deviceId << ": " << deviceProp.name << "\n";
    std::cout << "  PCI Bus ID: " << pciBusId << "\n";
    std::cout << "  PCI Device ID: " << pciDeviceId << "\n";
    std::cout << "  PCI Domain ID: " << pciDomainId << "\n";
    std::cout << "  PCIe Link Generation: " << pciLinkGeneration << "\n";
    std::cout << "  PCIe Link Width: " << pciLinkWidth << "\n";
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found.\n";
        return 1;
    }

    for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
        getPcieInfo(deviceId);
    }

    return 0;
}

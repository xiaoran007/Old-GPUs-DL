#include "nvml_helper.h"

void checkNvmlError(nvmlReturn_t result) {
    if (result != NVML_SUCCESS) {
        std::cerr << "NVML error: " << nvmlErrorString(result) << std::endl;
        exit(1);
    }
}

int getPCIEMaxLinkWidth(int deviceIndex) {
    nvmlReturn_t result;

    result = nvmlInit();
    checkNvmlError(result);


    nvmlDevice_t device;
    result = nvmlDeviceGetHandleByIndex(deviceIndex, &device);
    checkNvmlError(result);

    unsigned int pcieLinkWidth;
    result = nvmlDeviceGetMaxPcieLinkWidth(device, &pcieLinkWidth);
    checkNvmlError(result);

    result = nvmlShutdown();
    checkNvmlError(result);

    return pcieLinkWidth;
}

int getPCIEMaxLinkGeneration(int deviceIndex) {
    nvmlReturn_t result;

    result = nvmlInit();
    checkNvmlError(result);


    nvmlDevice_t device;
    result = nvmlDeviceGetHandleByIndex(deviceIndex, &device);
    checkNvmlError(result);

    unsigned int pcieLinkGeneration;
    result = nvmlDeviceGetMaxPcieLinkGeneration(device, &pcieLinkGeneration);
    checkNvmlError(result);

    result = nvmlShutdown();
    checkNvmlError(result);

    return pcieLinkGeneration;
}

int getPCIECurrentLinkWidth(int deviceIndex) {
    nvmlReturn_t result;

    result = nvmlInit();
    checkNvmlError(result);


    nvmlDevice_t device;
    result = nvmlDeviceGetHandleByIndex(deviceIndex, &device);
    checkNvmlError(result);

    unsigned int pcieLinkWidth;
    result = nvmlDeviceGetCurrPcieLinkWidth(device, &pcieLinkWidth);
    checkNvmlError(result);

    result = nvmlShutdown();
    checkNvmlError(result);

    return pcieLinkWidth;
}

int getPCIECurrentLinkGeneration(int deviceIndex) {
    nvmlReturn_t result;

    result = nvmlInit();
    checkNvmlError(result);


    nvmlDevice_t device;
    result = nvmlDeviceGetHandleByIndex(deviceIndex, &device);
    checkNvmlError(result);

    unsigned int pcieLinkGeneration;
    result = nvmlDeviceGetCurrPcieLinkGeneration(device, &pcieLinkGeneration);
    checkNvmlError(result);

    result = nvmlShutdown();
    checkNvmlError(result);

    return pcieLinkGeneration;
}

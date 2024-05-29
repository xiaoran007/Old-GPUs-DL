#include "cuda_helper.h"

// copy and modified from https://github.com/NVIDIA/cuda-samples

int ConvertSMVer2Cores(int major, int minor) {
    std::unordered_map<int, int> nGpuArchCoresPerSM = {
        {30, 192},
        {32, 192},
        {35, 192},
        {37, 192},
        {50, 128},
        {52, 128},
        {53, 128},
        {60,  64},
        {61, 128},
        {62, 128},
        {70,  64},
        {72,  64},
        {75,  64},
        {80,  64},
        {86, 128},
        {87, 128},
        {89, 128},
        {90, 128},
        {-1, -1}
    };
    int SMVer = major * 10 + minor;
    if (nGpuArchCoresPerSM.find(SMVer) != nGpuArchCoresPerSM.end()) {
        return nGpuArchCoresPerSM[SMVer];
    }
    else {
        return nGpuArchCoresPerSM[-1];
    }
}

std::string ConvertSMVer2ArchName(int major, int minor) {
    std::unordered_map<int, std::string> nGpuArchNameSM = {
        {0x30, "Kepler"},
        {0x32, "Kepler"},
        {0x35, "Kepler"},
        {0x37, "Kepler"},
        {0x50, "Maxwell"},
        {0x52, "Maxwell"},
        {0x53, "Maxwell"},
        {0x60, "Pascal"},
        {0x61, "Pascal"},
        {0x62, "Pascal"},
        {0x70, "Volta"},
        {0x72, "Xavier"},
        {0x75, "Turing"},
        {0x80, "Ampere"},
        {0x86, "Ampere"},
        {0x87, "Ampere"},
        {0x89, "Ada"},
        {0x90, "Hopper"},
        {-1, "Graphics Device"}};
        int SMVer = major * 10 + minor;
        if (nGpuArchNameSM.find(SMVer) != nGpuArchNameSM.end()) {
            return nGpuArchNameSM[SMVer];
        }
        else {
            return nGpuArchNameSM[-1];
        }
}


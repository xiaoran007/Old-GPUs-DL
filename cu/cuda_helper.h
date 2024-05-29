#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

// copy from https://github.com/NVIDIA/cuda-samples

#include <map>
#include <unordered_map>
#include <string>
#include <iostream>

int ConvertSMVer2Cores(int, int);
std::string ConvertSMVer2ArchName(int, int);


#endif
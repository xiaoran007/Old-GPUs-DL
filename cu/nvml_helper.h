#ifndef NVML_HELPER_H
#define NVML_HELPER_H

#include <iostream>
#include <nvml.h>
#include <unordered_map>
#include <string>

void checkNvmlError(nvmlReturn_t result);
int getPCIEMaxLinkWidth(int deviceIndex);
int getPCIEMaxLinkGeneration(int deviceIndex);
int getPCIECurrentLinkWidth(int deviceIndex);
int getPCIECurrentLinkGeneration(int deviceIndex);



#endif
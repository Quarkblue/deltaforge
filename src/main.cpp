#include <cuda_runtime.h>
#include <iostream>

#include "model.hpp"


int checkGPUStatus() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr <<  "No CUDA devices found" << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Global memory: " << prop.totalGlobalMem / B_TO_MB << " MB\n";
    std::cout << "SM count: " << prop.multiProcessorCount << "\n";
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << "\n";
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "Free memory: " << free_mem / (double)B_TO_GB << " GB, Total memory. " << total_mem / (double)B_TO_GB << " GB\n";
    return 0;
}

int main() {
    checkGPUStatus();
    Model model(L"D:\\AI\\Inferencing\\deltaforge\\model\\Qwen3.5-2B\\",
                L"model.safetensors-00001-of-00001.safetensors", L"config.json");
    model.selfCheck();
    model.dump();
    return 0;
}
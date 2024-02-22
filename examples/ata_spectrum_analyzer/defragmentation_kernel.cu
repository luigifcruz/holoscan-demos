#include "defragmentation_kernel.hh"

#include <cuda_runtime.h>
#include <tuple>
#include <stdio.h>

// Partial Packet [A=1,  F=96,   T=32, P=2]
// Total Packet   [A=5, F=192, T=8192, P=2]

__global__ void DefragmentationKernel(void* defragmentedData, 
                                      void** fragmentedData, 
                                      int totalFragments,
                                      uint64_t fragmentSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int transferIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= totalFragments) {
        return;
    }

    uint64_t fragmentStartIndex = idx * fragmentSize;
    uint16_t* src = ((uint16_t**)fragmentedData)[idx];
    uint16_t* dst = &((uint16_t*)defragmentedData)[fragmentStartIndex];

    for (int i = 0; i < fragmentSize; i += blockDim.y * gridDim.y) {
        if (transferIndex + i < fragmentSize) {
            dst[transferIndex + i] = src[transferIndex + i];
        }
    }
}

cudaError_t LaunchDefragmentationKernel(void* defragmentedData, 
                                        void** fragmentedData, 
                                        int totalFragments,
                                        int fragmentSize,
                                        cudaStream_t stream) {
    dim3 threadsPerBlock = {8, 64, 1};
    dim3 blocksPerGrid = {
        (totalFragments + threadsPerBlock.x - 1) / threadsPerBlock.x, 
        (fragmentSize + threadsPerBlock.y - 1) / threadsPerBlock.y, 
        1
    };

    DefragmentationKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(defragmentedData, 
                                                                         fragmentedData, 
                                                                         totalFragments,
                                                                         fragmentSize);

    return cudaGetLastError();
}

#include "defragmentation_kernel.hh"

#include <cuda_runtime.h>
#include <tuple>
#include <stdio.h>

// Partial Packet [A=1,  F=96,   T=32, P=2]
// Total Packet   [A=5, F=192, T=8192, P=2]

__global__ void DefragmentationKernel(void* defragmented_gpu_data, 
                                      void** fragmented_gpu_data, 
                                      int total_fragments,
                                      uint64_t fragmentSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int transferIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= total_fragments) {
        return;
    }

    uint64_t fragmentStartIndex = idx * fragmentSize;
    uint16_t* src = ((uint16_t**)fragmented_gpu_data)[idx];
    uint16_t* dst = &((uint16_t*)defragmented_gpu_data)[fragmentStartIndex];

    for (int i = 0; i < fragmentSize; i += blockDim.y * gridDim.y) {
        if (transferIndex + i < fragmentSize) {
            dst[transferIndex + i] = src[transferIndex + i];
        }
    }
}

cudaError_t LaunchDefragmentationKernel(void* defragmented_gpu_data, 
                                        void** fragmented_gpu_data, 
                                        int total_fragments,
                                        cudaStream_t stream, 
                                        std::tuple<uint64_t, uint64_t, uint64_t, uint64_t> partialShape,
                                        std::tuple<uint64_t, uint64_t, uint64_t, uint64_t> totalShape) {
    const auto& [pA, pF, pT, pP] = partialShape;
    const auto& [tA, tF, tT, tP] = totalShape;

    size_t totalFragments = total_fragments;
    size_t fragmentSize = pA * pF * pT * pP;

    dim3 threadsPerBlock = {8, 64, 1};
    dim3 blocksPerGrid = {(totalFragments + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                          (fragmentSize + threadsPerBlock.y - 1) / threadsPerBlock.y, 
                          1};

    DefragmentationKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(defragmented_gpu_data, 
                                                                         fragmented_gpu_data, 
                                                                         totalFragments,
                                                                         fragmentSize);

    return cudaGetLastError();
}

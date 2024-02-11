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

    if (idx >= total_fragments) {
        return;
    }

    uint64_t fragmentStartIndex = idx * fragmentSize;
    uint16_t* src = ((uint16_t**)fragmented_gpu_data)[idx];
    uint16_t* dst = &((uint16_t*)defragmented_gpu_data)[fragmentStartIndex];

    for (uint64_t i = 0; i < fragmentSize; i += 4) {
        dst[i + 0] = src[i + 0];
        dst[i + 1] = src[i + 1];
        dst[i + 2] = src[i + 2];
        dst[i + 3] = src[i + 3];
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

    int threadsPerBlock = 512;
    int blocksPerGrid = (total_fragments + threadsPerBlock - 1) / threadsPerBlock;

    DefragmentationKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(defragmented_gpu_data, 
                                                                         fragmented_gpu_data, 
                                                                         total_fragments,
                                                                         pA * pF * pT * pP);

    return cudaGetLastError();
}

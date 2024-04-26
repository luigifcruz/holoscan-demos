#include "defragmentation_kernel.hh"

#include <cuda_runtime.h>
#include <tuple>
#include <stdio.h>

// Partial Packet [A=1,  F=96,   T=16, P=2]
// Total Packet   [A=5, F=192, T=8192, P=2]
// Fragment       [A=5, F=  2, T= 512, P=1]

__global__ void DefragmentationKernel(void* defragmentedData, 
                                      void** fragmentedData, 
                                      int totalFragments,
                                      uint64_t fragmentSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= totalFragments) {
        return;
    }

    // Transform the fragment index into shape.

    int fragmentPolarizationIndex = idx % 1;
    int fragmentTimeIndex = (idx / 1) % 512;
    int fragmentFrequencyIndex = (idx / (1 * 512)) % 2;
    int fragmentAntennaIndex = (idx / (1 * 512 * 2)) % 5;

    // Transform the shape into the defragmented data offset.

    int defragmentationOffset = 0;
    defragmentationOffset += (fragmentAntennaIndex * 1) * 192 * 8192 * 2;
    defragmentationOffset += (fragmentFrequencyIndex * 96) * 8192 * 2;
    defragmentationOffset += (fragmentTimeIndex * 16) * 2;
    defragmentationOffset += (fragmentPolarizationIndex * 2);

    // Get the source and destination pointers.

    uint16_t* src = ((uint16_t**)fragmentedData)[idx];
    uint16_t* dst = &((uint16_t*)defragmentedData)[defragmentationOffset];

    // Copy the fragment into the defragmented data.

    for (int A = 0; A < 1; A++) {
        for (int F = 0; F < 96; F++) {
            for (int T = 0; T < 16; T++) {
                for (int P = 0; P < 2; P++) {
                    int srcOffset = A * 96 * 16 * 2 + F * 16 * 2 + T * 2 + P;
                    int dstOffset = A * 192 * 8192 * 2 + F * 8192 * 2 + T * 2 + P;

                    dst[dstOffset] = src[srcOffset];
                }
            }
        }
    }
}

cudaError_t LaunchDefragmentationKernel(void* defragmentedData, 
                                        void** fragmentedData, 
                                        int totalFragments,
                                        int fragmentSize,
                                        cudaStream_t stream) {
    dim3 threadsPerBlock = {512, 1, 1};
    dim3 blocksPerGrid = {
        (totalFragments + threadsPerBlock.x - 1) / threadsPerBlock.x, 
        1, 
        1
    };

    DefragmentationKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(defragmentedData, 
                                                                         fragmentedData, 
                                                                         totalFragments,
                                                                         fragmentSize);

    return cudaGetLastError();
}

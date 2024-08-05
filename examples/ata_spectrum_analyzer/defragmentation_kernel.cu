#include "defragmentation_kernel.hh"

#include <cuda_runtime.h>
#include <tuple>

// TODO: Improve performance.

__global__ void DefragmentationKernel(void* defragmentedData, 
                                      void** fragmentedData, 
                                      int totalFragments,
                                      BlockShape totalShape,
                                      BlockShape partialShape,
                                      BlockShape fragmentShape) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= totalFragments) {
        return;
    }

    // Breakout the shapes.

    int tF = totalShape.number_of_channels;
    int tT = totalShape.number_of_samples;
    int tP = totalShape.number_of_polarizations;

    int pA = partialShape.number_of_antennas;
    int pF = partialShape.number_of_channels;
    int pT = partialShape.number_of_samples;
    int pP = partialShape.number_of_polarizations;

    int fA = fragmentShape.number_of_antennas;
    int fF = fragmentShape.number_of_channels;
    int fT = fragmentShape.number_of_samples;
    int fP = fragmentShape.number_of_polarizations;

    // Transform the fragment index into shape.

    int fragmentPolarizationIndex = idx % fP;
    int fragmentTimeIndex = (idx / fP) % fT;
    int fragmentFrequencyIndex = (idx / (fP * fT)) % fF;
    int fragmentAntennaIndex = (idx / (fP * fT * fF)) % fA;

    // Transform the shape into the defragmented data offset.

    int defragmentationOffset = 0;
    defragmentationOffset += (fragmentAntennaIndex * pA) * tF * tT * tP;
    defragmentationOffset += (fragmentFrequencyIndex * pF) * tT * tP;
    defragmentationOffset += (fragmentTimeIndex * pT) * tP;
    defragmentationOffset += (fragmentPolarizationIndex * pP);

    // Get the source and destination pointers.

    uint16_t* src = ((uint16_t**)fragmentedData)[idx];
    uint16_t* dst = &((uint16_t*)defragmentedData)[defragmentationOffset];

    // Copy the fragment into the defragmented data.

    for (int A = 0; A < pA; A++) {
        for (int F = 0; F < pF; F++) {
            for (int T = 0; T < pT; T++) {
                for (int P = 0; P < pP; P++) {
                    int srcOffset = A * pF * pT * pP + F * pT * pP + T * pP + P;
                    int dstOffset = A * tF * tT * tP + F * tT * tP + T * tP + P;

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
                                        BlockShape totalShape,
                                        BlockShape partialShape,
                                        BlockShape fragmentShape,
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
                                                                         totalShape,
                                                                         partialShape,
                                                                         fragmentShape);

    return cudaGetLastError();
}

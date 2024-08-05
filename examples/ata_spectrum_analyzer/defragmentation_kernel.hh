#ifndef DEFRAGMENTATION_KERNEL_HH
#define DEFRAGMENTATION_KERNEL_HH

#include <cuda_runtime.h>
#include <cstdint>

#include "types.hh"

cudaError_t LaunchDefragmentationKernel(void* defragmentedData, 
                                        void** fragmentedData, 
                                        int totalFragments,
                                        int fragmentSize,
                                        BlockShape totalShape,
                                        BlockShape partialShape,
                                        BlockShape fragmentShape,
                                        cudaStream_t stream);

#endif  // DEFRAGMENTATION_KERNEL_HH
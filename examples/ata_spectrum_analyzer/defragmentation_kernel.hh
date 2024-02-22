#ifndef DEFRAGMENTATION_KERNEL_HH
#define DEFRAGMENTATION_KERNEL_HH

#include <cuda_runtime.h>
#include <tuple>
#include <cstdint>

cudaError_t LaunchDefragmentationKernel(void* defragmentedData, 
                                        void** fragmentedData, 
                                        int totalFragments,
                                        int fragmentSize,
                                        cudaStream_t stream);

#endif  // DEFRAGMENTATION_KERNEL_HH
#ifndef DEFRAGMENTATION_KERNEL_HH
#define DEFRAGMENTATION_KERNEL_HH

#include <cuda_runtime.h>
#include <tuple>
#include <cstdint>

cudaError_t LaunchDefragmentationKernel(void* defragmented_gpu_data, 
                                        void** fragmented_gpu_data, 
                                        int _total_fragments, 
                                        cudaStream_t stream, 
                                        std::tuple<uint64_t, uint64_t, uint64_t, uint64_t> p,
                                        std::tuple<uint64_t, uint64_t, uint64_t, uint64_t> t);

#endif  // DEFRAGMENTATION_KERNEL_HH
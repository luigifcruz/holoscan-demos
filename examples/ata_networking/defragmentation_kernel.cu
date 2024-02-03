#include "defragmentation_kernel.hh"

// Partial Packet [A=1,  F=96,   T=32, P=2]
// Total Packet   [A=5, F=192, T=8192, P=2]

__global__ void DefragmentationKernel(void* defragmented_gpu_data, 
                                      void** fragmented_gpu_data, 
                                      int total_fragments, 
                                      std::tuple<uint64_t, uint64_t, uint64_t, uint64_t> partialShape,
                                      std::tuple<uint64_t, uint64_t, uint64_t, uint64_t> totalShape) {
    const auto& [pA, pF, pT, pP] = partialShape;
    const auto& [tA, tF, tT, tP] = totalShape;

    int fragment_id = blockIdx.x;
    int thread_id = threadIdx.x;

    int total_elements = tA * tF * tT * tP;
    int elements_per_fragment = total_elements / total_fragments;

    int start = fragment_id * elements_per_fragment;
    int end = (fragment_id + 1) * elements_per_fragment;

    if (fragment_id == total_fragments - 1) {
        end = total_elements;
    }

    int total_packets = tA * tF * tT;
    int packets_per_fragment = total_packets / total_fragments;

    int start_packet = fragment_id * packets_per_fragment;
    int end_packet = (fragment_id + 1) * packets_per_fragment;

    if (fragment_id == total_fragments - 1) {
        end_packet = total_packets;
    }

    for (int i = start; i < end; i++) {
        int packet_id = i / (pF * pT * pP);
        int packet_offset = i % (pF * pT * pP);
        int fragment_offset = i % elements_per_fragment;

        int packet_start = packet_id * pF * pT * pP;
        int packet_end = (packet_id + 1) * pF * pT * pP;

        if (packet_id >= start_packet && packet_id < end_packet) {
            int fragment_id = packet_id - start_packet;
            int fragment_start = fragment_id * pF * pT * pP;
            int fragment_end = (fragment_id + 1) * pF * pT * pP;

            int fragment_offset = packet_offset + fragment_start;
            int defragmented_offset = fragment_offset + fragment_id * elements_per_fragment;

            ((uint16_t*)(defragmented_gpu_data))[defragmented_offset] = ((uint16_t**)(fragmented_gpu_data))[fragment_id][fragment_offset];
        }
    }
}

cudaError_t DefragmentationKernel(void* defragmented_gpu_data, 
                                  void** fragmented_gpu_data, 
                                  int total_fragments, 
                                  cudaStream_t stream, 
                                  std::tuple<uint64_t, uint64_t, uint64_t, uint64_t> partialShape,
                                  std::tuple<uint64_t, uint64_t, uint64_t, uint64_t> totalShape) {
    DefragmentationKernel<<<total_fragments, 32, 0, stream>>>(defragmented_gpu_data, 
                                                              fragmented_gpu_data, 
                                                              total_fragments, 
                                                              partialShape, 
                                                              totalShape);
    return cudaGetLastError();
}
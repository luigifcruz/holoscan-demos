#ifndef TYPES_HH
#define TYPES_HH

#include <cstdint>

struct BlockShape {
    uint64_t number_of_antennas;
    uint64_t number_of_channels;
    uint64_t number_of_samples;
    uint64_t number_of_polarizations;
};

#endif  // TYPES_HH
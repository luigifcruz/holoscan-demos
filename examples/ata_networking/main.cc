#include "adv_network_rx.h"
#include "adv_network_tx.h"
#include "adv_network_kernels.h"

#include "holoscan/holoscan.hpp"
#include "holoscan/core/operator.hpp"

#include <netinet/in.h>

#include <ostream>
#include <cstdint>
#include <cassert>
#include <queue>
#include <chrono>
#include <tuple>
#include <vector>

#include "defragmentation_kernel.hh"

#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        HOLOSCAN_LOG_ERROR("CUDA error: {} ({})", cudaGetErrorString(error), error); \
        std::abort(); \
    } \
}

#define ASSERT(condition) { \
    if (!(condition)) { \
        HOLOSCAN_LOG_ERROR("Assertion failed."); \
        std::abort(); \
    } \
}



struct VoltagePacket {
  uint8_t  version;
  uint8_t  type;
  uint16_t number_of_channels;
  uint16_t channel_number;
  uint16_t antenna_id;
  uint64_t timestamp;
  uint8_t  data[];

  VoltagePacket(const uint8_t* ptr) {
    const auto* p = reinterpret_cast<const VoltagePacket*>(ptr);
    version = p->version;
    type = p->type;
    number_of_channels = ntohs(p->number_of_channels);
    channel_number = ntohs(p->channel_number);
    antenna_id = ntohs(p->antenna_id);
    timestamp = be64toh(p->timestamp);
  }
} __attribute__((packed));



struct BlockShape {
    uint64_t number_of_antennas;
    uint64_t number_of_channels;
    uint64_t number_of_samples;
    uint64_t number_of_polarizations;
};

template<>
struct fmt::formatter<BlockShape> {
    constexpr auto parse(format_parse_context& ctx) {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(const BlockShape& b, FormatContext& ctx) {
        return format_to(ctx.out(), "[A: {}, C: {}, S: {}, P: {}]", b.number_of_antennas, 
                                                                    b.number_of_channels, 
                                                                    b.number_of_samples, 
                                                                    b.number_of_polarizations);
    }
};

template <>
struct YAML::convert<BlockShape> {
    static Node encode(const BlockShape& input_spec) {
        Node node;

        node["number_of_antennas"] = std::to_string(input_spec.number_of_antennas);
        node["number_of_channels"] = std::to_string(input_spec.number_of_channels);
        node["number_of_samples"] = std::to_string(input_spec.number_of_samples);
        node["number_of_polarizations"] = std::to_string(input_spec.number_of_polarizations);

        return node;
    }

    static bool decode(const Node& node, BlockShape& input_spec) {
        if (!node.IsMap()) {
            GXF_LOG_ERROR("InputSpec: expected a map");
            return false;
        }

        if (!node["number_of_antennas"] || 
            !node["number_of_channels"] || 
            !node["number_of_samples"]  || 
            !node["number_of_polarizations"]) {
            GXF_LOG_ERROR("InputSpec: missing required fields");
            return false;
        }

        input_spec.number_of_antennas = node["number_of_antennas"].as<uint64_t>();
        input_spec.number_of_channels = node["number_of_channels"].as<uint64_t>();
        input_spec.number_of_samples = node["number_of_samples"].as<uint64_t>();
        input_spec.number_of_polarizations = node["number_of_polarizations"].as<uint64_t>();

        return true;
    }
};



namespace holoscan::ops {

class AtaTransportOpRx : public Operator {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(AtaTransportOpRx)

    AtaTransportOpRx() = default;

    void initialize() {
        register_converter<BlockShape>();

        Operator::initialize();
    }

    void setup(OperatorSpec& spec) override {
        spec.input<AdvNetBurstParams>("burst_in");

        spec.param(concurrent_blocks_, "concurrent_blocks");
        spec.param(time_step_length_, "time_step_length");
        spec.param(transport_header_size_, "transport_header_size");
        spec.param(voltage_header_size_, "voltage_header_size");

        spec.param(total_block_, "total_block");
        spec.param(partial_block_, "partial_block");
        spec.param(offset_block_, "offset_block");
    }

    void start() {
        block_seed_time = 0;
        start_timestamp = 0;
        end_timestamp = 0;

        HOLOSCAN_LOG_INFO("Creating {} checkerboards with params:", concurrent_blocks_.get());
        HOLOSCAN_LOG_INFO("  - Total: {}", total_block_.get());
        HOLOSCAN_LOG_INFO("  - Partial: {}", partial_block_.get());
        HOLOSCAN_LOG_INFO("  - Offset: {}", offset_block_.get());

        if (offset_block_.get().number_of_polarizations != 0) {
            HOLOSCAN_LOG_WARN("Polarization block offset does nothing.");
        }

        for (uint64_t i = 0; i < concurrent_blocks_; i++) {
            checkerboard_pool.emplace_back(std::make_shared<Checkerboard>(total_block_, 
                                                                          partial_block_, 
                                                                          offset_block_,
                                                                          time_step_length_));
        }

        checkerboards.reserve(concurrent_blocks_);
        for (uint64_t i = 0; i < concurrent_blocks_; i++) {
            checkerboards.push_back(checkerboard_pool[i]);
        }

        block_time_length = checkerboards[0]->total_time_steps() * time_step_length_;

        // Setup report system.

        obsolete_rewinds_counter = 0;
        complete_rewinds_counter = 0;

        report_thread_running = true;
        report_thread = std::thread([&]{
            while (report_thread_running) {
                report_thread_callback();
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            }
        });
    }

    void stop() {
        report_thread_running = false;
        if (report_thread.joinable()) {
            report_thread.join();
        }
    }

    void compute(InputContext& op_input, OutputContext&, ExecutionContext&) override {
        auto packet_burst = op_input.receive<std::shared_ptr<AdvNetBurstParams>>("burst_in").value();

        // Check for defragmented checkerboards.

        if (!defragment_queue.empty()) {
            auto checkerboard = defragment_queue.front();
            if (!checkerboard->is_processing()) {
                complete_rewinds_counter++;
                
                checkerboard->rewind(end_timestamp);
                start_timestamp += block_time_length;
                end_timestamp += block_time_length;

                defragment_queue.pop();
                checkerboards.push_back(checkerboard);
            }
        }

        // Check for obsolete checkerboards.

        for (const auto& checkerboard : checkerboards) {
            if (checkerboard->time_range().end < start_timestamp) {
                obsolete_rewinds_counter++;
                
                checkerboard->rewind(end_timestamp);
                start_timestamp += block_time_length;
                end_timestamp += block_time_length;
            }
        }

        // TODO: This introduces a deadlock condition:
        //       When the checkerboard holds the maximum number of ANO packets (`num_concurrent_batches * batch_size`)
        //       the checkerboard will never be rewound causing the deadlock. This is because the logic relies on the
        //       `packet.timestamp` to drop obsolete packets. If there is no new packets, the logic will never be
        //       triggered. Possible solutions: 1) Use a timeout to rewind the checkerboard; 2) Monitor the number of
        //       packets in the checkerboard and rewind when the number of packets is greater than a threshold; For now,
        //       the number of checkerbords fragments shouldn't be greater than the maxium number of ANO packets.

        // Process the packet burst.

        if (!packet_burst) {
            return;
        }

        const auto& number_of_packets = adv_net_get_num_pkts(packet_burst);
        packet_counter += number_of_packets;

        // TODO: Cache the cache vectors.
        std::vector<void*> cpu_pkts;
        cpu_pkts.reserve(1000);
        std::vector<void*> gpu_pkts;
        gpu_pkts.reserve(1000);

        std::shared_ptr<Checkerboard> checkerboard;

        for (uint64_t i = 0; i < number_of_packets; i++) {
            const auto* p = reinterpret_cast<uint8_t*>(adv_net_get_cpu_pkt_ptr(packet_burst, i));
            const VoltagePacket packet(p + transport_header_size_);

#ifdef DEBUG
            HOLOSCAN_LOG_INFO("Received voltage packet with version: {}, type: {}, number_of_channels: {}, "
                              "channel_number: {}, antenna_id: {}, timestamp: {}", packet.version,
                                                                                   packet.type,
                                                                                   packet.number_of_channels,
                                                                                   packet.channel_number,
                                                                                   packet.antenna_id,
                                                                                   packet.timestamp);
#endif

            // Initialize packet based timer.
            // TODO: Add assertions for timer wrap.
        
            if (block_seed_time == 0) {
                HOLOSCAN_LOG_INFO("Initializing checkerboards with {} seed time.", packet.timestamp);

                block_seed_time = packet.timestamp;
                start_timestamp = block_seed_time;
                end_timestamp = block_seed_time;

                for (uint64_t i = 0; i < concurrent_blocks_; i++) {
                    checkerboards[i]->rewind(end_timestamp);
                    end_timestamp += block_time_length;
                }
            }

            // Rewind checkerboards if necessary. 
            // TODO: I don't like this `while` here. Remove it.

            while (packet.timestamp >= end_timestamp && checkerboards.size() > 0) {
                for (const auto& checkerboard : checkerboards) {
                    obsolete_rewinds_counter++;
                    
                    checkerboard->rewind(end_timestamp);
                    start_timestamp += block_time_length;
                    end_timestamp += block_time_length;
                }
            }
/*
            if (packet.antenna_id >= 5) {
                cpu_pkts.push_back(packet_burst->cpu_pkts[i]);
                gpu_pkts.push_back(packet_burst->gpu_pkts[i]);
                continue;
            }
*/
            // Check packet timestamp is within the range of the checkerboards.

            if (packet.timestamp < start_timestamp || packet.timestamp >= end_timestamp) {
                cpu_pkts.push_back(packet_burst->cpu_pkts[i]);
                gpu_pkts.push_back(packet_burst->gpu_pkts[i]);
                continue;
            }

            // Get range checkerboard.

            uint64_t id = 0;
            
            if (!checkerboard || !checkerboard->in_range(packet.timestamp)) {
                for (uint64_t x = 0; x < checkerboards.size(); x++) {
                    if (checkerboards[x]->in_range(packet.timestamp)) {
                        id = x;
                        checkerboard = checkerboards[x];
                        break;
                    }
                }
            }

            // Try to add fragment to checkerboard. Free packets if it fails.

            if (!checkerboard || !checkerboard->add_fragment(packet_burst, i, packet)) {
                cpu_pkts.push_back(packet_burst->cpu_pkts[i]);
                gpu_pkts.push_back(packet_burst->gpu_pkts[i]);
                continue;
            }

            // Check if the checkerboard is complete. Start processing if it is.

            if (checkerboard->is_complete()) {
                checkerboard->defragment();
                defragment_queue.push(checkerboard);
                checkerboards.erase(checkerboards.begin() + id);
            }
        }

        if (!cpu_pkts.empty()) {
            adv_net_free_pkts(cpu_pkts.data(), cpu_pkts.size());
            adv_net_free_pkts(gpu_pkts.data(), gpu_pkts.size());
        }

        if (packet_burst.unique()) {
            adv_net_free_rx_burst(packet_burst);
        }
    }

 private:
    struct Fragment {
        bool is_complete;
        std::shared_ptr<AdvNetBurstParams> burst;
        void* cpu_packet;
        void* gpu_packet;
    };

    struct Checkerboard {
     public:
        struct TimeRange {
            uint64_t start;
            uint64_t end;
        };

        constexpr const uint64_t& total_time_steps() const {
            return _total_time_steps;
        }

        constexpr const uint64_t& time_step_length() const {
            return _time_step_length;
        }

        constexpr const uint64_t& fragment_counter() const {
            return _fragment_counter;
        }

        constexpr const uint64_t& total_fragments() const {
            return _total_fragments;
        }

        constexpr const TimeRange& time_range() const {
            return _time_range;
        }

        explicit Checkerboard(const BlockShape& total, 
                              const BlockShape& partial,
                              const BlockShape& offset,
                              const uint64_t& time_step_length) : _total(total), 
                                                                  _partial(partial),
                                                                  _offset(offset),
                                                                  _fragment_counter(0),
                                                                  _time_step_length(time_step_length) {
            ASSERT(_total.number_of_antennas >= _partial.number_of_antennas);
            ASSERT(_total.number_of_channels >= _partial.number_of_channels);
            ASSERT(_total.number_of_samples >= _partial.number_of_samples);
            ASSERT(_total.number_of_polarizations >= _partial.number_of_polarizations);

            ASSERT((_total.number_of_antennas % _partial.number_of_antennas) == 0);
            ASSERT((_total.number_of_channels % _partial.number_of_channels) == 0);
            ASSERT((_total.number_of_samples % _partial.number_of_samples) == 0);
            ASSERT((_total.number_of_polarizations % _partial.number_of_polarizations) == 0);

            _total_fragments = 1;
            _total_fragments *= _antenna_slots = _total.number_of_antennas / _partial.number_of_antennas;
            _total_fragments *= _channel_slots = _total.number_of_channels / _partial.number_of_channels;
            _total_fragments *= _sample_slots = _total.number_of_samples / _partial.number_of_samples;
            _total_fragments *= _total.number_of_polarizations / _partial.number_of_polarizations;

            fragments.resize(_total_fragments);

            const uint64_t partial_size = _partial.number_of_antennas * 
                                          _partial.number_of_channels * 
                                          _partial.number_of_samples * 
                                          _partial.number_of_polarizations;

            CHECK_CUDA(cudaMallocHost(&fragmented_gpu_data, _total_fragments * sizeof(void*)));
            CHECK_CUDA(cudaMalloc(&defragmented_gpu_data, _total_fragments * partial_size * sizeof(std::complex<int8_t>)));
            CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

            _time_range.start = 0;
            _time_range.end = 0;

            _total_time_steps = _sample_slots;
        }

        ~Checkerboard() {
            CHECK_CUDA(cudaFreeHost(fragmented_gpu_data));
            CHECK_CUDA(cudaFree(defragmented_gpu_data));
            CHECK_CUDA(cudaStreamDestroy(stream));
        }

        inline bool in_range(const uint64_t& timestamp) const {
            return timestamp >= _time_range.start && timestamp < _time_range.end;
        }

        inline bool is_complete() const {
            return _fragment_counter == _total_fragments;
        }

        void rewind(const uint64_t& initial_timestamp) {
            _fragment_counter = 0;

            _time_range.start = initial_timestamp;
            _time_range.end = initial_timestamp + (_total_time_steps * _time_step_length);

            // TODO: Cache the cache vectors.
            std::vector<void*> cpu_pkts;
            cpu_pkts.reserve(_total_fragments);
            std::vector<void*> gpu_pkts;
            gpu_pkts.reserve(_total_fragments);

            for (auto& fragment : fragments) {
                if (!fragment.is_complete) {
                    continue;
                }

                cpu_pkts.push_back(fragment.cpu_packet);
                gpu_pkts.push_back(fragment.gpu_packet);

                if (fragment.burst.unique()) {
                    adv_net_free_rx_burst(fragment.burst);
                }
                fragment.burst.reset();

                fragment.is_complete = false;
            }

            if (!cpu_pkts.empty()) {
                adv_net_free_pkts(cpu_pkts.data(), cpu_pkts.size());
                adv_net_free_pkts(gpu_pkts.data(), gpu_pkts.size());
            }
        }

        bool add_fragment(const std::shared_ptr<AdvNetBurstParams>& burst,
                          const uint64_t& packet_id,
                          const VoltagePacket& packet) {
            uint64_t antenna_index = packet.antenna_id;
            uint64_t channel_index = packet.channel_number;
            uint64_t samples_index = (packet.timestamp - _time_range.start) / _time_step_length;

            if (antenna_index < _offset.number_of_antennas) {
                return false;
            }

            if (channel_index < _offset.number_of_channels) {
                return false;
            }

            if (samples_index < _offset.number_of_samples) {
                return false;
            }

            antenna_index -= _offset.number_of_antennas;
            channel_index -= _offset.number_of_channels;
            samples_index -= _offset.number_of_samples;

            antenna_index /= _partial.number_of_antennas;
            channel_index /= _partial.number_of_channels;

            if (antenna_index >= _antenna_slots) {
                return false;
            }

            if (channel_index >= _channel_slots) {
                return false;
            }

            if (samples_index >= _sample_slots) {
                return false;
            }

            uint64_t fragment_index = 0;
            fragment_index += antenna_index * _channel_slots * _sample_slots;
            fragment_index += channel_index * _sample_slots;
            fragment_index += samples_index;

            if (fragment_index >= _total_fragments) {
                HOLOSCAN_LOG_ERROR("Fragment index out of range: {} [{}, {}, {}] {} {}/{}", fragment_index, 
                                                                                            antenna_index,
                                                                                            channel_index, 
                                                                                            samples_index, 
                                                                                            packet.timestamp, 
                                                                                            _fragment_counter, 
                                                                                            _total_fragments);
            }

            fragments[fragment_index].is_complete = true;
            fragments[fragment_index].burst = burst;
            fragments[fragment_index].cpu_packet = burst->cpu_pkts[packet_id];
            fragments[fragment_index].gpu_packet = burst->gpu_pkts[packet_id];

            fragmented_gpu_data[fragment_index] = adv_net_get_gpu_pkt_ptr(burst, packet_id);

            _fragment_counter++;
            return true;
        }

        inline bool is_processing() const {
            return cudaStreamQuery(stream) != cudaSuccess;
        }

        void defragment() {
            const auto pA = _partial.number_of_antennas;
            const auto pF = _partial.number_of_channels;
            const auto pT = _partial.number_of_samples;
            const auto pP = _partial.number_of_polarizations;

            const auto tA = _total.number_of_antennas;
            const auto tF = _total.number_of_channels;
            const auto tT = _total.number_of_samples;
            const auto tP = _total.number_of_polarizations;

            CHECK_CUDA(DefragmentationKernel(defragmented_gpu_data, 
                                             fragmented_gpu_data, 
                                             _total_fragments, 
                                             stream, 
                                             {pA, pF, pT, pP}, 
                                             {tA, tF, tT, tP}));
        }

     private:
        BlockShape _total;
        BlockShape _partial;
        BlockShape _offset;

        uint64_t _fragment_counter;
        uint64_t _total_fragments;
        uint64_t _time_step_length;
        uint64_t _total_time_steps;

        uint64_t _antenna_slots;
        uint64_t _channel_slots;
        uint64_t _sample_slots;

        TimeRange _time_range;

        std::vector<Fragment> fragments;
        void** fragmented_gpu_data;
        void* defragmented_gpu_data;
        cudaStream_t stream;
    };

    std::vector<std::shared_ptr<Checkerboard>> checkerboard_pool;
    std::queue<std::shared_ptr<Checkerboard>> defragment_queue;
    std::vector<std::shared_ptr<Checkerboard>> checkerboards;

    uint64_t block_seed_time;
    uint64_t start_timestamp;
    uint64_t end_timestamp;
    uint64_t block_time_length;

    Parameter<BlockShape> total_block_;
    Parameter<BlockShape> partial_block_;
    Parameter<BlockShape> offset_block_;

    std::thread report_thread;
    bool report_thread_running;
    std::chrono::time_point<std::chrono::system_clock> last_report_time;
    uint64_t obsolete_rewinds_counter;
    uint64_t complete_rewinds_counter;
    uint64_t packet_counter;
    uint64_t last_report_packet_counter;
    
    Parameter<uint64_t> concurrent_blocks_;
    Parameter<uint64_t> time_step_length_;
    Parameter<uint16_t> transport_header_size_;
    Parameter<uint16_t> voltage_header_size_;

    // TODO: It would be nice if this data could be made available to a monitoring system.
    void report_thread_callback() {
        HOLOSCAN_LOG_INFO("Rewinds: Obsolete: {}, Complete: {}, Packets: {} ({})", obsolete_rewinds_counter, 
                                                                                    complete_rewinds_counter,
                                                                                    packet_counter,
                                                                                    packet_counter - 
                                                                                    last_report_packet_counter);

        HOLOSCAN_LOG_INFO("Checkerboards state ({}, {}):", defragment_queue.size(), checkerboards.size());
        for (uint64_t i = 0; i < concurrent_blocks_; i++) {
            const auto& c = checkerboard_pool[i];

            HOLOSCAN_LOG_INFO("  - Checkerboard {}: [{} - {}] [{}] {}/{} ", i, c->time_range().start,
                                                                               c->time_range().end,
                                                                               c->is_processing() ? "PROC" : "IDLE",
                                                                               c->fragment_counter(),
                                                                               c->total_fragments());
        }

        last_report_packet_counter = packet_counter;
    }
};

}  // namespace holoscan::ops



class App : public holoscan::Application {
 public:
    void compose() override {
        using namespace holoscan;
        HOLOSCAN_LOG_INFO("Initializing Allen Telescope Array Holoscan interface.");
    
        auto ata_transport_rx = make_operator<ops::AtaTransportOpRx>("ata_transport_rx", 
                                                                     from_config("ata_transport_rx"));
        auto adv_net_rx = make_operator<ops::AdvNetworkOpRx>("adv_network_rx",
                                                             from_config("advanced_network"),
                                                             make_condition<BooleanCondition>("is_alive", true));

        add_flow(adv_net_rx, ata_transport_rx, {{"bench_rx_out", "burst_in"}});
    }
};

int main(int argc, char** argv) {
    auto app = holoscan::make_application<App>();

    if (argc < 2) {
        HOLOSCAN_LOG_ERROR("Usage: {} config_file", argv[0]);
        return -1;
    }

    auto config_path = std::filesystem::canonical(argv[0]).parent_path();
    config_path += "/" + std::string(argv[1]);
    app->config(config_path);

    using Scheduler = holoscan::MultiThreadScheduler;
    const auto& scheduler = app->make_scheduler<Scheduler>("multithread-scheduler", 
                                                           app->from_config("scheduler"));
    app->scheduler(scheduler);
    app->run();

    return 0;
}
#include <chrono>
#include <filesystem>
#include <future>
#include <memory>
#include <string>
#include <thread>

#include <cyberbridge/holoscan.hh>
#include <jetstream/memory/utils/juggler.hh>

#include <holoscan/holoscan.hpp>

#include "logo.hh"

using namespace Jetstream;

//
// DSP Ping (CyberEther + Holoscan Interop)
//
// This is a simple example that demonstrates how to use CyberEther and Holoscan
// together. A simple Holoscan app is created with SineTxOp, LogoOp, NoiseOp, and DummyRxOp.
// Inside a headless instance of CyberEther interconnected with the Holoscan app via CyberBridge,
// a HoloscanSource block is created to receive data from the PingTxOp. The data is then 
// displayed in a constellation diagram.
//

namespace holoscan::ops {

template<typename T>
class DummyRxOp : public Operator {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(DummyRxOp)

    DummyRxOp() = default;

    using TensorType = Jetstream::Tensor<Device::CPU, T>;

    void setup(OperatorSpec& spec) override { 
        spec.input<std::shared_ptr<TensorType>>("in");
    }

    void compute(InputContext& op_input, OutputContext&, ExecutionContext&) override {
        op_input.receive<std::shared_ptr<TensorType>>("in").value();
    };
};

class SineTxOp : public Operator {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(SineTxOp)

    SineTxOp() = default;

    using TensorType = Jetstream::Tensor<Device::CPU, CF32>;

    void setup(OperatorSpec& spec) override { 
        spec.output<std::shared_ptr<TensorType>>("out"); 
    }

    void start() {
        juggler.resize(5, std::vector<U64>{8, 2048});
    }

    void compute(InputContext&, OutputContext& op_output, ExecutionContext&) override {
        std::shared_ptr<TensorType> ptr;

        // Get a buffer from the pool.

        while ((ptr = juggler.get()) == nullptr) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        // Generate a sine wave.

        auto& tensor = *ptr;
        const F32 w = 2.0 * JST_PI * 10;

        for (U64 j = 0; j < tensor.shape(0); j++) {
            for (U64 i = 0; i < tensor.shape(1); ++i) {
                const F32 t = i / static_cast<F32>(tensor.shape(1));

                const F32 real_part = std::cos(w * t);
                const F32 imag_part = std::sin(w * t);

                tensor[{j, i}] = {real_part, imag_part};
            }
        }

        // Send the buffer to the output.

        JST_DEBUG("Sine buffer sent: {}", tensor.shape());
        op_output.emit(ptr, "out");
    };

 private:
    Memory::Juggler<TensorType> juggler;
};

class NoiseOp : public Operator {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(NoiseOp)

    NoiseOp() = default;

    using TensorType = Jetstream::Tensor<Device::CPU, CF32>;

    void setup(OperatorSpec& spec) override { 
        spec.input<std::shared_ptr<TensorType>>("in");
        spec.output<std::shared_ptr<TensorType>>("out"); 
    }

    void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override {
        auto in_value = op_input.receive<std::shared_ptr<TensorType>>("in").value();

        // Apply noise to the input buffer.

        auto& tensor = *in_value;
        
        for (U64 j = 0; j < tensor.shape(0); j++) {
            for (U64 i = 0; i < tensor.shape(1); ++i) {
                tensor[{j, i}] = {
                    tensor[{j, i}].real() + noise() * 0.1f,
                    tensor[{j, i}].imag() + noise() * 0.1f
                };
            }
        }

        op_output.emit(in_value, "out");
    };

    static F32 noise() {
        return 2.0f * (std::rand() / static_cast<F32>(RAND_MAX)) - 1.0f;
    }
};

class LogoOp : public Operator {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(LogoOp)

    LogoOp() = default;

    using TensorType = Jetstream::Tensor<Device::CPU, CF32>;

    void setup(OperatorSpec& spec) override {
        spec.input<std::shared_ptr<TensorType>>("in");
        spec.output<std::shared_ptr<TensorType>>("out"); 
    }

    void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override {
        auto in_value = op_input.receive<std::shared_ptr<TensorType>>("in").value();

        // Apply the logo to the input buffer.

        auto& tensor = *in_value;

        for (U64 i = 0; i < (logo_len / sizeof(CF32)); i++) {
            tensor[i] = reinterpret_cast<CF32*>(logo_raw)[i];
            tensor[i] = {
                tensor[i].real() + noise() * 0.01f,
                tensor[i].imag() + noise() * 0.01f,
            };
        }

        op_output.emit(in_value, "out");
    };

    static F32 noise() {
        return 2.0f * (std::rand() / static_cast<F32>(RAND_MAX)) - 1.0f;
    }
};

}  // namespace holoscan::ops

class PingApp : public holoscan::Application {
 public:
    void compose() override {
        using namespace holoscan;
        using namespace std::chrono_literals;

        const auto& condition = make_condition<PeriodicCondition>("periodic-condition", 100ms);
        auto sine_tx = make_operator<ops::SineTxOp>("sine-tx", condition);
        auto sine_noise = make_operator<ops::NoiseOp>("sine-noise");
        auto logo_draw = make_operator<ops::LogoOp>("logo-draw");
        auto rx = make_operator<ops::DummyRxOp<CF32>>("dummy-rx");

        add_flow(sine_tx, sine_noise);
        add_flow(sine_noise, logo_draw);
        add_flow(logo_draw, rx);
    }

    static std::shared_ptr<PingApp> Factory() {
        auto app = holoscan::make_application<PingApp>();

        auto config_path = std::filesystem::path("./holoscan.yaml");
        app->config(config_path);

        return app;
    }
};



//
// CyberEther and Holoscan Instantiation
//

int main() {
    // Initialize CUDA context.

    if (Backend::Initialize<Device::CUDA>({}) != Result::SUCCESS) {
        JST_FATAL("Cannot initialize compute backend.");
        return 1;
    }

    // Initialize Holoscan.

    JST_CHECK_THROW(CyberBridge::Holoscan::RegisterApp(PingApp::Factory));

    CyberBridge::Holoscan::StartApp();

    // Initialize CyberEther.
    //
    // This will create a simple headless interface that can be accessed
    // remotelly via another CyberEther instance. This is nice because the
    // application can run on a server and just the frame is sent to the
    // client. This also uses NVENC to encode the Vulkan frame, so it is
    // very efficient.
    //
    // This will also register the special CyberBridge blocks that are used
    // to interface with Holoscan.

    Backend::Config backendConfig;
    backendConfig.headless = true;
    backendConfig.deviceId = 0;

    Viewport::Config viewportConfig;
    viewportConfig.endpoint = "0.0.0.0:5002";
    viewportConfig.codec = Render::VideoCodec::H264;

    CyberBridge::Holoscan::StartRender("./cyberether.yml", backendConfig, viewportConfig, [&]{
        if (!CyberBridge::Holoscan::IsAppRunning()) {
            ImGui::Text("Holoscan app is not running.");
            return Result::SUCCESS;
        }

        auto& app = CyberBridge::Holoscan::GetApp();
        const auto& nodes = app.graph().get_nodes();

        if (ImGui::Begin("Holoscan Nodes")) {
            ImGui::Text("Number of nodes: %ld", nodes.size());

            for (const auto& node : nodes) {
                if (ImGui::TreeNode(node->name().c_str())) {
                    ImGui::Text("Node ID: %ld", node->id());
                    
                    ImGui::Text("Number of inputs: %ld", node->spec()->inputs().size());
                    for (const auto& [name, spec] : node->spec()->inputs()) {
                        ImGui::BulletText("Input Name: %s", name.c_str());
                    }

                    ImGui::Text("Number of outputs: %ld", node->spec()->outputs().size());
                    for (const auto& [name, spec] : node->spec()->outputs()) {
                        ImGui::BulletText("Output Name: %s", name.c_str());
                    }
                    ImGui::TreePop();
                }
            }
        }
        ImGui::End();

        return Result::SUCCESS;
    });

    // Deintialize CUDA context.

    Backend::DestroyAll();

    return 0;
}

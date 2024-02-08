#include <chrono>
#include <filesystem>
#include <future>
#include <memory>
#include <string>
#include <thread>

#include <cyberbridge/holoscan.hh>
#include <jetstream/memory/utils/juggler.hh>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/inference/inference.hpp>


#include <SoapySDR/Device.hpp>
#include <SoapySDR/Types.hpp>
#include <SoapySDR/Formats.hpp>
#include <SoapySDR/Modules.hpp>

using namespace Jetstream;

//
// SoapySDR FM + Neural Demodulation (SoapySDR + CyberEther + Holoscan Interop)
//
// This is a more complex example that demonstrates how to receive data from a SDR using SoapySDR
// wrapped in a Holoscan operator, and then display the data in CyberEther. This is literally
// demodulating FM radio complex-valued IQ samples using a neural network (CursedNet). This example 
// demonstrates how to use TensorRT to accelerate the inference of a neural network inside a Holoscan 
// operator while receiving data from a SDR using SoapySDR. As usual, the demolated audio is sent to 
// CyberEther for visualization and playback.
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

class SoapyTxOp : public Operator {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(SoapyTxOp)

    SoapyTxOp() = default;

    using TensorType = Jetstream::Tensor<Device::CPU, CF32>;

    void setup(OperatorSpec& spec) override { 
        spec.output<std::shared_ptr<TensorType>>("out");

        spec.param(deviceName, "device");
        spec.param(sampleRate, "sample_rate");
        spec.param(centerFrequency, "center_frequency");
        spec.param(agcMode, "agc");
    }

    void start() {
        // Allocate rotary buffer.

        juggler.resize(5, std::vector<U64>{1, 256000});

        // Initialize SoapySDR.

        device = SoapySDR::Device::make(deviceName);

        device->setSampleRate(SOAPY_SDR_RX, 0, sampleRate);
        device->setFrequency(SOAPY_SDR_RX, 0, centerFrequency);
        device->setGainMode(SOAPY_SDR_RX, 0, agcMode);

        // Start streaming.

        rx_stream = device->setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32);
        if (rx_stream == nullptr) {
            JST_ERROR("Failed to setup RX stream.");
            JST_CHECK_THROW(Result::ERROR);
        }
        device->activateStream(rx_stream, 0, 0, 0);
    }

    void compute(InputContext&, OutputContext& op_output, ExecutionContext&) override {
        std::shared_ptr<TensorType> ptr;

        // Get a buffer from the pool.

        while ((ptr = juggler.get()) == nullptr) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        // Collect samples from the SDR.

        auto& tensor = *ptr;

        U64 samples = 0;

        while (samples < tensor.size()) {
            void* buffs[] = { tensor.data() + samples };
            int flags;
            long long time_ns;
            int ret;

            if ((ret = device->readStream(rx_stream, buffs, 1024, flags, time_ns, 1e5)) < 0) {
                JST_ERROR("Failed to read from RX stream.");
                JST_CHECK_THROW(Result::ERROR);
            }

            samples += ret;
        }

        // Send the buffer to the output.

        op_output.emit(ptr, "out");
    };

    void stop() {
        device->deactivateStream(rx_stream, 0, 0);
        device->closeStream(rx_stream);
        SoapySDR::Device::unmake(device);
    }

 private:
    Memory::Juggler<TensorType> juggler;

    SoapySDR::Device* device;
    SoapySDR::Stream* rx_stream;

    Parameter<std::string> deviceName;
    Parameter<F64> sampleRate;
    Parameter<F64> centerFrequency;
    Parameter<bool> agcMode;
};

class FmDemOp : public Operator {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(FmDemOp)

    FmDemOp() = default;

    using InputTensorType = Jetstream::Tensor<Device::CPU, CF32>;
    using OutputTensorType = Jetstream::Tensor<Device::CPU, F32>;

    void setup(OperatorSpec& spec) override { 
        spec.input<std::shared_ptr<InputTensorType>>("in");
        spec.output<std::shared_ptr<OutputTensorType>>("out");

        spec.param(sampleRate, "sample_rate");
    }

    void start() {
        // Allocate rotary buffer.

        juggler.resize(5, std::vector<U64>{1, 256000});

        // Initialize constants.

        kf = 100e3f / sampleRate;
        ref = 1.0f / (2.0f * JST_PI * kf);
    }

    void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override {
        auto input_ptr = op_input.receive<std::shared_ptr<InputTensorType>>("in").value();
        std::shared_ptr<OutputTensorType> output_ptr;

        // Get a buffer from the pool.

        while ((output_ptr = juggler.get()) == nullptr) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        // Demodulate the FM signal.

        auto& input_tensor = *input_ptr;
        auto& output_tensor = *output_ptr;

        for (U64 n = 1; n < input_tensor.size(); n++) {
            output_tensor[n] = std::arg(std::conj(input_tensor[n - 1]) * input_tensor[n]) * ref;
        }

        // Send the buffer to the output.

        op_output.emit(output_ptr, "out");
    };

 private:
    Memory::Juggler<OutputTensorType> juggler;

    F32 kf;
    F32 ref;

    Parameter<F64> sampleRate;
};

class ModelPreprocessor : public Operator {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(ModelPreprocessor)

    ModelPreprocessor() = default;

    using InputTensorType = Jetstream::Tensor<Device::CPU, CF32>;

    void setup(OperatorSpec& spec) override { 
        spec.input<std::shared_ptr<InputTensorType>>("in");
        spec.output<gxf::Entity>("out");

        spec.param(allocator_, "allocator");
    }

    void start() {
        buffer = Jetstream::Tensor<Device::CPU, F32>({1, 2, 256000});
        MapOn<Device::CUDA>(buffer);
        hol_buffer = CyberBridge::Holoscan::TensorToHoloscan(buffer);
    }

    void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& context) override {
        auto input_ptr = op_input.receive<std::shared_ptr<InputTensorType>>("in").value();

        // Cast input tensor to CursedNet input type.

        auto& input_tensor = *input_ptr;
        auto& output_tensor = buffer;

        for (U64 i = 0; i < buffer.shape(2); i++) {
            output_tensor[i] = input_tensor[i].real();
            output_tensor[i + buffer.shape(2)] = input_tensor[i].imag();
        }

        // Send the buffer to the output.

        auto out_message = gxf::Entity::New(&context);
        out_message.add(hol_buffer, "input");
        op_output.emit(out_message, "out");
    };

 private:
    Jetstream::Tensor<Device::CPU, F32> buffer;
    std::shared_ptr<holoscan::Tensor> hol_buffer;

    Parameter<std::shared_ptr<Allocator>>  allocator_;
};

class ModelPostprocessor : public Operator {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(ModelPostprocessor)

    ModelPostprocessor() = default;

    using OutputTensorType = Jetstream::Tensor<Device::CPU, F32>;

    void setup(OperatorSpec& spec) override { 
        spec.input<gxf::Entity>("in");
        spec.output<std::shared_ptr<OutputTensorType>>("out");
    }

    void start() {
        buffer = std::make_shared<OutputTensorType>(std::vector<U64>{1, 32000});
    }

    void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override {
        auto in_message = op_input.receive<gxf::Entity>("in").value();
        auto in_tensor = in_message.get<holoscan::Tensor>("output");

        // Copy the tensor to the buffer.

        cudaMemcpy(buffer->data(), in_tensor->data(), buffer->size_bytes(), cudaMemcpyHostToHost);

        // Send the buffer to the output.

        JST_INFO("FF")

        op_output.emit(buffer, "out");
    };

 private:
    std::shared_ptr<OutputTensorType> buffer;
};

}  // namespace holoscan::ops

class SoapyApp : public holoscan::Application {
 public:
    void compose() override {
        using namespace holoscan;
        using namespace std::chrono_literals;

        auto sdr_tx = make_operator<ops::SoapyTxOp>("sdr-tx", from_config("sdr_rx"));
        auto fm_dem = make_operator<ops::FmDemOp>("fm-dem", from_config("fm_dem"));
        auto sdr_rx = make_operator<ops::DummyRxOp<CF32>>("sdr-dummy-rx");
        auto fm_rx = make_operator<ops::DummyRxOp<F32>>("fm-dummy-rx");

        std::shared_ptr<Resource> pool = make_resource<UnboundedAllocator>("pool");

        auto model_preprocessor = make_operator<ops::ModelPreprocessor>("model-preprocessor", 
            Arg("allocator") = pool);
        auto model_inference = make_operator<ops::InferenceOp>("model-inference",
            from_config("model_inference"),
            Arg("allocator") = pool);
        auto model_postprocessor = make_operator<ops::ModelPostprocessor>("model-postprocessor");
        auto model_rx = make_operator<ops::DummyRxOp<F32>>("model-dummy-rx");

        add_flow(sdr_tx, sdr_rx);
        add_flow(sdr_tx, fm_dem);
        add_flow(fm_dem, fm_rx);

        add_flow(sdr_tx, model_preprocessor);
        add_flow(model_preprocessor, model_inference, {{"out", "receivers"}});
        add_flow(model_inference, model_postprocessor, {{"transmitter", "in"}});
        add_flow(model_postprocessor, model_rx);
    }

    static std::shared_ptr<SoapyApp> Factory() {
        auto app = holoscan::make_application<SoapyApp>();

        app->config(std::filesystem::path("./holoscan.yaml"));

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

    JST_CHECK_THROW(CyberBridge::Holoscan::RegisterApp(SoapyApp::Factory));

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

    CyberBridge::Holoscan::StartRender("", backendConfig, viewportConfig, [&]{
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

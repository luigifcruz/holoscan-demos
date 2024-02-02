#include <mutex>
#include <vector>
#include <future>
#include <string>
#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>
#include <condition_variable>

#include "holoscan/holoscan.hpp"
#include "holoscan/operators/inference/inference.hpp"

#include <jetstream/base.hh>
#include <jetstream/block.hh>
#include <jetstream/instance.hh>
#include <jetstream/store.hh>
#include <jetstream/blocks/manifest.hh>

#include <SoapySDR/Device.hpp>
#include <SoapySDR/Types.hpp>
#include <SoapySDR/Formats.hpp>
#include <SoapySDR/Modules.hpp>

#include "logo.hh"

using namespace Jetstream;



//
// Extensions
//
// This is a collection of classes that I felt were missing from CyberEther, 
// BLADE, or Holoscan. They should be ultimately upstreamed somewhere.
//

namespace Jetstream {

template<typename T>
class Juggler {
 public:
    // I didn't find anything to reuse memory in the Holoscan documentation. 
    // So, I'm writing my own implementation.

    Juggler() = default;

    template<typename... Args>
    Juggler(const U64& size, Args&&... args) {
        resize(size, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void resize(const U64& size, Args&&... args) {
        clear();
        pool.reserve(size);
        used.reserve(size);
        for (U64 i = 0; i < size; ++i) {
            pool.push_back(std::make_shared<T>(std::forward<Args>(args)...));
        }
    }

    void clear() {
        pool.clear();
        used.clear();
    }

    std::shared_ptr<T> get() {
        // Recycle unused pointers.

        for (auto it = used.begin(); it != used.end();) {
            if ((*it).unique()) {
                pool.push_back(*it);
                it = used.erase(it);
            } else {
                ++it;
            }
        }

        // Check if there are any pointers available.

        if (pool.empty()) {
            return nullptr;
        }

        // Get the pointer from the pool.

        auto ptr = pool.back();
        pool.pop_back();

        // Add the pointer to the used list.

        used.push_back(ptr);

        // Return the pointer to caller.

        return ptr;
    }

 private:
    std::vector<std::shared_ptr<T>> pool;
    std::vector<std::shared_ptr<T>> used;
};

template<typename T>
class TensorCircularBuffer {
 public:
    TensorCircularBuffer() = default;

    template<typename... Args>
    TensorCircularBuffer(const U64& size, Args&&... args) {
        resize(size, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void resize(const U64& size, Args&&... args) {
        clear();
        pool.reserve(size);
        for (U64 i = 0; i < size; ++i) {
            pool.push_back(std::forward<Args>(args)...);
        }
    }

    void clear() {
        pool.clear();
        head = 0;
        tail = 0;
    }

    U64 capacity() const {
        return pool.size();
    }

    U64 occupancy() const {
        return tail - head;
    }

    bool empty() const {
        return head == tail;
    }

    bool full() const {
        return occupancy() == capacity();
    }

    const U64& overflows() const {
        return overflowCount;
    }

    bool wait(const U64& timeout = 1000) {
        std::unique_lock<std::mutex> sync(mtx);
        while (empty()) {
            if (cv.wait_for(sync, std::chrono::milliseconds(timeout)) == std::cv_status::timeout) {
                return false;
            }
        }
        return true;
    }

    bool get(T& buffer, const U64& timeout = 1000) {
        if (empty()) {
            return false;
        }

        if (!wait(timeout)) {
            return false;
        }

        {
            std::lock_guard<std::mutex> guard(mtx);
            buffer = pool[head % capacity()];
            ++head;
        }

        return true;
    }

    void put(const std::function<void(T&)>& callback) {
        {
            std::lock_guard<std::mutex> guard(mtx);

            if (full()) {
                ++overflowCount;
                ++head;
            }

            callback(pool[tail % capacity()]);
            ++tail;
        }

        cv.notify_one();
    }

 private:
    std::vector<T> pool;
    U64 head = 0;
    U64 tail = 0;
    U64 overflowCount = 0;

    std::mutex mtx;
    std::condition_variable cv;
};

template<Device D, typename T>
inline std::shared_ptr<holoscan::Tensor> TensorToHoloscan(Tensor<D, T>& tensor) {
    // Convert type.
    // NOTE: Looks like GXF still doesn't support complex numbers.

    nvidia::gxf::PrimitiveType type;

    if constexpr (std::is_same<T, F32>::value) {
        type = nvidia::gxf::PrimitiveType::kFloat32;
    } else {
        JST_ERROR("Unsupported data type.");
        JST_CHECK_THROW(Result::ERROR);
    }

    // Convert device.

    nvidia::gxf::MemoryStorageType device;

    if constexpr (D == Device::CPU) {
        // NOTE: Assuming all Jetstream::Tensor<Device::CPU> are pinned memory.
        device = nvidia::gxf::MemoryStorageType::kHost;
    } else if constexpr (D == Device::CUDA) {
        device = nvidia::gxf::MemoryStorageType::kDevice;
    } else {
        JST_ERROR("Unsupported device.");
        JST_CHECK_THROW(Result::ERROR);
    }

    // Convert shape.

    std::vector<I32> tmp;
    for (const auto& dim : tensor.shape()) {
        tmp.push_back(static_cast<I32>(dim));
    }
    nvidia::gxf::Shape shape(tmp);

    // Convert stride.

    const auto stride = nvidia::gxf::ComputeTrivialStrides(shape, sizeof(T));

    // Convert data.

    void* data = static_cast<void*>(tensor.data());

    // Create deallocator.

    auto dealloc = [buffer = tensor](void*) {
        (void)buffer;
        return nvidia::gxf::Success;
    };

    // Create GXF tensor.

    auto tg = std::make_shared<nvidia::gxf::Tensor>();
    tg->wrapMemory(shape, type, sizeof(T), stride, device, data, dealloc);
    return holoscan::gxf::GXFTensor(*tg).as_tensor();
}

}  // namespace Jetstream



//
// CyberBridge (Holoscan)
//
// This is a appropriately named class that bridges the gap between CyberEther
// and Holoscan. It is used to shuttle data between the two frameworks.
//

namespace Jetstream {

class HoloscanBridge {
 public:
    HoloscanBridge(HoloscanBridge const&) = delete;
    void operator=(HoloscanBridge const&) = delete;

    typedef std::function<std::shared_ptr<holoscan::Application>()> AppFactory;

    static HoloscanBridge& GetInstance();

    static bool IsAppRunning() {
        return GetInstance().isAppRunning();
    }

    static holoscan::Application& GetApp() {
        return GetInstance().getApp();
    }

    static Result RegisterApp(const AppFactory& factory) {
        return GetInstance().registerApp(factory);
    }

    static Result RegisterTap(const std::string& name, std::function<Result()> tap) {
        return GetInstance().registerTap(name, tap);
    }

    static Result RemoveTap(const std::string& name) {
        return GetInstance().removeTap(name);
    }

    static Result LockContext(std::function<Result()> callback) {
        return GetInstance().lockContext(callback);
    }

    static Result StartApp() {
        return GetInstance().startApp();
    }

    static Result StopApp() {
        return GetInstance().stopApp();
    }

 private:
    HoloscanBridge() = default;

    std::mutex mtx;
    AppFactory factory;
    std::shared_ptr<holoscan::Application> app;
    std::future<Result> holoscanThreadHandler;
    std::unordered_map<std::string, std::function<Result()>> tapList;

    bool isAppRunning();
    holoscan::Application& getApp();
    Result registerApp(const AppFactory& factory);
    Result registerTap(const std::string& name, std::function<Result()> tap);
    Result removeTap(const std::string& name);
    Result lockContext(std::function<Result()> callback);
    Result startApp();
    Result stopApp();
};

HoloscanBridge& HoloscanBridge::GetInstance() {
    static HoloscanBridge instance;
    return instance;
}

bool HoloscanBridge::isAppRunning() {
    if (!app) {
        return false;
    }

    if (!holoscanThreadHandler.valid()) {
        return false;
    }

    if (holoscanThreadHandler.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
        return false;
    }

    return true;
}

holoscan::Application& HoloscanBridge::getApp() {
    if (!app) {
        JST_ERROR("[BRIDGE] Holoscan app is not registered.");
        JST_CHECK_THROW(Result::ERROR);
    }

    return *app;
}

Result HoloscanBridge::registerApp(const std::function<std::shared_ptr<holoscan::Application>()>& factory) {
    this->factory = factory;
    this->app = this->factory();
    return Result::SUCCESS;
}

Result HoloscanBridge::registerTap(const std::string& name, std::function<Result()> tap) {
    std::lock_guard<std::mutex> guard(mtx);

    if (tapList.contains(name)) {
        JST_ERROR("[BRIDGE] Tap '{}' already exists.", name);
        return Result::ERROR;
    }

    tapList[name] = tap;

    Result result = Result::SUCCESS;

    if (isAppRunning()) {
        JST_CHECK(stopApp());

        for (const auto& [id, tap] : tapList) {
            const auto& res = tap();

            if (res != Result::SUCCESS && id == name) {
                result = res;
            }
        }

        JST_CHECK(startApp());
    }

    if (result != Result::SUCCESS) {
        tapList.erase(name);
    }

    return result;
}

Result HoloscanBridge::removeTap(const std::string& name) {
    std::lock_guard<std::mutex> guard(mtx);

    if (!tapList.contains(name)) {
        JST_ERROR("[BRIDGE] Tap '{}' does not exist.", name);
        return Result::ERROR;
    }

    tapList.erase(name);

    if (isAppRunning()) {
        JST_CHECK(stopApp());

        for (const auto& [name, tap] : tapList) {
            JST_CHECK(tap());
        }

        JST_CHECK(startApp());
    }

    return Result::SUCCESS;
}

Result HoloscanBridge::lockContext(std::function<Result()> callback) {
    if (!isAppRunning()) {
        JST_ERROR("[BRIDGE] Holoscan app is not running.");
        return Result::ERROR;
    }

    {
        std::lock_guard<std::mutex> guard(mtx);
        JST_CHECK(callback());
    }

    return Result::SUCCESS;
}

Result HoloscanBridge::startApp() {
    if (isAppRunning()) {
        JST_ERROR("[BRIDGE] Holoscan app is still running.");
        return Result::ERROR;
    }

    holoscanThreadHandler = std::async(std::launch::async, [&](){
        JST_INFO("[BRIDGE] Starting Holoscan app.");

        try {
            std::future<void> future;
            future = app->run_async();
            future.wait();
        } catch (const std::exception& e) {
            JST_ERROR("[BRIDGE] Holoscan app has crashed: {}", e.what());
            return Result::ERROR;
        }

        JST_INFO("[BRIDGE] Holoscan app has stopped.");
        return Result::SUCCESS;
    });

    // TODO: This is a hack to wait for the Holoscan executor to start.
    //       It would be nice to interface directly with GXF rather than
    //       these unreliable heuristics. But I can't find any public API
    //       to do so.

    
    for (const auto& node : app->graph().get_nodes()) {
        while (true) {
            if (node->id() >= 0) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    // TODO: This is still returning before all the nodes are scheduled.

    return Result::SUCCESS;
}

Result HoloscanBridge::stopApp() {
    if (!isAppRunning()) {
        JST_ERROR("[BRIDGE] Holoscan app is not running.");
        return Result::ERROR;
    }

    JST_INFO("[BRIDGE] Stopping Holoscan app.");

    // TODO: This is a hack to wait for the app to stop. It wouldn't be necessary
    //       if the interrupt method returned an error code.

    while (isAppRunning()) {
        app->executor().interrupt();
        holoscanThreadHandler.wait_for(std::chrono::milliseconds(50));
    }
    
    // TODO: It would be nicer to keep the internal graph state and reload GXF
    //       instead of destroying and recreating it. But this would require
    //       modifying the Holoscan API.
    // 
    //       Something like `holoscan/core/fragment.cpp`:
    //       ```
    //       void Fragment::reload_context() {
    //           if (graph_) { graph_.reset(); }
    //           if (executor_) { executor_.reset(); }
    //           if (scheduler_) { scheduler_.reset(); }
    //           if (network_context_) { network_context_.reset(); }
    //       }
    //       ```
    //       
    //       Keeping this "hack" for now:
    app.reset();
    app = factory();
    app->compose_graph();

    return Result::SUCCESS;
}

}  // namespace Jetstream



//
// CyberEther Modules for CyberBridge (Holoscan)
//
// These modules are called by the blocks to interface with Holoscan using CyberBridge.
// They are used to register and remove taps from the Holoscan graph. The taps are
// registered when the block is created and removed when the block is destroyed.
//

namespace Jetstream {

// TODO: Implement `HoloscanSource` module.

template<Device D, typename T = CF32>
class HoloscanSource : public Module, public Compute {
 public:
    // Configuration 

    struct Config {
        std::string nodeName;
        std::string nodeOutputName;

        JST_SERDES(nodeName, nodeOutputName);
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        JST_SERDES_INPUT();
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        Tensor<D, T> buffer;

        JST_SERDES_OUTPUT(buffer);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Tensor<D, T>& getOutputBuffer() const {
        return this->output.buffer;
    }

    // Taint & Housekeeping

    constexpr Device device() const {
        return D;
    }

    void info() const final {
        JST_INFO("  Node name: {}", config.nodeName);
        JST_INFO("  Node output name: {}", config.nodeOutputName);
    }

    // Constructor

    Result create() {
        JST_DEBUG("Initializing Holoscan Source module.");
        JST_INIT_IO();

        // Validate config values.

        if (config.nodeName.empty()) {
            JST_ERROR("Node name is empty.");
            return Result::ERROR;
        }

        if (config.nodeOutputName.empty()) {
            JST_ERROR("Node output name is empty.");
            return Result::ERROR;
        }

        // Check if the app is running.

        if (!HoloscanBridge::IsAppRunning()) {
            JST_ERROR("Holoscan app is not running.");
            return Result::ERROR;
        }

        // Register the tap.

        holoscanModuleId = jst::fmt::format("cyberbridge-{}", locale());

        JST_CHECK(HoloscanBridge::RegisterTap(holoscanModuleId, [&](){
            auto& app = HoloscanBridge::GetApp();

            // Check if the node exists.

            const auto& node = app.graph().find_node(config.nodeName);
            if (!node) {
                JST_ERROR("Node '{}' does not exist.", config.nodeName);
                return Result::ERROR;
            }

            // Check if the node output exists.

            if (!node->spec()->outputs().contains(config.nodeOutputName)) {
                JST_ERROR("Node output '{}' does not exist.", config.nodeOutputName);
                return Result::ERROR;
            }

            // NOTE: CyberBridge will assume the input data is composed of Jetstream::Tensor
            //       objects. Other data types can be supported in the future without too much
            //       effort. For now, this is enough.

            // Check node output data type and device.

            const auto& spec = node->spec()->outputs().at(config.nodeOutputName);
            if (*spec->typeinfo() != typeid(std::shared_ptr<Tensor<D, T>>)) {
                JST_ERROR("Node output '{}' is not of type Tensor<{}, {}>.", config.nodeOutputName,
                                                                             GetDevicePrettyName(D),
                                                                             NumericTypeInfo<T>::name);
                return Result::ERROR;
            }

            // Create the operator.

            const auto& tx = app.graph().find_node(config.nodeName);
            const auto& txName = holoscanModuleId.c_str();
            op = app.make_operator<CyberBridgeOp>(txName);

            app.add_flow(tx, op, {{config.nodeOutputName, "in"}});

            return Result::SUCCESS;
        }));

        // Allocate the output buffer.

        output.buffer = Tensor<D, T>({1, 32000});

        return Result::SUCCESS;
    }

    Result destroy() {
        JST_DEBUG("Destroying Holoscan Source module.");

        JST_CHECK(HoloscanBridge::RemoveTap(holoscanModuleId));

        return Result::SUCCESS;
    }

 protected:
    Result computeReady() {
        if (!op->buffer.wait()) {
            return Result::TIMEOUT;
        }
        return Result::SUCCESS;
    }

    Result compute(const RuntimeMetadata& meta) final {
        // Running inside a locked context because the operator can 
        // be reloaded at any time by the bridge.
        return HoloscanBridge::LockContext([&]{
            Tensor<D, T> tensor;

            if (!op->buffer.get(tensor)) {
                return Result::SKIP;
            }
            
            // TODO: Replace by proper copy.
            if constexpr (D == Device::CPU) {
                for (U64 i = 0; i < tensor.size(); ++i) {
                    output.buffer.data()[i] = tensor.data()[i];
                }
            }

            return Result::SUCCESS;
        });
    }

    class CyberBridgeOp : public holoscan::Operator {
     public:
        HOLOSCAN_OPERATOR_FORWARD_ARGS(CyberBridgeOp)

        CyberBridgeOp() = default;

        void setup(holoscan::OperatorSpec& spec) override { 
            spec.input<std::shared_ptr<Tensor<D, T>>>("in"); 
        }

        void start() {
            buffer.resize(5, std::vector<U64>{1, 32000});
        }

        void compute(holoscan::InputContext& op_input, 
                     holoscan::OutputContext&, 
                     holoscan::ExecutionContext&) override {
            auto in_value = op_input.receive<std::shared_ptr<Tensor<D, T>>>("in").value();

            buffer.put([&](auto& tensor){
                // TODO: Replace by proper copy.
                if constexpr (D == Device::CPU) {
                    for (U64 i = 0; i < in_value->size(); ++i) {
                        tensor.data()[i] = in_value->data()[i];
                    }
                }
            });
        };

     protected:
        TensorCircularBuffer<Tensor<D, T>> buffer;
    
        friend class HoloscanSource;
    };

 private:
    JST_DEFINE_IO();

    std::string holoscanModuleId;
    std::shared_ptr<CyberBridgeOp> op;
};

}  // namespace Jetstream



//
// CyberEther Blocks for CyberBridge (Holoscan)
//
// These blocks will handle the user interactions. They can be added to the flowgraph
// by dragging and dropping them from the block store. The user can then configure them
// to connect to a Holoscan node and receive data from one of its outputs.
//

namespace Jetstream::Blocks {

// TODO: Implement `HoloscanSink` block.

template<Device D, typename IT, typename OT>
class HoloscanSource : public Block {
 public:
    // Configuration

    struct Config {
        std::string nodeName;
        std::string nodeOutputName;

        JST_SERDES(nodeName, nodeOutputName);
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        JST_SERDES();
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        Tensor<D, IT> buffer;

        JST_SERDES(buffer);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Tensor<D, IT>& getOutputBuffer() const {
        return this->output.buffer;
    }

    // Housekeeping

    constexpr Device device() const {
        return D;
    }

    std::string id() const {
        return "holoscan-source";
    }

    std::string name() const {
        return "Holoscan Source";
    }

    std::string summary() const {
        return "Provides a source of data from Holoscan.";
    }

    std::string description() const {
        return "This block provides a source of data from Holoscan. "
               "It is used to connect to a Holoscan node and receive data from one of its outputs. "
               "First select the node and then the output you want to receive data from.\n"
               "\n"
               "## Suggestions\n"
               "  * To send data to Holoscan, use the *Holoscan Sink* block.\n"
               "\n"
               "## What is NVIDIA Holoscan?\n"
               "Holoscan SDK is an AI sensor processing platform that combines hardware systems for low-latency "
               "sensor and network connectivity, optimized libraries for data processing and AI, and core microservices "
               "to run streaming, imaging, and other applications, from embedded to edge to cloud. It can be used to build "
               "streaming AI pipelines for a variety of domains, including Medical Devices, High Performance Computing at the "
               "Edge, Industrial Inspection and more. For more information, visit [NVIDIA Holoscan](https://developer.nvidia.com/holoscan-sdk).";
    }

    // Constructor

    Result create() {
        JST_CHECK(instance().addModule(
            source, "source", {
                .nodeName = config.nodeName,
                .nodeOutputName = config.nodeOutputName
            }, {},
            locale()
        ));

        JST_CHECK(Block::LinkOutput("buffer", output.buffer, source->getOutputBuffer()));

        return Result::SUCCESS;
    }

    Result destroy() {
        JST_CHECK(instance().eraseModule(source->locale()));

        return Result::SUCCESS;
    }

    void drawControl() {
        if (!HoloscanBridge::IsAppRunning()) {
            ImGui::Text("Holoscan app is not running.");
            return;
        }

        auto& app = HoloscanBridge::GetApp();

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Node");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        static const char* noNodeMessage = "No nodes found";
        if (ImGui::BeginCombo("##NodeList", app.graph().is_empty() ? noNodeMessage : config.nodeName.c_str())) {
            for (const auto& node : app.graph().get_nodes()) {
                const bool isSelected = config.nodeName == node->name();
                if (ImGui::Selectable(node->name().c_str())) {
                    config.nodeName = node->name();
                }
                if (isSelected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        const auto& node = app.graph().find_node(config.nodeName);
        if (node) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("Node Output");
            ImGui::TableSetColumnIndex(1);
            ImGui::SetNextItemWidth(-1);
            static const char* noNodeOutputMessage = "No node output found";
            if (ImGui::BeginCombo("##NodeOutputList", node->spec()->outputs().empty() ? noNodeOutputMessage : config.nodeOutputName.c_str())) {
                for (const auto& [name, spec] : node->spec()->outputs()) {
                    const bool isSelected = config.nodeOutputName == name;
                    if (ImGui::Selectable(name.c_str())) {
                        config.nodeOutputName = name;
                    }
                    if (isSelected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
        }

        if (node && node->spec()->outputs().contains(config.nodeOutputName)) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TableSetColumnIndex(1);
            const F32 fullWidth = ImGui::GetContentRegionAvail().x;
            if (ImGui::Button("Connect Holoscan Tap", ImVec2(fullWidth, 0))) {
                JST_DISPATCH_ASYNC([&](){
                    ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Connecting..." });
                    JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
                });
            }
        }
    }

    constexpr bool shouldDrawControl() const {
        return true;
    }

 private:
    JST_DEFINE_IO();

    std::shared_ptr<Jetstream::HoloscanSource<D, IT>> source;
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(HoloscanSource, (std::is_same<IT,  F32>::value ||
                                  std::is_same<IT, CF32>::value) &&
                                 std::is_same<OT, void>::value)



//
// Basic CyberEther UI
//

class UI {
 public:
    UI(Instance& instance) : instance(instance) {
        JST_CHECK_THROW(create());
    }

    ~UI() {
        destroy();
    }

    Result create() {
        running = true;

        computeWorker = std::thread([&]{
            while(running && instance.viewport().keepRunning()) {
                JST_CHECK_THROW(instance.compute());
            }
        });

        graphicalWorker = std::thread([&]{
            while (running && instance.viewport().keepRunning()) {
                if (instance.begin() == Result::SKIP) {
                    continue;
                }

                [&]{
                    if (!HoloscanBridge::IsAppRunning()) {
                        ImGui::Text("Holoscan app is not running.");
                        return;
                    }

                    auto& app = HoloscanBridge::GetApp();
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
                }();

                JST_CHECK_THROW(instance.present());
                if (instance.end() == Result::SKIP) {
                    continue;
                }
            }
        });

        return Result::SUCCESS;
    }

    Result destroy() {
        running = false;
        computeWorker.join();
        graphicalWorker.join();
        instance.destroy();
        Backend::DestroyAll();

        JST_DEBUG("The UI was destructed.");

        return Result::SUCCESS;
    }

 private:
    std::thread graphicalWorker;
    std::thread computeWorker;
    Instance& instance;
    bool running = false;
};



//
// Custom Holoscan Ops
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

}  // namespace holoscan::ops



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
    Juggler<TensorType> juggler;
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

        auto config_path = std::filesystem::path("../ping-holoscan.yaml");
        app->config(config_path);

        return app;
    }
};



//
// SoapySDR FM + Neural Demodulation (SoapySDR + CyberEther + Holoscan Interop)
//
// This is a more complex example that demonstrates how to receive data from a SDR using SoapySDR
// wrapped in a Holoscan operator, and then display the data in CyberEther. The same mechanism with
// CyberBridge from the Fancy Ping example is used here too. This is literally demodulating FM radio 
// complex-valued IQ samples using a neural network (CursedNet). This example demonstrates how to use 
// TensorRT to accelerate the inference of a neural network inside a Holoscan operator while receiving 
// data from a SDR using SoapySDR. As usual, the demolated audio is sent to CyberEther for visualization 
// and playback.
//

namespace holoscan::ops {

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
    Juggler<TensorType> juggler;

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
    Juggler<OutputTensorType> juggler;

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
        hol_buffer = TensorToHoloscan(buffer);
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

        // Looks like this returns GPU mapped memory as a CPU pointer. This is mostly fine, 
        // but if you try to read it, it will crash. Weird?
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

        auto config_path = std::filesystem::path("../soapy-holoscan.yaml");
        app->config(config_path);

        return app;
    }
};



//
// CyberEther and Holoscan Instantiation
//

int main(int argc, char** argv) {
    // Initialize Holoscan.

    if (argc < 2) {
        JST_ERROR("Missing Holoscan application.");
        return 1;
    }

    if (argv[1] == std::string("ping")) {
        JST_CHECK_THROW(Jetstream::HoloscanBridge::RegisterApp(PingApp::Factory));
    } else if (argv[1] == std::string("soapy")) {
        JST_CHECK_THROW(Jetstream::HoloscanBridge::RegisterApp(SoapyApp::Factory));
    } else {
        JST_ERROR("Invalid Holoscan application.");
        return 1;
    }

    Jetstream::HoloscanBridge::StartApp();

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

    Instance instance;

    Backend::Config backendConfig;
    Viewport::Config viewportConfig;
    Render::Window::Config renderConfig;

    backendConfig.headless = true;

    viewportConfig.endpoint = "0.0.0.0:5002";
    viewportConfig.codec = Render::VideoCodec::H264;

    JST_CHECK_THROW(instance.buildInterface(Device::Vulkan, backendConfig, viewportConfig, renderConfig));

    instance.compositor().showStore(true)
                         .showFlowgraph(true);

    Store::LoadBlocks([](Block::ConstructorManifest& constructorManifest, 
                         Block::MetadataManifest& metadataManifest) {
        JST_TRACE("[BLOCKS] Loading block manifest list.");

        JST_BLOCKS_MANIFEST(
            Jetstream::Blocks::HoloscanSource,
        )

        return Result::SUCCESS;
    });

    if (argc >= 3) {
        const std::string flowgraphPath = argv[2];
        JST_CHECK_THROW(instance.flowgraph().create(flowgraphPath));
    }

    {
        auto ui = UI(instance);

        while (instance.viewport().keepRunning()) {
            instance.viewport().pollEvents();
        }
    }

    return 0;
}

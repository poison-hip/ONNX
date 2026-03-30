#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include <cuda_runtime_api.h>
#include <NvInfer.h>

#define CHECK(status)                                                           \
    do                                                                          \
    {                                                                           \
        auto ret = (status);                                                    \
        if (ret != 0)                                                           \
        {                                                                       \
            std::cerr << "CUDA failure: " << ret << std::endl;                  \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

namespace
{

class Logger : public nvinfer1::ILogger
{
public:
    // Severity -> 日志级别
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kERROR)
        {
            std::cerr << "[TensorRT] " << msg << std::endl;
        }
    }
};

template <typename T>
struct InferDeleter
{
    void operator()(T* obj) const
    {
        delete obj;
    }
};

template <typename T>
using InferUniquePtr = std::unique_ptr<T, InferDeleter<T>>;

// nvinfer1::Dims  tensor的shape
int64_t volume(const nvinfer1::Dims& dims)
{
    int64_t v = 1;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] < 0)
        {
            return -1;
        }
        v *= dims.d[i];
    }
    return v;
}

} // namespace

using namespace nvinfer1;

constexpr char kInputName[] = "input";
constexpr char kOutputName[] = "output";
constexpr int kInputH = 224;
constexpr int kInputW = 224;
constexpr int kBatchSize = 1;

void doInference(IExecutionContext& context, float* input, float* output)
{
    const ICudaEngine& engine = context.getEngine();
    assert(engine.getNbIOTensors() == 2);

    Dims input_dims{};
    input_dims.nbDims = 4;
    input_dims.d[0] = kBatchSize;
    input_dims.d[1] = 3;
    input_dims.d[2] = kInputH;
    input_dims.d[3] = kInputW;
    bool ok = context.setInputShape(kInputName, input_dims);
    assert(ok);

    const Dims output_dims = context.getTensorShape(kOutputName);
    const int64_t input_count = volume(input_dims);
    const int64_t output_count = volume(output_dims);
    assert(input_count > 0 && output_count > 0);

    void* input_buffer = nullptr;
    void* output_buffer = nullptr;
    CHECK(cudaMalloc(&input_buffer, input_count * sizeof(float)));
    CHECK(cudaMalloc(&output_buffer, output_count * sizeof(float)));

    cudaStream_t stream{};
    CHECK(cudaStreamCreate(&stream));

    CHECK(cudaMemcpyAsync(
        input_buffer, input, input_count * sizeof(float), cudaMemcpyHostToDevice, stream));

    ok = context.setTensorAddress(kInputName, input_buffer);
    assert(ok);
    ok = context.setTensorAddress(kOutputName, output_buffer);
    assert(ok);

    ok = context.enqueueV3(stream);
    assert(ok);

    CHECK(cudaMemcpyAsync(
        output, output_buffer, output_count * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));

    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaFree(input_buffer));
    CHECK(cudaFree(output_buffer));
}

int main()
{
    std::ifstream file("model.engine", std::ios::binary);
    if (!file)
    {
        std::cerr << "Failed to open model.engine" << std::endl;
        return 1;
    }

    file.seekg(0, file.end);
    const std::streamsize size = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> engine_data(static_cast<size_t>(size));
    file.read(engine_data.data(), size);

    Logger logger;
    InferUniquePtr<IRuntime> runtime{createInferRuntime(logger)};
    assert(runtime != nullptr);

    InferUniquePtr<ICudaEngine> engine{
        runtime->deserializeCudaEngine(engine_data.data(), engine_data.size())};
    assert(engine != nullptr);

    InferUniquePtr<IExecutionContext> context{engine->createExecutionContext()};
    assert(context != nullptr);

    std::cout << "Engine IO tensors:" << std::endl;
    for (int i = 0; i < engine->getNbIOTensors(); ++i)
    {
        const char* name = engine->getIOTensorName(i);
        const auto mode = engine->getTensorIOMode(name);
        std::cout << "  " << name << " : "
                  << (mode == TensorIOMode::kINPUT ? "input" : "output") << std::endl;
    }

    std::vector<float> input(kBatchSize * 3 * kInputH * kInputW, 1.0f);
    std::vector<float> output(kBatchSize * 3 * kInputH * kInputW / 4);

    doInference(*context, input.data(), output.data());

    std::cout << "Inference finished, first output value: " << output[0] << std::endl;
    return 0;
}

// OnnxRuntimeResNet.cpp : This file contains the 'main' function. Program execution begins and ends there.

#include <onnxruntime_cxx_api.h>
#include <iostream>

#include "Helpers.cpp"

int main()
{
    Ort::Env env;
    Ort::RunOptions runOptions;
    Ort::Session session(nullptr);


    constexpr int64_t numChannels = 3;
    constexpr int64_t width = 224;
    constexpr int64_t height = 224;
    constexpr int64_t numClasses = 1000;
    constexpr int64_t numInputElements = numChannels * height * width;
    const std::string labelFile = "./assets./imagenet_classes.txt";
    auto modelPath = L"./assets/dynamic_resnet50v2.onnx";
    //load labels
    std::vector<std::string> labels = loadLabels(labelFile);

    if (labels.empty()) {
        std::cout << "Failed to load labels: " << labelFile << std::endl;
        return 1;
    }



    //读取图像,保证图像数量大于batchsize;
    const std::string imagePath = "./assets/";
    std::string _strPattern = imagePath + "*.jpg";  // test_images
    std::vector<cv::String> filesVec;
    cv::glob(_strPattern, filesVec);
    int Batchsize = 0;
    std::vector<std::vector<float>> ImageBatch;
    for (int i = 0; i < filesVec.size(); i++)
    {
        const std::vector<float> imageVec = loadImage(filesVec[i]);
        if (imageVec.empty()) {
            std::cout << "Failed to load image: " << filesVec[i] << std::endl;
            return 1;
        }
        if (imageVec.size() != numInputElements) {

            std::cout << "Invalid image format. Must be 224x224 RGB image." << std::endl;
            return 1;
        }
        Batchsize++;
        ImageBatch.push_back(imageVec);
    }

    // Use CUDA GPU
    Ort::SessionOptions ort_session_options;

    OrtCUDAProviderOptions options;
    options.device_id = 0;
    //options.arena_extend_strategy = 0;
    //options.gpu_mem_limit = 2 * 1024 * 1024 * 1024;
    //options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
    //options.do_copy_in_default_stream = 1;
    OrtSessionOptionsAppendExecutionProvider_CUDA(ort_session_options, options.device_id);
    //create session
    session = Ort::Session(env, modelPath, ort_session_options);

    // Use CPU
    //session = Ort::Session(env, modelPath, Ort::SessionOptions{ nullptr });


    // Define shape
    const std::array<int64_t, 4> inputShape = { Batchsize, numChannels, height, width };
    const std::array<int64_t, 2> outputShape = { Batchsize, numClasses };



    // Define array
    std::vector<float> input(Batchsize * numInputElements);
    std::vector<float> results(Batchsize * numClasses);

    // Define Tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    auto inputTensor = Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(), inputShape.data(), inputShape.size());
    auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, results.data(), results.size(), outputShape.data(), outputShape.size());

    // Copy image data to input array
    for (int i = 0; i < Batchsize; ++i) {
        std::copy(ImageBatch[i].begin(), ImageBatch[i].end(), input.begin() + i * numInputElements);
    }

    // Define names
    Ort::AllocatorWithDefaultOptions ort_alloc;
    Ort::AllocatedStringPtr inputName = session.GetInputNameAllocated(0, ort_alloc);
    Ort::AllocatedStringPtr outputName = session.GetOutputNameAllocated(0, ort_alloc);
    const std::array<const char*, 1> inputNames = { inputName.get() };
    const std::array<const char*, 1> outputNames = { outputName.get() };
    inputName.release();
    outputName.release();

    // Run inference
    try {
        session.Run(runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);
    }
    catch (Ort::Exception& e) {
        std::cout << e.what() << std::endl;
        return 1;
    }

    // Sort results and show Top5 for each image in the batch
    for (int b = 0; b < Batchsize; b++) {
        std::vector<std::pair<size_t, float>> indexValuePairs;
        for (size_t i = 0; i < numClasses; ++i) {
            indexValuePairs.emplace_back(i, results[b * numClasses + i]);
        }
        std::sort(indexValuePairs.begin(), indexValuePairs.end(), [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });

        std::cout << "Image " << b + 1 << ":" << std::endl;
        // Show Top5
        for (size_t i = 0; i < 5; ++i) {
            const auto& result = indexValuePairs[i];
            std::cout << i + 1 << ": " << labels[result.first] << " " << result.second << std::endl;
        }
    }
    system("pause");
    return 0;
}
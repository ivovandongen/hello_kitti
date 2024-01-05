
#include <onnxruntime_cxx_api.h>
#include <coreml_provider_factory.h>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <sstream>

struct Detection {
    int clazz{};
    float confidence{};
    cv::Rect_<float> bbox;
    cv::Mat mask;
    std::vector<float> keypoints{};
};

// pretty prints a shape dimension vector
std::string print_shape(const std::vector<std::int64_t> &v) {
    std::stringstream ss("");
    for (std::size_t i = 0; i < v.size() - 1; i++) ss << v[i] << "x";
    ss << v[v.size() - 1];
    return ss.str();
}

int main() {

    // onnxruntime setup
    Ort::Env env(ORT_LOGGING_LEVEL_VERBOSE, "hello-yolo");
    Ort::SessionOptions session_options;
    uint32_t coreml_flags = 0;
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(session_options, coreml_flags));
    Ort::Session session = Ort::Session(env, "/Users/ivo/git/ivovandongen/hello_kitti/yolo/yolov8n.onnx",
                                        session_options);
    std::cout << "Available providers\n";
    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    for (auto &provider: availableProviders) {
        std::cout << "\t" << provider << "\n";
    }

    // print name/shape of inputs
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<std::string> input_names;
    std::vector<std::int64_t> input_shapes;
    std::cout << "Input Node Name/Shape (" << input_names.size() << "):" << std::endl;
    for (std::size_t i = 0; i < session.GetInputCount(); i++) {
        input_names.emplace_back(session.GetInputNameAllocated(i, allocator).get());
        input_shapes = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "\t" << input_names.at(i) << " : " << print_shape(input_shapes) << std::endl;
    }
    // some models might have negative shape values to indicate dynamic shape, e.g., for variable batch size.
    for (auto &s: input_shapes) {
        if (s < 0) {
            s = 1;
        }
    }

    // print name/shape of outputs
    std::vector<std::string> output_names;
    std::cout << "Output Node Name/Shape (" << output_names.size() << "):" << std::endl;
    for (std::size_t i = 0; i < session.GetOutputCount(); i++) {
        output_names.emplace_back(session.GetOutputNameAllocated(i, allocator).get());
        auto output_shapes = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "\t" << output_names.at(i) << " : " << print_shape(output_shapes) << std::endl;
    }
}
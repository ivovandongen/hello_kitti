#pragma once

#include "ml_model.hpp"

#include <opencv2/opencv.hpp>

#include <vector>

namespace ivd::ml {

    // TODO: Find a better place for this
    template<class T>
    struct Size {
        union {
            T x, width;
        };
        union {
            T y, height;
        };
    };

    struct Detection {
        int classIndex{};
        std::string className{};
        float confidence{};
        // TODO: Maybe not use cv:: types in interface
        cv::Rect_<float> bbox;
    };

    class DetectMLModel : public MLModel {

    public:
        explicit DetectMLModel(std::filesystem::path model);

        ~DetectMLModel() override = default;

        std::vector<Detection> predict(cv::Mat &image);

        Size<int64_t> inputSize() const {
            return inputSize_;
        };
    private:
        Size<int64_t> inputSize_;
    };

}

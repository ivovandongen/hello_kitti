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
        cv::Mat mask{};
    };

    class DetectMLModel : public MLModel {

    public:
        struct PredictionOptions {
            float scoreThreshold{0.45};
            float iouThreshold{0.50};
        };

    public:
        explicit DetectMLModel(std::filesystem::path model);

        ~DetectMLModel() override = default;

        std::vector<Detection> predict(const cv::Mat& image, PredictionOptions options = {0.25, 0.45});

        Size<int64_t> inputSize() const {
            return inputSize_;
        };
    private:
        struct PreprocessedImage {
            cv::Mat blob;
            cv::Mat image;
            Size<double> scale;
            struct Padding {
                int top;
                int bottom;
                int left;
                int right;
            } padding;
            cv::Size originalSize;
        };
        PreprocessedImage preprocess(const cv::Mat& image) const;

        cv::Mat processMask(cv::Mat protos, const cv::Rect &box, const std::vector<float> &mask,
                            const PreprocessedImage &) const;
    private:
        Size<int64_t> inputSize_{};
    };

}

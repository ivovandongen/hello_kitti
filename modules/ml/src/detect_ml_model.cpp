#include <ml/detect_ml_model.hpp>
#include <ml/yolo/yolo_classes.hpp>

namespace ivd::ml {
    // TODO: Arguments
    const constexpr float modelScoreThreshold{0.45};
    const constexpr float modelNMSThreshold{0.50};

    DetectMLModel::DetectMLModel(std::filesystem::path model) : MLModel(model) {
        auto inputNode = std::find_if(inputNodes().begin(), inputNodes().end(), [](const auto &node) {
            // TODO: YOLO specific
            return node.name == "images";
        });
        // TODO: YOLO specific
        assert((*inputNode).dimensions.size() == 4);
        assert((*inputNode).dimensions[0] == 1); // 1 image
        assert((*inputNode).dimensions[1] == 3); // 3 channels
        inputSize_ = {(*inputNode).dimensions[2], (*inputNode).dimensions[3]}; // 640x640
    }

    std::vector<Detection> DetectMLModel::predict(cv::Mat &image) {
        // TODO:
        // - Square input image (pad) or letterbox
        // - Color space conversion?
        auto imageSize = image.size();
        auto inputShape = inputNodes()[0].dimensions;

        // Calculate scale factors
        auto scale_x = imageSize.width / (double) inputSize_.width;
        auto scale_y = imageSize.height / (double) inputSize_.height;

        cv::Mat blob = cv::dnn::blobFromImage(image, 1 / 255.0, cv::Size(inputSize_.width, inputSize_.height),
                                              cv::Scalar(0, 0, 0), true, false);

        auto inputTensor = Ort::Value::CreateTensor<float>(
                Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
                (float *) blob.data, blob.total(), inputShape.data(),
                inputShape.size());

        std::vector<const char *> inputNames{inputNodes()[0].name.c_str()};
        std::vector<const char *> outputNames{outputNodes()[0].name.c_str()};
        auto outputs = session_.Run(Ort::RunOptions{nullptr},
                                    inputNames.data(), &inputTensor, 1, outputNames.data(), outputNames.size());

        auto output0DataShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        cv::Mat output0 = cv::Mat(cv::Size((int) output0DataShape[2], (int) output0DataShape[1]), CV_32F,
                                  outputs[0].GetTensorMutableData<float>()).t();

        auto *data = (float *) output0.data;
        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        for (size_t i = 0; i < output0.rows; i++) {
            float *classes_scores = data + 4;

            cv::Mat scores(1, yolo::class_names.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double maxClassScore;

            cv::minMaxLoc(scores, nullptr, &maxClassScore, nullptr, &class_id);

            if (maxClassScore > modelScoreThreshold) {
                confidences.push_back(maxClassScore);
                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = int((x - 0.5 * w) * scale_x);
                int top = int((y - 0.5 * h) * scale_y);

                int width = int(w * scale_x);
                int height = int(h * scale_y);

                boxes.emplace_back(left, top, width, height);
            }
            data += output0.cols;
        }

        std::vector<int> nms_result;
        cv::dnn::NMSBoxes(boxes, confidences, modelScoreThreshold, modelNMSThreshold, nms_result);

        std::vector<Detection> detections{};
        for (int idx: nms_result) {
            Detection result;
            result.classIndex = class_ids[idx];
            result.confidence = confidences[idx];

            result.className = yolo::class_names[result.classIndex];
            result.bbox = boxes[idx];
            detections.push_back(result);
        }

        return detections;
    }

}
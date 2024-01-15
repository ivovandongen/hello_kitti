#include <ml/detect_ml_model.hpp>
#include <ml/yolo/yolo_classes.hpp>

namespace ivd::ml {

    DetectMLModel::DetectMLModel(std::filesystem::path model) : MLModel(std::move(model)) {
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

    std::vector<Detection> DetectMLModel::predict(const cv::Mat& image, PredictionOptions options) {
        auto inputShape = inputNodes()[0].dimensions;

        auto preprocessedImage = preprocess(image);

        auto inputTensor = Ort::Value::CreateTensor<float>(
                Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
                (float *) preprocessedImage.blob.data, preprocessedImage.blob.total(), inputShape.data(),
                inputShape.size());

        auto outputs = session_.Run(Ort::RunOptions{nullptr},
                                    inputNames_.data(), &inputTensor, inputNames().size(), outputNames_.data(),
                                    outputNames_.size());
        bool segmentation = outputs.size() > 1;

        auto output0DataShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        cv::Mat output0 = cv::Mat(cv::Size((int) output0DataShape[2], (int) output0DataShape[1]), CV_32F,
                                  outputs[0].GetTensorMutableData<float>()).t();

        auto *data = (float *) output0.data;
        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        std::vector<std::vector<float>> masks;

        const auto rowWidth = output0.cols;
        const auto boxWidth = 4;
        // Output tensor layout:
        // - output0: [x, y, h, w, class_1, …, class_80]
        // - output1: [x, y, h, w, ]

        for (size_t i = 0; i < output0.rows; i++) {
            float *classScores = data + 4;

            cv::Mat scores(1, yolo::class_names.size(), CV_32FC1, classScores);
            cv::Point classId;
            double maxClassScore;

            cv::minMaxLoc(scores, nullptr, &maxClassScore, nullptr, &classId);

            if (maxClassScore > options.scoreThreshold) {
                confidences.push_back(maxClassScore);
                classIds.push_back(classId.x);

                if (segmentation) {
                    masks.push_back(std::vector<float>(data + 4 + yolo::class_names.size(), data + rowWidth));
                }

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = int(preprocessedImage.scale.width * (x - 0.5 * w - preprocessedImage.padding.left));
                int top = int(preprocessedImage.scale.width * (y - 0.5 * h - preprocessedImage.padding.top));
                int width = int(w * preprocessedImage.scale.x);
                int height = int(h * preprocessedImage.scale.y);

                // TODO: top left, bottom right instead
                // TODO: Bounding boxes boundary clamp

                boxes.emplace_back(left, top, width, height);
            }
            data += rowWidth;
        }

        std::vector<int> nmsResult;
        cv::dnn::NMSBoxes(boxes, confidences, options.scoreThreshold, options.iouThreshold, nmsResult);

        std::vector<Detection> detections{};
        for (int idx: nmsResult) {
            Detection result;
            result.classIndex = classIds[idx];
            result.confidence = confidences[idx];

            result.className = yolo::class_names[result.classIndex];
            result.bbox = boxes[idx];
            detections.push_back(result);
        }

        if (segmentation) {
            auto output1DataShape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
            // Single image, skip first dimension (1)
            std::vector<int> maskDimensions{(int) output1DataShape[1], (int) output1DataShape[2],
                                            (int) output1DataShape[3]};
            auto protos = cv::Mat(maskDimensions, CV_32F, outputs[1].GetTensorMutableData<float>());
            for (size_t i = 0; i < nmsResult.size(); i++) {
                auto idx = nmsResult[i];
                detections[i].mask = processMask(protos, boxes[idx], masks[idx], preprocessedImage);
            }
        }

        return detections;
    }

    DetectMLModel::PreprocessedImage DetectMLModel::preprocess(const cv::Mat& inputImage) const {

        // Ensure size matches, letterbox if needed
        auto imageSize = inputImage.size();
        cv::Size newSize(inputSize_.width, inputSize_.height);
        auto r = std::min(newSize.width / double(imageSize.width), newSize.height / double(imageSize.width));

        cv::Mat image;
        cv::Size sizeUnpadded{int(round(imageSize.width * r)), int(round(imageSize.height * r))};
        cv::Size_<double> padding = {(newSize.width - sizeUnpadded.width) / 2.0, (newSize.height - sizeUnpadded.height) / 2.0};
        if (imageSize != sizeUnpadded) {
            cv::resize(inputImage, image, sizeUnpadded);
        }

        int top = int(round(padding.height - 0.1));
        int bottom = int(round(padding.height + 0.1));
        int left = int(round(padding.width - 0.1));
        int right = int(round(padding.width + 0.1));

        cv::copyMakeBorder(image, image, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

        return {
                cv::dnn::blobFromImage(image, 1 / 255.0, cv::Size(inputSize_.width, inputSize_.height),
                                       cv::Scalar(0, 0, 0), true, false),
                image, // TODO: Remove after debugging
                {1 / r, 1 / r},
                {top, bottom, left, right},
                inputImage.size()
        };
    }

    cv::Mat DetectMLModel::processMask(cv::Mat protos, const cv::Rect &box, const std::vector<float> &maskIn,
                                       const DetectMLModel::PreprocessedImage &preprocessedImage) const {
        auto c = protos.size[0];
        auto mh = protos.size[1];
        auto mw = protos.size[2];
        // Matrix multiplication of mask 1x32 * protos reshaped to 32x25600(160x160)
        // TODO: take in all masks and do 1 multiplication instead?
        cv::Mat mask = cv::Mat(1, c, CV_32F, (void *) maskIn.data()) *
                       cv::Mat(std::vector<int>{c, mw * mh}, protos.type(), protos.ptr<float>(0));
        // Reshape to 160x160
        mask = cv::Mat(mw, mh, mask.type(), mask.ptr<float>(0));

        // tl br of mask
        auto scaleW = mh / double(inputSize_.width);
        auto scaleH = mh / double(inputSize_.height);
        cv::Rect roi(
                round(preprocessedImage.padding.left * scaleW - 0.1),
                round(preprocessedImage.padding.top * scaleH - 0.1),
                round(mw - (preprocessedImage.padding.left + preprocessedImage.padding.right) * scaleW + 0.1),
                round(mh - (preprocessedImage.padding.top + preprocessedImage.padding.bottom) * scaleH + 0.1)
        );
        mask = mask(roi);

        // Scale mask
        cv::resize(mask, mask, preprocessedImage.originalSize);

        // Filter by box and threshold
        return mask(box) > 0.5; // TODO: Param?
    }

}
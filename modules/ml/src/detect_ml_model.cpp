#include <ml/detect_ml_model.hpp>
#include <ml/yolo/yolo_classes.hpp>

namespace {
    cv::Mat getMask(const cv::Mat &mask_info, const cv::Mat &mask_data, const cv::Mat &inputImage, cv::Rect bound) {
        int seg_ch = 32;
        int seg_w = 160, seg_h = 160;
        int net_w = 640, net_h = 640;
        float mask_thresh = 0.5;

        cv::Mat mast_out;

        cv::Vec4f trans = {640.0f / inputImage.cols, 640.0f / inputImage.rows, 0, 0};
        int r_x = floor((bound.x * trans[0] + trans[2]) / net_w * seg_w);
        int r_y = floor((bound.y * trans[1] + trans[3]) / net_h * seg_h);
        int r_w = ceil(((bound.x + bound.width) * trans[0] + trans[2]) / net_w * seg_w) - r_x;
        int r_h = ceil(((bound.y + bound.height) * trans[1] + trans[3]) / net_h * seg_h) - r_y;
        r_w = MAX(r_w, 1);
        r_h = MAX(r_h, 1);
        if (r_x + r_w > seg_w) //crop
        {
            seg_w - r_x > 0 ? r_w = seg_w - r_x : r_x -= 1;
        }
        if (r_y + r_h > seg_h) {
            seg_h - r_y > 0 ? r_h = seg_h - r_y : r_y -= 1;
        }
        std::vector<cv::Range> roi_rangs = {cv::Range(0, 1), cv::Range::all(), cv::Range(r_y, r_h + r_y),
                                            cv::Range(r_x, r_w + r_x)};
        cv::Mat temp_mask = mask_data(roi_rangs).clone();
        cv::Mat protos = temp_mask.reshape(0, {seg_ch, r_w * r_h});
        cv::Mat matmul_res = (mask_info * protos).t();
        cv::Mat masks_feature = matmul_res.reshape(1, {r_h, r_w});
        cv::Mat dest;
        exp(-masks_feature, dest);//sigmoid
        dest = 1.0 / (1.0 + dest);
        int left = floor((net_w / seg_w * r_x - trans[2]) / trans[0]);
        int top = floor((net_h / seg_h * r_y - trans[3]) / trans[1]);
        int width = ceil(net_w / seg_w * r_w / trans[0]);
        int height = ceil(net_h / seg_h * r_h / trans[1]);
        cv::Mat mask;
        cv::resize(dest, mask, cv::Size(width, height));
        return mask(bound - cv::Point(left, top)) > mask_thresh;
    }
}

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

    std::vector<Detection> DetectMLModel::predict(cv::Mat &image, PredictionOptions options) {
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

                int left = int((x - 0.5 * w) * scale_x);
                int top = int((y - 0.5 * h) * scale_y);

                int width = int(w * scale_x);
                int height = int(h * scale_y);

                //  TODO Bounding boxes boundary clamp

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
            std::vector<int> maskDimensions{1, (int) output1DataShape[1], (int) output1DataShape[2],
                                     (int) output1DataShape[3]};
            cv::Mat output1 = cv::Mat(maskDimensions, CV_32F, outputs[1].GetTensorMutableData<float>());

            for (size_t i = 0; i < nmsResult.size(); i++) {
                auto idx = nmsResult[i];
                auto &result = detections[i];
                result.mask = getMask(cv::Mat(masks[idx]).t(), output1, image, result.bbox);
            }
        }

        return detections;
    }

}
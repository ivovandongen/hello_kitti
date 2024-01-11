#include <test.hpp>

#include <ml/detect_ml_model.hpp>

using namespace ivd::test;

TEST(DetectMLModel, loadYolo) {
    ivd::ml::DetectMLModel model{getModelsPath() / "yolo" / "yolov8n.onnx"};
    ASSERT_EQ(model.inputSize().width, 640);
    ASSERT_EQ(model.inputSize().height, 640);
    ASSERT_EQ(model.outputNodes().size(), 1);
}

TEST(DetectMLModel, predict) {
    std::filesystem::path testDir{getFixturesPath() / "ml" / "yolov8" / "image01"};
    ivd::ml::DetectMLModel model{getModelsPath() / "yolo" / "yolov8n.onnx"};

    cv::Mat image = cv::imread(testDir / "image.png");

    auto detections = model.predict(image);
    ASSERT_FALSE(detections.empty());
    for (auto &detection: detections) {
        cv::Rect box = detection.bbox;
        cv::Scalar color = cv::Scalar(0, 0, 255);

        // Detection box
        cv::rectangle(image, box, color, 2);

        // Detection box text
        const float fontScale = 0.5;
        const int thickness = 1;
        const int verticalPadding = 10;
        std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
        cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, fontScale, thickness, nullptr);
        cv::Rect textBox(box.x, box.y - textSize.height * 2, textSize.width + verticalPadding, textSize.height * 2);

        cv::rectangle(image, textBox, color, cv::FILLED);
        cv::putText(image, classString, cv::Point(box.x + verticalPadding / 2, box.y - textSize.height / 2),
                    cv::FONT_HERSHEY_DUPLEX, fontScale,
                    cv::Scalar(0, 0, 0), thickness, cv::LINE_AA);
    }

    cv::imwrite(testDir / "output.png", image);

    const float threshold = 0.001;
    auto expected = cv::imread(testDir / "expected.png");
    auto diff = image.clone();
    auto mismatched = pixelmatch(image.data, image.channels() * image.size().width, expected.data,
                                 expected.channels() * expected.size().width, image.size().width,
                                 image.size().height, diff.data, threshold);
    cv::imwrite(testDir / "diff.png", diff);
    EXPECT_LT(mismatched, image.total() * threshold);
}

TEST(DetectMLModel, loadYoloSeg) {
    ivd::ml::DetectMLModel model{getModelsPath() / "yolo" / "yolov8n-seg.onnx"};
    ASSERT_EQ(model.inputSize().width, 640);
    ASSERT_EQ(model.inputSize().height, 640);
    ASSERT_EQ(model.outputNodes().size(), 2);
}

TEST(DetectMLModel, segment) {
    std::filesystem::path testDir{getFixturesPath() / "ml" / "yolov8-seg" / "image01"};
    ivd::ml::DetectMLModel model{getModelsPath() / "yolo" / "yolov8n-seg.onnx"};

    cv::Mat image = cv::imread(testDir / "image.png");
    cv::Mat mask = image.clone();

    auto detections = model.predict(image);
    ASSERT_FALSE(detections.empty());
    for (auto &detection: detections) {
        ASSERT_TRUE(detection.mask.size().width > 0);
        ASSERT_TRUE(detection.mask.size().height > 0);
        cv::Scalar color = cv::Scalar(0, 0, 255);
        mask(detection.bbox).setTo(color, detection.mask);
    }

    cv::addWeighted(image, 0.6, mask, 0.4, 0, image);
    cv::imwrite(testDir / "output.png", image);

    const float threshold = 0.001;
    auto expected = cv::imread(testDir / "expected.png");
    auto diff = image.clone();
    auto mismatched = pixelmatch(image.data, image.channels() * image.size().width, expected.data,
                                 expected.channels() * expected.size().width, image.size().width,
                                 image.size().height, diff.data, threshold);
    cv::imwrite(testDir / "diff.png", diff);
    EXPECT_LT(mismatched, image.total() * threshold);
}

TEST(DetectMLModel, predictAndSegment) {
    std::filesystem::path testDir{getFixturesPath() / "ml" / "yolov8-predict-seg" / "image01"};
    ivd::ml::DetectMLModel model{getModelsPath() / "yolo" / "yolov8n-seg.onnx"};

    cv::Mat image = cv::imread(testDir / "image.png");
    cv::Mat mask = image.clone();

    auto detections = model.predict(image);
    ASSERT_FALSE(detections.empty());
    for (auto &detection: detections) {
        ASSERT_TRUE(detection.mask.size().width > 0);
        ASSERT_TRUE(detection.mask.size().height > 0);
        cv::Scalar color = cv::Scalar(0, 0, 255);
        mask(detection.bbox).setTo(color, detection.mask);
    }
    cv::addWeighted(image, 0.6, mask, 0.4, 0, image);

    ASSERT_FALSE(detections.empty());
    for (auto &detection: detections) {
        ASSERT_TRUE(detection.mask.size().width > 0);
        ASSERT_TRUE(detection.mask.size().height > 0);
        cv::Scalar color = cv::Scalar(0, 0, 255);
        cv::Rect box = detection.bbox;

        // Detection box
        cv::rectangle(image, box, color, 2);

        // Detection box text
        const float fontScale = 0.5;
        const int thickness = 1;
        const int verticalPadding = 10;
        std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
        cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, fontScale, thickness, nullptr);
        cv::Rect textBox(box.x, box.y - textSize.height * 2, textSize.width + verticalPadding, textSize.height * 2);

        cv::rectangle(image, textBox, color, cv::FILLED);
        cv::putText(image, classString, cv::Point(box.x + verticalPadding / 2, box.y - textSize.height / 2),
                    cv::FONT_HERSHEY_DUPLEX, fontScale,
                    cv::Scalar(0, 0, 0), thickness, cv::LINE_AA);
    }

    cv::imwrite(testDir / "output.png", image);

    const float threshold = 0.001;
    auto expected = cv::imread(testDir / "expected.png");
    auto diff = image.clone();
    auto mismatched = pixelmatch(image.data, image.channels() * image.size().width, expected.data,
                                 expected.channels() * expected.size().width, image.size().width,
                                 image.size().height, diff.data, threshold);
    cv::imwrite(testDir / "diff.png", diff);
    EXPECT_LT(mismatched, image.total() * threshold);
}
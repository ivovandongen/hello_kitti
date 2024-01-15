#include <test.hpp>

#include <common/string.hpp>
#include <ml/detect_ml_model.hpp>

using namespace ivd::test;

void overlaySegmentationMasks(cv::Mat &image, const std::vector<ivd::ml::Detection> &detections) {
    auto mask = image.clone();
    for (auto &detection: detections) {
        ASSERT_TRUE(detection.mask.size().width > 0);
        ASSERT_TRUE(detection.mask.size().height > 0);
        cv::Scalar color = cv::Scalar(0, 0, 255);
        mask(detection.bbox).setTo(color, detection.mask);
    }

    cv::addWeighted(image, 0.6, mask, 0.4, 0, image);
}

void overlayDetections(cv::Mat &image, const std::vector<ivd::ml::Detection> &detections) {
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
}

struct TestParam {
    std::filesystem::path model;
    std::filesystem::path input;
    std::filesystem::path expected;
    std::filesystem::path output;
    std::filesystem::path diff;
};

std::vector<TestParam>
testParameters(const std::filesystem::path &model, const std::filesystem::path &testBaseDir) {
    std::vector<TestParam> testParams;

    for (auto &subDir: std::filesystem::directory_iterator(testBaseDir)) {
        if (!subDir.is_directory() || !ivd::common::startsWith(subDir.path().filename().string(), "image")) {
            continue;
        }
        auto &testDir = subDir.path();
        testParams.push_back({model, testDir / "image.png", testDir / "expected.png", testDir / "output.png",
                              testDir / "diff.png"});
    }
    return testParams;
}

static std::string testName(const testing::TestParamInfo<TestParam> &param) {
    return std::to_string(param.index) + "_" + param.param.input.parent_path().filename().string();
}

class YoloDetect : public testing::TestWithParam<TestParam> {
};

class YoloSegment : public testing::TestWithParam<TestParam> {
};

class YoloPredictSegment : public testing::TestWithParam<TestParam> {
};

// Basic tests

TEST(DetectMLModel, loadYolo) {
    ivd::ml::DetectMLModel model{getModelsPath() / "yolo" / "yolov8n.onnx"};
    ASSERT_EQ(model.inputSize().width, 640);
    ASSERT_EQ(model.inputSize().height, 640);
    ASSERT_EQ(model.outputNodes().size(), 1);
}

TEST(DetectMLModel, loadYoloSeg) {
    ivd::ml::DetectMLModel model{getModelsPath() / "yolo" / "yolov8n-seg.onnx"};
    ASSERT_EQ(model.inputSize().width, 640);
    ASSERT_EQ(model.inputSize().height, 640);
    ASSERT_EQ(model.outputNodes().size(), 2);
}

// Parameterized tests

TEST_P(YoloDetect, Predict) {
    auto &params = GetParam();
    std::cout << "Testing: " << params.input.parent_path() << std::endl;
    ASSERT_TRUE(exists(params.model));
    ASSERT_TRUE(exists(params.input));
    ASSERT_TRUE(exists(params.expected));

    ivd::ml::DetectMLModel model{params.model};
    cv::Mat image = cv::imread(params.input);

    auto detections = model.predict(image);
    ASSERT_FALSE(detections.empty());

    overlayDetections(image, detections);
    cv::imwrite(params.output, image);

    const float threshold = 0.001;
    auto expected = cv::imread(params.expected);
    auto diff = image.clone();
    auto mismatched = pixelmatch(image.data, image.channels() * image.size().width, expected.data,
                                 expected.channels() * expected.size().width, image.size().width,
                                 image.size().height, diff.data, threshold);
    cv::imwrite(params.diff, diff);
    EXPECT_LT(mismatched, image.total() * threshold);
}

TEST_P(YoloSegment, Segment) {
    auto &params = GetParam();
    std::cout << "Testing: " << params.input.parent_path() << std::endl;
    ASSERT_TRUE(exists(params.model));
    ASSERT_TRUE(exists(params.input));
    ASSERT_TRUE(exists(params.expected));

    ivd::ml::DetectMLModel model{params.model};
    cv::Mat image = cv::imread(params.input);

    auto detections = model.predict(image);
    ASSERT_FALSE(detections.empty());

    overlaySegmentationMasks(image, detections);
    cv::imwrite(params.output, image);

    const float threshold = 0.001;
    auto expected = cv::imread(params.expected);
    auto diff = image.clone();
    auto mismatched = pixelmatch(image.data, image.channels() * image.size().width, expected.data,
                                 expected.channels() * expected.size().width, image.size().width,
                                 image.size().height, diff.data, threshold);
    cv::imwrite(params.diff, diff);
    EXPECT_LT(mismatched, image.total() * threshold);
}

TEST_P(YoloPredictSegment, PredictAndSegment) {
    auto &params = GetParam();
    std::cout << "Testing: " << params.input.parent_path() << std::endl;
    ASSERT_TRUE(exists(params.model));
    ASSERT_TRUE(exists(params.input));
    ASSERT_TRUE(exists(params.expected));

    ivd::ml::DetectMLModel model{params.model};
    cv::Mat image = cv::imread(params.input);
    cv::Mat mask = image.clone();

    auto detections = model.predict(image);
    ASSERT_FALSE(detections.empty());

    overlaySegmentationMasks(image, detections);
    overlayDetections(image, detections);
    cv::imwrite(params.output, image);

    const float threshold = 0.001;
    auto expected = cv::imread(params.expected);
    auto diff = image.clone();
    auto mismatched = pixelmatch(image.data, image.channels() * image.size().width, expected.data,
                                 expected.channels() * expected.size().width, image.size().width,
                                 image.size().height, diff.data, threshold);
    cv::imwrite(params.diff, diff);
    EXPECT_LT(mismatched, image.total() * threshold);
}

INSTANTIATE_TEST_SUITE_P(
        YoloV8,
        YoloDetect,
        testing::ValuesIn(testParameters(getModelsPath() / "yolo" / "yolov8n.onnx",
                                         getFixturesPath() / "ml" / "yolov8")),
        testName
);

INSTANTIATE_TEST_SUITE_P(
        YoloV8,
        YoloSegment,
        testing::ValuesIn(testParameters(getModelsPath() / "yolo" / "yolov8n-seg.onnx",
                                         getFixturesPath() / "ml" / "yolov8-seg")),
        testName
);

INSTANTIATE_TEST_SUITE_P(
        YoloV8,
        YoloPredictSegment,
        testing::ValuesIn(testParameters(getModelsPath() / "yolo" / "yolov8n-seg.onnx",
                                         getFixturesPath() / "ml" / "yolov8-predict-seg")),
        testName
);

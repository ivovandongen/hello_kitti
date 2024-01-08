#include <test.hpp>

#include <ml/detect_ml_model.hpp>
#include <mapbox/pixelmatch.hpp>

TEST(DetectMLModel, load) {
    ivd::ml::DetectMLModel model{getModelsPath() / "yolo" / "yolov8n.onnx"};
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
    auto mismatched = mapbox::pixelmatch(image.data, image.channels() * image.size().width, expected.data,
                                         expected.channels() * expected.size().width, image.size().width,
                                         image.size().height, diff.data, threshold);
    cv::imwrite(testDir / "diff.png", diff);
    EXPECT_LT(mismatched, image.total() * threshold);
}
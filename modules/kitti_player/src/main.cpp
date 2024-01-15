
#include <kitti_parser/Parser.h>
#include <cxxopts.hpp>
#include <ml/detect_ml_model.hpp>

#include <iostream>
#include <filesystem>
#include <regex>

void annotateImage(cv::Mat &image, std::vector<ivd::ml::Detection> &detections, bool mask) {

    // Draw mask
    if (mask) {
        // TODO: Fancy outline
        cv::Scalar maskColor = cv::Scalar(0, 0, 255);

        auto masked = image.clone();
        for (auto &detection: detections) {
            if (!detection.mask.empty()) {
                masked(detection.bbox).setTo(maskColor, detection.mask);
            }
        }

        cv::addWeighted(image, 0.6, masked, 0.4, 0, image);
    }

    for (auto &detection: detections) {
        cv::Rect box = detection.bbox;

        // Detection box
        cv::Scalar color = cv::Scalar(0, 0, 255);
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

struct Options {
    std::filesystem::path model;
    std::filesystem::path data;
    std::string index;
    uint32_t wait;
    double speed;
    bool mask;
};

Options parseOpts(int argc, char **argv) {
    cxxopts::Options options(argv[0], argv[0]);

    // clang-format off
    options.add_options()
            ("m, model", "Model file", cxxopts::value<std::string>())
            ("d,data", "Data directory", cxxopts::value<std::string>())
            ("index", "Dataset index to play from data directory", cxxopts::value<std::string>()->default_value(""))
            ("w,wait", "Wait time (ms) between frames - 0 == wait indefinitely",
             cxxopts::value<uint32_t>()->default_value("1"))
            ("s,speed", "Speed multiplier", cxxopts::value<double>()->default_value("1.0"))
            ("mask", "Segmentation mask", cxxopts::value<bool>()->default_value("false"))
            ("h,help", "Print usage");
    // clang-format on

    try {
        auto result = options.parse(argc, argv);
        if (result.count("help")) {
            std::cout << options.help().c_str() << std::endl;
            exit(0);
        }

        Options opts{
                result["model"].as<std::string>(),
                result["data"].as<std::string>(),
                result["index"].as<std::string>(),
                result["wait"].as<uint32_t>(),
                result["speed"].as<double>(),
                result["mask"].as<bool>(),
        };

        return opts;
    } catch (const cxxopts::exceptions::exception &e) {
        std::cerr << "Invalid options: " << e.what() << std::endl;
        std::cout << options.help().c_str() << std::endl;
        exit(1);
    } catch (...) {
        std::cout << options.help().c_str() << std::endl;
        exit(1);
    }
}

int main(int argc, char **argv) {
    const constexpr char *leftWindowColor = "left image (Color)";
    const constexpr char *rightWindowColor = "right image (Color)";
    const constexpr float windowScale = 0.7;
    cv::namedWindow(leftWindowColor, cv::WINDOW_NORMAL);
    cv::namedWindow(rightWindowColor, cv::WINDOW_NORMAL);

    std::cout << "Hello KITTI" << std::endl;
    auto options = parseOpts(argc, argv);

    kitti_parser::Parser parser(options.data, [&](auto& indexPath) {
        return options.index.empty() ||
               std::regex_match(indexPath.filename().string(), std::regex(options.data.filename().string() +
                                                                          "_drive_(" + options.index + ")_sync"));
    });
    ivd::ml::DetectMLModel model(options.model);

    parser.register_callback_stereo_color([&](kitti_parser::Config *config, long ts, kitti_parser::stereo_t *frame) {
        std::cout << "Ts: " << ts << "\n\tImage left: " << frame->image_left_path << "\n\tImage Right: "
                  << frame->image_right_path << std::endl;
        auto detections = model.predict(frame->image_left);
        std::cout << "\tDetections (" << detections.size() << "):" << std::endl;
        for (auto &detection: detections) {
            std::cout << "\t\t" << detection.className << ": " << detection.confidence << std::endl;
        }
        annotateImage(frame->image_left, detections, options.mask);
        cv::imshow(leftWindowColor, frame->image_left);
        cv::resizeWindow(leftWindowColor, frame->image_left.size().width * windowScale,
                         frame->image_left.size().height * windowScale);
        cv::imshow(rightWindowColor, frame->image_right);
        cv::resizeWindow(rightWindowColor, frame->image_right.size().width * windowScale,
                         frame->image_right.size().height * windowScale);
        cv::moveWindow(rightWindowColor, frame->image_right.size().width * windowScale, 0);
        cv::waitKey(options.wait);
    });

    parser.run(options.speed);
}
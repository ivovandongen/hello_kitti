#include <kitti/kitti_utils.hpp>
#include <common/opencv_utils.hpp>
#include <common/projection.hpp>
#include <ml/detect_ml_model.hpp>
#include <lidar/lidar.hpp>
#include <lidar/visualize.hpp>

#include <kitti_parser/Parser.h>
#include <cxxopts.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <iostream>
#include <filesystem>
#include <regex>

using namespace ivd;

void
annotateImage(cv::Mat &image, std::vector<ivd::ml::Detection> &detections, std::vector<double> distances,
              const cv::Mat &depthMap, bool mask) {

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

    for (size_t i = 0; i < detections.size(); i++) {
        auto &detection = detections[i];
        auto distance = distances[i];
        cv::Rect box = detection.bbox;

        // Detection box
        cv::Scalar color = cv::Scalar(0, 0, 255);
        cv::rectangle(image, box, color, 2);

        // Lidar points
        lidar::visualizeLidarDepthForBBox(image, depthMap, detection.bbox, detection.mask);

        // Detection box text
        const float fontScale = 0.5;
        const int thickness = 1;
        const int verticalPadding = 10;
        std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
        cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, fontScale, thickness, nullptr);
        cv::Rect textBox(box.x, box.y - textSize.height * 2, textSize.width + verticalPadding, textSize.height * 2);

        cv::rectangle(image, textBox, color, cv::FILLED);
        cv::putText(image, std::to_string(distance) + "m",
                    cv::Point(box.x + verticalPadding / 2, box.y - textSize.height / 2),
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

class Pipeline {
    const static constexpr char *leftWindowColor = "left image (Color)";
    const static constexpr char *rightWindowColor = "right image (Color)";
    const static constexpr char *lidarWindowColor = "Lidar";
    const static constexpr char *lidarDepthWindow = "Lidar depth raw";
    const static constexpr float windowScale = 1.0; //0.7;
public:
    explicit Pipeline(const Options &options, cv::Mat velo_cam2) : options_(options), model_(options.model),
                                                                   velo_cam2_(std::move(velo_cam2)) {
        cv::namedWindow(leftWindowColor, cv::WINDOW_NORMAL);
        cv::namedWindow(rightWindowColor, cv::WINDOW_NORMAL);
        cv::namedWindow(lidarWindowColor, cv::WINDOW_NORMAL);
        cv::namedWindow(lidarDepthWindow, cv::WINDOW_NORMAL);
    }

    ~Pipeline() {
        cv::destroyWindow(leftWindowColor);
        cv::destroyWindow(rightWindowColor);
    }

    void update(long ts, kitti_parser::stereo_t *frame) {
        cv::Mat leftGray, rightGray;
        cv::cvtColor(frame->image_left, leftGray, cv::ColorConversionCodes::COLOR_RGB2GRAY);
        cv::cvtColor(frame->image_right, rightGray, cv::ColorConversionCodes::COLOR_RGB2GRAY);

        // cv::Mat is reference counted...
        leftColor_ = frame->image_left;
        rightColor_ = frame->image_right;

        // Get some detections
        // TODO: Use grayscale instead
        detections_ = model_.predict(frame->image_left);

        update();
    }

    void update(long ts, kitti_parser::lidar_t *frame) {
        // Initialize Mat with #points rows x 4 columns and strip the last column (no-copy)
        cv::Mat data(frame->points.size(), 4, CV_32F, frame->points.data());
        data = data.colRange(0, 3);

        lidarDataFrameXYZW_ = lidar::makeHomogeneous(data);

        update();
    }

private:
    void update() {
        // See if frame is complete
        if (leftColor_.empty() || rightColor_.empty() || lidarDataFrameXYZW_.empty()) {
            return;
        }

        veloUVZ_ = lidar::project(lidarDataFrameXYZW_, velo_cam2_, leftColor_.size());
        depthMap_ = lidar::depthMapFromProjectedPoints(veloUVZ_, leftColor_.size());

        auto distances = [](std::vector<ivd::ml::Detection> &detections, cv::Mat &depthMap, bool useMask) {
            std::vector<double> distances;
            distances.reserve(detections.size());
            std::transform(detections.begin(), detections.end(), std::back_inserter(distances),
                           [&](const ivd::ml::Detection &detection) {
                               if (useMask) {
                                   return ivd::lidar::getDepth(depthMap, detection.bbox, detection.mask).value_or(-1);
                               } else {
                                   return ivd::lidar::getDepth(depthMap, detection.bbox).value_or(-1);
                               }
                           });
            return distances;
        }(detections_, depthMap_, options_.mask);

        // Left color image
        annotateImage(leftColor_, detections_, distances, depthMap_, options_.mask);
        cv::imshow(leftWindowColor, leftColor_);
        cv::resizeWindow(leftWindowColor, leftColor_.size().width * windowScale,
                         leftColor_.size().height * windowScale);


        // Right color image
        cv::imshow(rightWindowColor, rightColor_);
        cv::resizeWindow(rightWindowColor, rightColor_.size().width * windowScale,
                         rightColor_.size().height * windowScale);
        cv::moveWindow(rightWindowColor, cv::getWindowImageRect(leftWindowColor).width * windowScale, 0);

        // Draw lidar points
        std::cout << "Lidar point count:" << veloUVZ_.cols << std::endl;
        cv::Mat lidarMap(leftColor_.size(), CV_8UC3, cv::Scalar(0));
        lidar::visualizeLidarPoints(lidarMap, veloUVZ_);
        cv::imshow(lidarWindowColor, lidarMap);
        cv::moveWindow(lidarWindowColor, 0, cv::getWindowImageRect(leftWindowColor).height * windowScale);

        // Draw lidar depth Map raw
        cv::imshow(lidarDepthWindow, depthMap_);
        cv::moveWindow(lidarDepthWindow, cv::getWindowImageRect(lidarWindowColor).width,
                       cv::getWindowImageRect(leftWindowColor).height * windowScale);

        // Wait or not
        cv::waitKey(options_.wait);

        // Clear frame data
        leftColor_ = cv::Mat();
        rightColor_ = cv::Mat();
        detections_.clear();
        lidarDataFrameXYZW_ = cv::Mat();
        veloUVZ_ = cv::Mat();
        depthMap_ = cv::Mat();
    }

private:
    Options options_;
    ivd::ml::DetectMLModel model_;

    // Projection matrices
    cv::Mat velo_cam2_;

    // Frame data
    cv::Mat leftColor_, rightColor_;
    std::vector<ml::Detection> detections_;
    cv::Mat lidarDataFrameXYZW_;
    cv::Mat veloUVZ_;
    cv::Mat depthMap_;
};

int main(int argc, char **argv) {
    using namespace ivd::common;
    using namespace ivd::kitti;

    std::cout << "Hello KITTI" << std::endl;
    auto options = parseOpts(argc, argv);

    kitti_parser::Parser parser(options.data, [&](auto &indexPath) {
        return options.index.empty() ||
               std::regex_match(indexPath.filename().string(), std::regex(options.data.filename().string() +
                                                                          "_drive_(" + options.index + ")_sync"));
    });

    auto camCalibration = parser.getConfig().calib_cc;
    auto pRect02 = ivd::kitti::parseMatrix(camCalibration["P_rect_02"], 3, 4);
    auto pRect03 = ivd::kitti::parseMatrix(camCalibration["P_rect_03"], 3, 4);
    auto leftDecomp = ivd::common::decomposeProjectionMatrix(pRect02);
    auto rightDecomp = ivd::common::decomposeProjectionMatrix(pRect03);


    // Calculate projection to euclidian space from IMU point of reference

    // Rigid body transformation from Lidar to camera 0 (ref)
    auto T_velo_ref0 = composeRigidTransformationMatrix(parseMatrix(parser.getConfig().calib_vc['R'], 3, 3),
                                                        parseMatrix(parser.getConfig().calib_vc['T'], 3, 1));

    // Rigid body transformation from IMU to Lidar
    auto T_imu_velo = composeRigidTransformationMatrix(parseMatrix(parser.getConfig().calib_iv['R'], 3, 3),
                                                       parseMatrix(parser.getConfig().calib_iv['T'], 3, 1));

    // Rigid body transformation (rotation) matrix for left camera -> rectified left camera: 4x4
    auto R_ref0_rect2 = parseMatrix(camCalibration["R_rect_02"], 3, 3);
    R_ref0_rect2 = composeRigidTransformationMatrix(R_ref0_rect2);

    // Rigid body transformation from Camera 0 (ref) to Camera 2 (left color): 4x4
    auto T_ref0_ref2 = composeRigidTransformationMatrix(parseMatrix(camCalibration["R_02"], 3, 3),
                                                        parseMatrix(camCalibration["T_02"], 3, 1));

    // transform from velo (LiDAR) to left color camera (shape 3x4)
    auto P_rect2_cam2 = ivd::kitti::parseMatrix(camCalibration["P_rect_02"], 3, 4);
    auto T_velo_cam2 = P_rect2_cam2 * R_ref0_rect2 * T_ref0_ref2 * T_velo_ref0;

    // transform from IMU to left color camera (shape 3x4)
    auto T_imu_cam2 = T_velo_cam2 * T_imu_velo;

    // homogeneous transform from left color camera to IMU (shape: 4x4)
    cv::Mat T_cam2_imu;
    cv::vconcat(T_imu_cam2, cv::Mat((cv::Mat_<double>(1, 4) << 0, 0, 0, 1)), T_cam2_imu);
    T_cam2_imu = T_cam2_imu.inv();
    assert(T_cam2_imu.size() == cv::Size(4, 4));

    common::print<double>((cv::Mat) T_velo_cam2);
    assert(T_velo_cam2.size() == cv::Size(4, 3));

    Pipeline pipeline(options, T_velo_cam2);

    parser.register_callback_stereo_color([&](kitti_parser::Config *config, long ts, kitti_parser::stereo_t *frame) {
        std::cout << "Ts: " << ts << "\n\tImage left: " << frame->image_left_path << "\n\tImage Right: "
                  << frame->image_right_path << std::endl;
        pipeline.update(ts, frame);
    });

    parser.register_callback_lidar([&](kitti_parser::Config *config, long ts, kitti_parser::lidar_t *frame) {
        std::cout << "Ts: " << ts << std::endl;
        pipeline.update(ts, frame);
    });

    parser.run(options.speed);
}
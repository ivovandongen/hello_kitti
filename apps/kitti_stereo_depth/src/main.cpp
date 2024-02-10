#include <kitti/kitti_utils.hpp>
#include <common/opencv_utils.hpp>
#include <common/projection.hpp>
#include <stereo/disparity.hpp>
#include <stereo/depth_map.hpp>
#include <ml/detect_ml_model.hpp>

#include <kitti_parser/Parser.h>
#include <cxxopts.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <filesystem>
#include <regex>

void
annotateImage(cv::Mat &image, std::vector<ivd::ml::Detection> &detections, std::vector<double> distances, bool mask) {

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

cv::Point3d toEuclidian(const cv::Point3d &point, const cv::Mat &transformation) {
    cv::Mat homogeneous(4, 1, CV_64F);
    homogeneous.at<double>(0) = point.x * point.z;
    homogeneous.at<double>(1) = point.y * point.z;
    homogeneous.at<double>(2) = point.z;
    homogeneous.at<double>(3) = 1;
    cv::Mat xyzw = transformation * homogeneous;
    return {xyzw.at<double>(0), xyzw.at<double>(1), xyzw.at<double>(2)};
}

cv::Point3d toEuclidian(const cv::Rect &bbox, double distance, const cv::Mat &transformation) {
    cv::Mat homogeneous(4, 1, CV_64F);
    auto center = (bbox.br() - bbox.tl()) / 2.0 + bbox.tl();
    return toEuclidian(cv::Point3d(center.x, center.y, distance), transformation);
}

struct Options {
    std::filesystem::path model;
    std::filesystem::path data;
    std::string index;
    uint32_t wait;
    double speed;
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
    using namespace ivd::common;
    using namespace ivd::kitti;
    using namespace ivd::stereo;

    const constexpr char *leftWindowColor = "left image (Color)";
    const constexpr char *rightWindowColor = "right image (Color)";
    const constexpr char *disparityMapWindow = "Disparity map";
    const constexpr char *depthMapWindow = "Depth map";
    const constexpr char *eulerMapWindow = "World space";
    const constexpr float windowScale = 1.0; //0.7;
    cv::namedWindow(leftWindowColor, cv::WINDOW_NORMAL);
    cv::namedWindow(rightWindowColor, cv::WINDOW_NORMAL);
    cv::namedWindow(disparityMapWindow, cv::WINDOW_NORMAL);
    cv::namedWindow(depthMapWindow, cv::WINDOW_NORMAL);
    cv::namedWindow(eulerMapWindow, cv::WINDOW_NORMAL);

    std::cout << "Hello KITTI" << std::endl;
    auto options = parseOpts(argc, argv);

    kitti_parser::Parser parser(options.data, [&](auto &indexPath) {
        return options.index.empty() ||
               std::regex_match(indexPath.filename().string(), std::regex(options.data.filename().string() +
                                                                          "_drive_(" + options.index + ")_sync"));
    });

    ivd::ml::DetectMLModel model(options.model);

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

    parser.register_callback_stereo_color([&](kitti_parser::Config *config, long ts, kitti_parser::stereo_t *frame) {
        std::cout << "Ts: " << ts << "\n\tImage left: " << frame->image_left_path << "\n\tImage Right: "
                  << frame->image_right_path << std::endl;

        cv::Mat leftGray, rightGray;
        cv::cvtColor(frame->image_left, leftGray, cv::ColorConversionCodes::COLOR_RGB2GRAY);
        cv::cvtColor(frame->image_right, rightGray, cv::ColorConversionCodes::COLOR_RGB2GRAY);

        // Get some detections
        auto detections = model.predict(frame->image_left);

        // Calculate disparity map
        auto disparity = ivd::stereo::disparityMapSGBM(leftGray, rightGray);
        cv::medianBlur(disparity, disparity, 5);
        auto depthMap = ivd::stereo::calculateDepthMap(disparity, leftDecomp.cameraIntrinsic, leftDecomp.translation,
                                                       rightDecomp.translation);
        auto distances = [](std::vector<ivd::ml::Detection> &detections, cv::Mat &depthMap) {
            std::vector<double> distances;
            distances.reserve(detections.size());
            std::transform(detections.begin(), detections.end(), std::back_inserter(distances),
                           [&depthMap](const ivd::ml::Detection &detection) {
                               return ivd::stereo::getDepth(depthMap, detection.bbox);
                           });
            return distances;
        }(detections, depthMap);

        // Annotate image
        annotateImage(frame->image_left, detections, distances, false);

        // Left color image
        cv::imshow(leftWindowColor, frame->image_left);
        cv::resizeWindow(leftWindowColor, frame->image_left.size().width * windowScale,
                         frame->image_left.size().height * windowScale);

        // Right color image
        cv::imshow(rightWindowColor, frame->image_right);
        cv::resizeWindow(rightWindowColor, frame->image_right.size().width * windowScale,
                         frame->image_right.size().height * windowScale);
        cv::moveWindow(rightWindowColor, cv::getWindowImageRect(leftWindowColor).width * windowScale, 0);

        // Disparity map window

        // Normalize brightness to full range and convert to cividis
        cv::normalize(disparity, disparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        disparity.convertTo(disparity, CV_8U);
        cv::applyColorMap(disparity, disparity, cv::ColormapTypes::COLORMAP_CIVIDIS);

        cv::imshow(disparityMapWindow, disparity);
        cv::moveWindow(disparityMapWindow, 0, cv::getWindowImageRect(leftWindowColor).height);

        // Dept map window
        // Log and normalize for reasonable visualization
        cv::log(depthMap, depthMap);
        cv::normalize(depthMap, depthMap, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::applyColorMap(depthMap, depthMap, cv::ColormapTypes::COLORMAP_VIRIDIS);
        cv::imshow(depthMapWindow, depthMap);
        cv::moveWindow(depthMapWindow, cv::getWindowImageRect(leftWindowColor).width,
                       cv::getWindowImageRect(disparityMapWindow).height);


        // World space positions
        cv::Mat worldSpaceMap(500, 500, CV_8UC3, cv::Scalar(0));
        // Ego position
        cv::Point egoPosition(500 / 2, 500 / 4 * 3);
        cv::circle(worldSpaceMap, egoPosition, 5, cv::Scalar(255, 0, 0), 2, cv::FILLED);

        const constexpr int pixelMeterRatio = 5;
        for (size_t i = 0; i < detections.size(); i++) {
            auto &detection = detections[i];
            auto &distance = distances[i];
            auto point = toEuclidian(detection.bbox, distance, T_cam2_imu);
            cv::circle(worldSpaceMap, egoPosition - cv::Point(point.y * pixelMeterRatio, point.x * pixelMeterRatio), 5,
                       cv::Scalar(0, 255, 0), 2, cv::FILLED);
        }

        cv::imshow(eulerMapWindow, worldSpaceMap);
        cv::moveWindow(eulerMapWindow, 0, cv::getWindowImageRect(leftWindowColor).height +
                                          cv::getWindowImageRect(disparityMapWindow).height);

        // Wait or not
        cv::waitKey(options.wait);
    });

    parser.run(options.speed);
}
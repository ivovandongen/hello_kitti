#pragma once

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include <filesystem>

namespace ivd::kitti {
    cv::Mat parseMatrix(const YAML::Node &input, int rows, int cols);

    std::vector<std::array<float, 4>> readVeloBin(const std::filesystem::path &file);
}
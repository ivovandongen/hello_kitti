#pragma once

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

namespace ivd::kitti {
    cv::Mat parseMatrix(const YAML::Node &input, int rows, int cols);
}
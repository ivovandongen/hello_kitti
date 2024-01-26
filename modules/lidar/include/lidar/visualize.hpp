#pragma once

#include <opencv2/opencv.hpp>

namespace ivd::lidar {

    void visualizeLidarPoints(const cv::Mat &points, cv::Mat &image, const std::string &palette = "jet",
                              double maxDistance = 50);
}

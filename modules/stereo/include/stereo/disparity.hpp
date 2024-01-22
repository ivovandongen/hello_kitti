#pragma once

#include <opencv2/opencv.hpp>

namespace ivd::stereo {
    cv::Mat disparityMapSGBM(const cv::Mat &left, const cv::Mat &right);
}


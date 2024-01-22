#pragma once

#include <opencv2/opencv.hpp>

namespace ivd::stereo {
    cv::Mat
    calculateDepthMap(cv::Mat &disparityLeft, const cv::Mat &K_left, const cv::Mat &T_left, const cv::Mat &T_right);

    double getDepth(const cv::Mat& depthMap, cv::Rect box);
}
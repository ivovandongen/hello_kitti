#pragma once

#include <opencv2/opencv.hpp>

namespace ivd::lidar {

    cv::Mat makeHomogeneous(const cv::Mat &points);

    // TODO: Optimize by removing points that are not possibly in view (eg only positive X for forward looking camera)

    cv::Mat project(const cv::Mat &homogeneous, const cv::Mat &T, const cv::Size &size = {});

    cv::Mat depthMapFromProjectedPoints(const cv::Mat &points, cv::Size size);

    std::optional<double> getDepth(const cv::Mat &lidarPoints, const cv::Rect &bbox);
}
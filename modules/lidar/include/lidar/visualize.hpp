#pragma once

#include <opencv2/opencv.hpp>

namespace ivd::lidar {

    void visualizeLidarPoints(cv::Mat &image, const cv::Mat &points, const std::string &palette = "jet",
                              double maxDistance = 50);

    void visualizeLidarDepthForBBox(cv::Mat &image, const cv::Mat &depthMap, const cv::Rect &bbox,
                                    const cv::Mat &mask = cv::Mat(),
                                    const std::string &palette = "jet",
                                    double maxDistance = 50);
}

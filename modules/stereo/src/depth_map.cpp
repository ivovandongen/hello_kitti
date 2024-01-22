#include <stereo/depth_map.hpp>

#include <common/opencv_utils.hpp>

namespace ivd::stereo {

    cv::Mat
    calculateDepthMap(cv::Mat &disparityLeft, const cv::Mat &K_left, const cv::Mat &T_left, const cv::Mat &T_right) {
        assert(disparityLeft.channels() == 1);
        assert(disparityLeft.type() == CV_32F);

        // Get the focal length from the camera matrix
        auto f = K_left.at<double>(0, 0);

        // Get the distance between left and right cameras
        auto b = cv::Mat(cv::abs(T_left.row(0) - T_right.row(0))).at<double>(0);

        // Get rid of =< 0 values and replace with a small value (avoid divide by 0)
        cv::Mat mask(disparityLeft.size(), CV_8U);
        cv::compare(disparityLeft, 0, mask, cv::CmpTypes::CMP_LE);
        disparityLeft.setTo(1e-5, mask);

        // Calculate the depths
        auto depth_map = f * b / disparityLeft;
        return depth_map;
    }

    double getDepth(const cv::Mat &depthMap, cv::Rect bbox) {
        assert(depthMap.type() == CV_32F);
        assert(depthMap.channels() == 1);

        // Get depth slice for BBox
        cv::Mat depthSlice(depthMap, bbox);

        // Return median value of depth slice
        return ivd::common::median<float>(depthSlice);
    }

}
#include <test.hpp>
#include <stereo_test_utils.hpp>

#include <stereo/disparity.hpp>
#include <stereo/depth_map.hpp>

#include <opencv2/opencv.hpp>

#include <fstream>

using namespace ivd::test;

TEST(DepthMap, CalculateDepthE2E_SGBM) {
    auto baseDir = getFixturesPath() / "stereo" / "00_basic";
    auto left = cv::imread(baseDir / "left.png");
    auto right = cv::imread(baseDir / "right.png");
    cv::cvtColor(left, left, cv::ColorConversionCodes::COLOR_BGR2RGB);
    cv::cvtColor(left, left, cv::ColorConversionCodes::COLOR_RGB2GRAY);
    cv::cvtColor(right, right, cv::ColorConversionCodes::COLOR_BGR2RGB);
    cv::cvtColor(right, right, cv::ColorConversionCodes::COLOR_RGB2GRAY);
    auto disparity = ivd::stereo::disparityMapSGBM(left, right);
    ASSERT_FALSE(disparity.empty());

    auto K_left = readMat(baseDir / "K_left.txt");
    auto T_left = readMat(baseDir / "T_left.txt");
    auto T_right = readMat(baseDir / "T_right.txt");

    auto depthMap = ivd::stereo::calculateDepthMap(disparity, K_left, T_left, T_right);

    ASSERT_FALSE(depthMap.empty());

    auto depth = ivd::stereo::getDepth(depthMap, cv::Rect(334, 224, 2, 2));
    ASSERT_NEAR(11.872, depth, 0.001);
}
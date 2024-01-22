#include <test.hpp>

#include <stereo/disparity.hpp>
#include <opencv2/opencv.hpp>

using namespace ivd::test;

TEST(Disparity, SGBM_SmokeTest) {
    auto baseDir = getFixturesPath() / "stereo" / "00_basic";
    auto left = cv::imread(baseDir / "left.png");
    auto right = cv::imread(baseDir / "right.png");
    cv::cvtColor(left, left, cv::ColorConversionCodes::COLOR_RGB2GRAY);
    cv::cvtColor(right, right, cv::ColorConversionCodes::COLOR_RGB2GRAY);
    auto disparity = ivd::stereo::disparityMapSGBM(left, right);
    ASSERT_FALSE(disparity.empty());
    ASSERT_EQ(disparity.size(), left.size());
    ASSERT_EQ(disparity.type(), CV_32F);
    ASSERT_EQ(disparity.channels(), 1);
    ASSERT_EQ(disparity.dims, 2);
    cv::normalize(disparity, disparity, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite(baseDir / "disparity_out.png", disparity);
    const float threshold = 0.003;
    auto expected = cv::imread(baseDir / "disparity_expected.png", cv::IMREAD_GRAYSCALE);
    auto diff = disparity.clone();
    auto mismatched = pixelmatch(disparity.data, disparity.channels() * disparity.size().width, expected.data,
                                 expected.channels() * expected.size().width, disparity.size().width,
                                 disparity.size().height, diff.data, threshold);
    cv::imwrite(baseDir / "disparity_diff.png", diff);
    EXPECT_LT(mismatched, disparity.total() * threshold);
}
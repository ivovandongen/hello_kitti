#include <test.hpp>

#include <common/opencv_utils.hpp>

#include <fstream>

using namespace ivd::test;

TEST(OpenCV_Utils, Median_int) {
    cv::Mat in{(cv::Mat_<int>(1, 9) << 5, 6, 1, 2, 3, 4, 6, 777, 8)};
    ASSERT_EQ(ivd::common::median<int>(in), 5);
}

TEST(OpenCV_Utils, Median_float) {
    cv::Mat in{(cv::Mat_<float>(1, 9) << 5.3, 6.2, 1.3, 2.1, 3.9, 4.5, 6.5, 777.2, 8.8)};
    ASSERT_FLOAT_EQ(ivd::common::median<float>(in), 5.3);
}
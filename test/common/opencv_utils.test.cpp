#include <test.hpp>

#include <common/opencv_utils.hpp>

#include <fstream>

using namespace ivd;
using namespace ivd::test;

TEST(OpenCV_Utils, Median_int) {
    cv::Mat in{(cv::Mat_<int>(1, 9) << 5, 6, 1, 2, 3, 4, 6, 777, 8)};
    ASSERT_EQ(ivd::common::median<int>(in), 5);
}

TEST(OpenCV_Utils, Median_float) {
    cv::Mat in{(cv::Mat_<float>(1, 9) << 5.3, 6.2, 1.3, 2.1, 3.9, 4.5, 6.5, 777.2, 8.8)};
    ASSERT_FLOAT_EQ(ivd::common::median<float>(in), 5.3);
}

TEST(OpenCV_Utils, Median_filtered) {
    cv::Mat in{(cv::Mat_<float>(1, 5) << 30, 10, -1, 0, 20)};
    ASSERT_FLOAT_EQ(*ivd::common::median<float>(in, [](auto &val) {
        return val > 0;
    }), 20);
}

TEST(OpenCV_Utils, MakeMat) {
    auto mat = common::createMat<double>(cv::Size{3, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
    ASSERT_EQ(cv::Size(3, 3), mat.size());
    ASSERT_EQ(1, mat.channels());
    ASSERT_EQ(9, mat.total());
    common::iterate<double>(mat, [](auto &val, auto row, auto col) {
        ASSERT_EQ(row * 3 + col + 1, val);
    });
}

TEST(OpenCV_Utils, MakeMat_Multiple_Channels) {
    auto mat = common::createMat<double>(cv::Size{2, 2}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, 2);
    ASSERT_EQ(cv::Size(2, 2), mat.size());
    ASSERT_EQ(2, mat.channels());
    ASSERT_EQ(4, mat.total());
    size_t idx = 0;
    std::for_each(mat.begin<cv::Point2d>(), mat.end<cv::Point2d>(), [&](auto &val) {
        ASSERT_EQ(val.x, idx + 1);
        ASSERT_EQ(val.y, idx + 2);
        idx += 2;
    });
}
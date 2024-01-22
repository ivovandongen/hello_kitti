#include <test.hpp>

#include <common/projection.hpp>
#include <common/opencv_utils.hpp>

#include <fstream>

using namespace ivd::test;

TEST(Projection, DecomposeProjectionMatrix) {
    cv::Mat projectionMatrix = (cv::Mat_<double>(3, 4)
            << 7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01, 0.000000e+00, 7.215377e+02,
            1.728540e+02, 2.163791e-01, 0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03);

    auto [camera, rotation, translation] = ivd::common::decomposeProjectionMatrix(projectionMatrix);
    ASSERT_EQ(camera.size(), cv::Size(3, 3));
    ASSERT_EQ(rotation.size(), cv::Size(3, 3));
    ASSERT_EQ(translation.size(), cv::Size(1, 4));
}

TEST(Projection, ComposeRigidTransformationMatrix) {
    cv::Mat rotation((cv::Mat_<double>(3, 3) << 0, 1, 2, 3, 4, 5, 6, 7, 8));
    cv::Mat translation((cv::Mat_<double>(3, 1) << 9, 10, 11));
    auto result = ivd::common::composeRigidTransformationMatrix(rotation, translation);

    ASSERT_EQ(result.size(), cv::Size(4, 4));

    ivd::common::iterate<double>(rotation, [&](auto &val, auto row, auto col) {
        ASSERT_DOUBLE_EQ(val, result.at<double>(row, col));
    });
    ivd::common::iterate<double>(translation, [&](auto &val, auto row, auto col) {
        ASSERT_DOUBLE_EQ(val, result.at<double>(row, 3));
    });

    ASSERT_DOUBLE_EQ(result.at<double>(3, 0), 0);
    ASSERT_DOUBLE_EQ(result.at<double>(3, 1), 0);
    ASSERT_DOUBLE_EQ(result.at<double>(3, 2), 0);
    ASSERT_DOUBLE_EQ(result.at<double>(3, 3), 1);
}
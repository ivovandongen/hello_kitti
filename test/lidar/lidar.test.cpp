#include <test.hpp>

#include <common/opencv_utils.hpp>
#include <kitti/kitti_utils.hpp>
#include <lidar/lidar.hpp>
#include <lidar/visualize.hpp>

#include <npy.hpp>

using namespace ivd;
using namespace ivd::test;


TEST(Lidar, MakeHomogeneous_Single) {
    auto input = common::createMat<double>({3, 1}, {2.0, 3.0, 4.0});
    auto output = lidar::makeHomogeneous(input);
    ASSERT_EQ(cv::Size(4, 1), output.size());
    ASSERT_EQ(output.at<double>(0, 0), 2);
    ASSERT_EQ(output.at<double>(0, 1), 3);
    ASSERT_EQ(output.at<double>(0, 2), 4);
    ASSERT_EQ(output.at<double>(0, 3), 1);
}

TEST(Lidar, MakeHomogeneous_Multiple) {
    auto input = common::createMat<double>({3, 2}, {2.0, 3.0, 4.0, 5.0, 6.0, 7.0});
    auto output = lidar::makeHomogeneous(input);
    ASSERT_EQ(cv::Size(4, 2), output.size());
    ASSERT_EQ(output.at<double>(0, 0), 2);
    ASSERT_EQ(output.at<double>(0, 1), 3);
    ASSERT_EQ(output.at<double>(0, 2), 4);
    ASSERT_EQ(output.at<double>(0, 3), 1);
    ASSERT_EQ(output.at<double>(1, 0), 5);
    ASSERT_EQ(output.at<double>(1, 1), 6);
    ASSERT_EQ(output.at<double>(1, 2), 7);
    ASSERT_EQ(output.at<double>(1, 3), 1);
}

TEST(Lidar, Project_Identity) {
    auto input = common::createMat<double>({4, 2}, {1, 1, 1, 1,
                                                    2, 4, 2, 1});
    auto T = common::createMat<double>({4, 3}, {1, 0, 0, 0,
                                                0, 1, 0, 0,
                                                0, 0, 1, 0});

    auto output = lidar::project(input, T);
    ASSERT_EQ(cv::Size(2, 3), output.size());

    ASSERT_EQ(output.at<double>(0, 0), 1);
    ASSERT_EQ(output.at<double>(1, 0), 1);
    ASSERT_EQ(output.at<double>(2, 0), 1);

    ASSERT_EQ(output.at<double>(0, 1), 1);
    ASSERT_EQ(output.at<double>(1, 1), 2);
    ASSERT_EQ(output.at<double>(2, 1), 2);
}

TEST(Lidar, E2E) {
    auto input = common::createMat(cv::Size(3, 4), {
            78.37, 10.449, 2.883,
            74.894, 10.464, 2.766,
            73.294, 10.358, 2.712,
            71.736, 10.367, 2.66});
    auto T = common::createMat(cv::Size(4, 3), {
            6.09695406e+02, -7.21421595e+02, -1.25125972e+00, -7.83959658e+01,
            1.80384201e+02, 7.64480142e+00, -7.19651522e+02, -1.00984287e+02,
            9.99945384e-01, 1.24365765e-04, 1.04513027e-02, -2.66641028e-01
    });

    auto homogeneous = lidar::makeHomogeneous(input);
    auto result = lidar::project(homogeneous, T);
    ASSERT_EQ(cv::Size(input.rows, input.cols), result.size());
    ASSERT_NEAR(result.at<double>(0, 0), 514.03340, 0.0001);
    ASSERT_NEAR(result.at<double>(1, 0), 154.11202, 0.0001);
    ASSERT_NEAR(result.at<double>(2, 0), 78.13051, 0.0001);

    ASSERT_NEAR(result.at<double>(0, 1), 509.44331, 0.0001);
    ASSERT_NEAR(result.at<double>(1, 1), 154.02027, 0.0001);
    ASSERT_NEAR(result.at<double>(2, 1), 74.65348, 0.0001);
    common::print<double>(result);
}

TEST(Lidar, Visualize) {
    auto basePath = getFixturesPath() / "lidar" / "00_basic";
    auto image = cv::imread(basePath / "left.png");
    auto npydata = npy::read_npy<double>(basePath / "lidar_points_cam.npy");
    std::vector<double> data = npydata.data;
    std::vector<unsigned long> shape = npydata.shape;

    ASSERT_EQ(shape[1], 3); // 3 rows (u,v,z)

    cv::Mat points(cv::Size(shape[0], shape[1]), CV_64F, data.data());

    lidar::visualizeLidarPoints(points, image);
    cv::imwrite(basePath / "Visualize_out.png", image);

    const float threshold = 0.005;
    auto expected = cv::imread(basePath / "Visualize_expected.png");
    auto diff = image.clone();
    auto mismatched = pixelmatch(image.data, image.channels() * image.size().width, expected.data,
                                 expected.channels() * expected.size().width, image.size().width,
                                 image.size().height, diff.data, threshold);
    cv::imwrite(basePath / "Visualize_diff.png", diff);
    EXPECT_LT(mismatched, image.total() * threshold);
}

TEST(Lidar, E2E_Image) {
    auto basePath = getFixturesPath() / "lidar" / "00_basic";
    auto points = kitti::readVeloBin(basePath / "lidar_points_raw.bin");
    auto image = cv::imread(basePath / "left.png");
    auto T = common::createMat(cv::Size(4, 3), {
            6.09695406e+02, -7.21421595e+02, -1.25125972e+00, -7.83959658e+01,
            1.80384201e+02, 7.64480142e+00, -7.19651522e+02, -1.00984287e+02,
            9.99945384e-01, 1.24365765e-04, 1.04513027e-02, -2.66641028e-01
    });

    // Convert raw velo points into xyz in velo frame
    cv::Mat data(points.size(), 4, CV_32F, points.data());
    data = data.colRange(0, 3);
    ASSERT_FLOAT_EQ(data.at<float>(0, 0), points[0][0]);
    ASSERT_FLOAT_EQ(data.at<float>(1, 0), points[1][0]);
    ASSERT_FLOAT_EQ(data.at<float>(points.size() - 1, 0), points[points.size() - 1][0]);

    //
    auto homogeneous = lidar::makeHomogeneous(data);
    auto result = lidar::project(homogeneous, T, image.size());

    // Write out numpy file
    npy::npy_data<double> npy_data;
    npy_data.shape = {static_cast<unsigned long>(result.cols), static_cast<unsigned long>(result.rows)};
    npy_data.fortran_order = true;
    common::copy(result, npy_data.data);
    npy::write_npy(basePath / "E2E_Image-lidar-points_out.npy", npy_data);

    lidar::visualizeLidarPoints(result, image);
    cv::imwrite(basePath / "E2E_Image-image_out.png", image);

    const float threshold = 0.005;
    auto expected = cv::imread(basePath / "E2E_Image-image_expected.png");
    auto diff = image.clone();
    auto mismatched = pixelmatch(image.data, image.channels() * image.size().width, expected.data,
                                 expected.channels() * expected.size().width, image.size().width,
                                 image.size().height, diff.data, threshold);
    cv::imwrite(basePath / "E2E_Image-image_diff.png", diff);
    EXPECT_LT(mismatched, image.total() * threshold);
}
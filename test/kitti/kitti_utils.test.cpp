#include <test.hpp>

#include <kitti/kitti_utils.hpp>

using namespace ivd::test;

TEST(KittiUtils, ParseProjectionMatrix) {
    std::vector<double> matrix{7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01, 0.000000e+00, 7.215377e+02,
                               1.728540e+02, 2.163791e-01, 0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03};
    YAML::Node input;
    input = matrix;
    ASSERT_TRUE(input.IsSequence());
    ASSERT_EQ(input.size(), 12);
    auto parsed = ivd::kitti::parseMatrix(input, 3, 4);
    ASSERT_EQ(parsed.rows, 3);
    ASSERT_EQ(parsed.cols, 4);

    for (size_t i = 0; i < matrix.size(); i++) {
        ASSERT_DOUBLE_EQ(matrix[i], parsed.at<double>(i));
    }
}
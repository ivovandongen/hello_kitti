#include <kitti/kitti_utils.hpp>

namespace ivd::kitti {
    cv::Mat parseMatrix(const YAML::Node &input, int rows, int cols) {
        assert(input.IsSequence());
        assert(input.size() == rows * cols);
        cv::Mat matrix(rows, cols, CV_64F);
        for (int i = 0; i < input.size(); i++) {
            matrix.at<double>(i) = input[i].as<double>();
        }
        return matrix;
    }
}
#include <common/projection.hpp>

namespace ivd::common {
    ProjectionMatrixDecomposition decomposeProjectionMatrix(const cv::Mat &projectionMatrix) {
        assert(projectionMatrix.rows == 3);
        assert(projectionMatrix.cols == 4);
        cv::Mat K{3, 3, CV_64F}, R{3, 3, CV_64F}, T{4, 1, CV_64F};
        cv::decomposeProjectionMatrix(projectionMatrix, K, R, T);
        T = T / T.at<double>(3);
        return {K, R, T};
    }

    cv::Mat composeRigidTransformationMatrix(const cv::Mat &rotation, const cv::Mat &translation) {
        assert(rotation.size() == cv::Size(3,3));
        assert(translation.size() == cv::Size(1, 3));

        cv::Mat out;
        cv::hconcat(rotation, translation, out);
        assert(out.size() == cv::Size(4,3));
        cv::vconcat(out, cv::Mat{(cv::Mat_<double>(1, 4) << 0, 0, 0, 1)}, out);
        assert(out.size() == cv::Size(4,4));
        return out;
    }
}
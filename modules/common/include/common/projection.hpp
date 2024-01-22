#pragma once

#include <opencv2/opencv.hpp>

namespace ivd::common {
    struct ProjectionMatrixDecomposition {
        cv::Mat cameraIntrinsic;
        cv::Mat rotation;
        cv::Mat translation;
    };

    ProjectionMatrixDecomposition decomposeProjectionMatrix(const cv::Mat &projectionMatrix);

    cv::Mat composeRigidTransformationMatrix(const cv::Mat &rotation,
                                             const cv::Mat &translation = cv::Mat((cv::Mat_<double>(3, 1) << 0, 0, 0)));
}
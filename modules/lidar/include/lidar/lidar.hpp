#pragma once

#include <opencv2/opencv.hpp>

namespace ivd::lidar {

    cv::Mat makeHomogeneous(const cv::Mat &points) {
        assert(points.channels() == 1);
        assert(points.cols == 3);

        // Make homogenous
        cv::Mat xyzw;
        cv::Mat cols{points.rows, 1, points.type(), cv::Scalar(1.0)};
        cv::hconcat(points, cols, xyzw);
        assert(xyzw.cols == 4);

        // Ensure doubles
        if (xyzw.type() != CV_64F) {
            cv::Mat out;
            xyzw.convertTo(out, CV_64F);
            xyzw = out;
        }

        return xyzw;
    }

    // TODO: Optimize by removing points that are not possibly in view (eg only positive X for forward looking camera)

    cv::Mat project(const cv::Mat &homogeneous, const cv::Mat &T, const cv::Size &size = {}) {
        assert(homogeneous.type() == T.type());
        assert(homogeneous.cols == 4);
        assert(T.size() == cv::Size(4, 3));

        cv::Mat veloUVZ = T * homogeneous.t();
        assert(veloUVZ.rows == 3);
        assert(veloUVZ.cols == homogeneous.rows);

        // TODO: Filter out Z =< 0 (avoid divide by 0)

        // Divide U,V by Z
        veloUVZ.row(0) /= veloUVZ.row(2);
        veloUVZ.row(1) /= veloUVZ.row(2);

        if (size.width > 0 && size.height > 0) {
            cv::Mat result;
            for (size_t i = 0; i < veloUVZ.cols; i++) {
                auto col = veloUVZ.col(i);

                auto u = col.at<double>(0);
                auto v = col.at<double>(1);

                // Skip out of range UV coordinates
                if (u < 0 || u > size.width || v < 0 || v > size.height) {
                    continue;
                }

                result.push_back(col.t());
            }

            return result.t();
        } else {
            return veloUVZ;
        }
    }
}
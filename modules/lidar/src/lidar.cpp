#include <lidar/lidar.hpp>

#include <common/opencv_utils.hpp>

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

    cv::Mat project(const cv::Mat &homogeneous, const cv::Mat &T, const cv::Size &size) {
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

    cv::Mat depthMapFromProjectedPoints(const cv::Mat &points, cv::Size size) {
        assert(points.rows == 3);
        assert(points.type() == CV_64F);

        cv::Mat result(size, points.type(), cv::Scalar(0));
        for (size_t col = 0; col < points.cols; col++) {
            auto point = points.col(col);
            auto u = point.at<double>(0);
            auto v = point.at<double>(1);
            auto z = point.at<double>(2);

            assert(u >= 0);
            assert(v >= 0);
            assert(u < size.width);
            assert(v < size.height);
            result.at<double>(v, u) = z;
        }
        return result;
    }

    std::optional<double> getDepth(const cv::Mat &depthMap, const cv::Rect &bbox) {
        return common::median<double>(depthMap(bbox), [](auto &val) { return val > 0; });
    }

    std::optional<double> getDepth(const cv::Mat &depthMap, const cv::Rect &bbox, const cv::Mat &mask) {
        // TODO: avoid copies
        auto boxed = depthMap(bbox);
        cv::Mat masked;
        boxed.copyTo(masked, mask);
        return common::median<double>(masked, [](auto &val) { return val > 0; });
    }
}
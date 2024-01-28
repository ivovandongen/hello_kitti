#include <lidar/visualize.hpp>

#include <colormap/colormap.hpp>

namespace ivd::lidar {

    void visualizeLidarPoints(cv::Mat &image, const cv::Mat &points, const std::string &palette,
                              double maxDistance) {
        assert(points.rows == 3);

        // Pretty colors
        auto pal = colormap::palettes.at(palette).rescale(0, maxDistance);

        auto u = points.row(0);
        auto v = points.row(1);
        auto z = points.row(2);

        for (size_t i = 0; i < u.total(); i++) {
            auto color = pal(z.at<double>(i));

            cv::circle(image, cv::Point(u.at<double>(i), v.at<double>(i)), 1,
                       cv::Scalar(color.getBlue().getValue(), color.getGreen().getValue(), color.getRed().getValue()),
                       -1);
        }
    }

    void visualizeLidarDepthForBBox(cv::Mat &image, const cv::Mat &depthMap, const cv::Rect &bbox,
                                    const cv::Mat &mask,
                                    const std::string &palette, double maxDistance) {

        // Pretty colors
        auto pal = colormap::palettes.at(palette).rescale(0, maxDistance);

        // Copy lidar distances within bbox and optional mask
        cv::Mat masked;
        auto boxed = depthMap(bbox);
        if (mask.empty()) {
            masked = boxed;
        } else {
            boxed.copyTo(masked, mask);
        }

        // For all points > 0 draw a circle (upscaling with the BBox)
        for (size_t r = 0; r < masked.rows; r++) {
            for (size_t c = 0; c < masked.cols; c++) {
                auto z = masked.at<double>(r, c);
                if (z > 0) {
                    auto color = pal(z);

                    cv::circle(image, cv::Point(bbox.x + c, bbox.y + r), 1,
                               cv::Scalar(color.getBlue().getValue(), color.getGreen().getValue(),
                                          color.getRed().getValue()),
                               -1);
                }
            }

        }
    }

}

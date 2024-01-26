#include <lidar/visualize.hpp>

#include <colormap/colormap.hpp>

namespace ivd::lidar {

    void visualizeLidarPoints(const cv::Mat &points, cv::Mat &image, const std::string &palette,
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

}

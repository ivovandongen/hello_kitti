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

    std::vector<std::array<float, 4>> readVeloBin(const std::filesystem::path &file) {
        // allocate 4 MB buffer (only ~130*4*4 KB are needed)
        std::vector<std::array<float, 4>> buffer;
        buffer.resize(1000000);

        // load point cloud
        FILE *stream = std::fopen(file.c_str(), "rb");
        auto numPoints = std::fread(buffer.data(), sizeof(float), buffer.size(), stream) / 4;
        std::fclose(stream);

        // Cut buffer size back
        buffer.resize(numPoints);
        return buffer;
    }
}
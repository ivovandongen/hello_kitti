#include <opencv2/opencv.hpp>

#include <filesystem>
#include <fstream>

namespace ivd::test {
    static cv::Mat readMat(const std::filesystem::path &input) {
        std::vector<double> raw;
        std::fstream inf{input};
        std::string line;
        while (getline(inf, line)) {
            raw.push_back(std::stod(line));
        }
        inf.close();
        return cv::Mat{raw, true};
    }
}
#include <test.hpp>
#include <stereo_test_utils.hpp>

#include <stereo/depth_map.hpp>

#include <opencv2/opencv.hpp>

#include <fstream>

using namespace ivd::test;

namespace {
    cv::Mat readDisparityBin(const std::filesystem::path &path) {
        std::ifstream input(path, std::ios::binary);

        // copies all data into buffer
        std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(input), {});

        // Reformat and return
        cv::Mat result(375, 1242, CV_32F, buffer.data());
        return result;
    }
}

TEST(DepthMap, CalculateDepthMap) {
    auto baseDir = getFixturesPath() / "stereo" / "00_basic";
    auto disparity = readDisparityBin(baseDir / "disparity.bin");
    ASSERT_FALSE(disparity.empty());

    auto K_left = ivd::test::readMat(baseDir / "K_left.txt");
    auto T_left = readMat(baseDir / "T_left.txt");
    auto T_right = readMat(baseDir / "T_right.txt");

    auto depthMap = ivd::stereo::calculateDepthMap(disparity, K_left, T_left, T_right);

    ASSERT_FALSE(depthMap.empty());
    cv::normalize(depthMap, depthMap, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imwrite(baseDir / "depth_out.png", depthMap);

    const float threshold = 0.005;
    auto expected = cv::imread(baseDir / "depth_expected.png", cv::IMREAD_GRAYSCALE);
    auto diff = depthMap.clone();
    auto mismatched = pixelmatch(depthMap.data, depthMap.channels() * depthMap.size().width, expected.data,
                                 expected.channels() * expected.size().width, depthMap.size().width,
                                 depthMap.size().height, diff.data, threshold);
    cv::imwrite(baseDir / "depthMap_diff.png", diff);
    EXPECT_LT(mismatched, depthMap.total() * threshold);
}
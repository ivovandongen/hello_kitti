#include <stereo/disparity.hpp>

namespace ivd::stereo {

    cv::Mat disparityMapSGBM(const cv::Mat &left, const cv::Mat &right) {
        assert(left.channels() == 1);
        assert(right.channels() == 1);
        assert(left.type() == CV_8U);
        assert(right.type() == CV_8U);
        assert(left.size() == right.size());

        // TODO: parameters
        const constexpr int minDisparity = 0;
        const constexpr int numDisparities = 80;
        const constexpr int blockSize = 11;
        const constexpr int windowSize = 5;

        int P1 = 8 * 3 * std::pow(windowSize, 2);
        int P2 = 32 * 3 * std::pow(windowSize, 2);

        // TODO: COLO
        // TODO: default values for optional arguments differ from header vs online documentation...
        auto sgbm = cv::StereoSGBM::create(minDisparity, numDisparities, blockSize,
                                           P1, P2, 0,
                                           0, 0,
                                           0, 0, cv::StereoSGBM::MODE_SGBM_3WAY);

//        auto sgbm = cv::StereoSGBM::create(minDisparity, numDisparities, blockSize,
//                                           P1, P2, 1,
//                                           0, 5,
//                                           400, 200, cv::StereoSGBM::MODE_SGBM_3WAY);

        cv::Mat disparity;
        sgbm->compute(left, right, disparity);

        // Converting disparity values to CV_32F from CV_16S
        cv::Mat out;
        disparity.convertTo(out, CV_32F);

        // Scaling down the disparity values
        // CV_16S containing a disparity map scaled by 16
        out = out / 16.0f;
        // TODO: Normalize?
        // out = (out / 16.0f - (float) minDisparity) / ((float) numDisparities);

        // TODO: Blur?

        assert(out.size().width == disparity.size().width);
        assert(out.size().height == disparity.size().height);
        assert(out.channels() == disparity.channels());
        return out;
    }

}
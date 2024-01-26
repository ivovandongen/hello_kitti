#pragma once

#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>

namespace ivd::common {

    template<class T>
    double median(const cv::Mat &in) {
        assert(in.elemSize() == sizeof(T));
        std::vector<T> tmp;
        for (int i = 0; i < in.rows; ++i) {
            tmp.insert(tmp.end(), in.ptr<T>(i), in.ptr<T>(i) + in.cols * in.channels());
        }
        std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
        return tmp[tmp.size() / 2];
    }


    template<class T>
    cv::Mat createMat(cv::Size size, std::initializer_list<T> values, int channels = 1) {
        assert(size.area() == values.size() / channels);
        cv::Mat mat{values};
        return mat.reshape(channels, size.height);
    }

    template<class T>
    std::vector<T> toVector(const cv::Mat &input) {
        std::vector<T> vec;
        vec.reserve(input.total());
        vec.assign(input.begin<T>(), input.end<T>());
        return vec;
    }

    template<class T>
    void copy(const cv::Mat &input, std::vector<T> &out) {
        out.reserve(input.total());
        out.assign(input.begin<T>(), input.end<T>());
    }

    template<class T, class Fn>
    void iterate(cv::Mat &mat, const Fn &fn) {
        for (auto row = 0; row < mat.rows; row++) {
            for (auto col = 0; col < mat.cols; col++) {
                fn(mat.at<T>(row, col), row, col);
            }
        }
    }

    template<class T, class Fn>
    void iterate(const cv::Mat &mat, const Fn &fn) {
        for (auto row = 0; row < mat.rows; row++) {
            for (auto col = 0; col < mat.cols; col++) {
                fn(mat.at<T>(row, col), row, col);
            }
        }
    }

    template<class T>
    void print(const cv::Mat &mat) {
        std::ios_base::fmtflags defaults(std::cout.flags());

        iterate<T>(mat, [&](auto &val, auto row, auto col) {
            if (row == 0 && col == 0) {
                std::cout << "[";
            }

            if (col == 0) {
                std::cout << "\n\t[";
            }

            if constexpr (std::is_floating_point<T>::value) {
                std::cout << std::fixed << std::setprecision(5) << val;
            } else {
                std::cout << val;
            }

            if (col == mat.cols - 1) {
                std::cout << "],";
                if (row == mat.rows - 1) {
                    std::cout << "\n]";
                }
            } else {
                std::cout << ", ";
            }
        });

        std::cout.flags(defaults);
    }
}
#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>

namespace ivd::test {

    std::filesystem::path getModelsPath();

    std::filesystem::path getFixturesPath();

    uint64_t pixelmatch(const uint8_t *img1,
                        std::size_t stride1,
                        const uint8_t *img2,
                        std::size_t stride2,
                        std::size_t width,
                        std::size_t height,
                        uint8_t *output = nullptr,
                        double threshold = 0.1,
                        bool includeAA = false);
}
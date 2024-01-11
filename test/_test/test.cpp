#include "test.hpp"

#include <mapbox/pixelmatch.hpp>

namespace ivd::test {

    uint64_t
    pixelmatch(const uint8_t *img1, std::size_t stride1, const uint8_t *img2, std::size_t stride2, std::size_t width,
               std::size_t height, uint8_t *output, double threshold, bool includeAA) {
        return mapbox::pixelmatch(img1, stride1, img2, stride2, width, height, output, threshold, includeAA);
    }

    std::filesystem::path getFixturesPath() {
        return FIXTURES_DIR;
    }

    std::filesystem::path getModelsPath() {
        return MODELS_DIR;
    }

}

#include <test.hpp>

#include <comma/utils.hpp>
#include <common/file.hpp>

using namespace ivd;
using namespace ivd::test;

TEST(CommaUtils, DecompressBZ2Text) {
    auto uncompressed = getFixturesPath() / "comma" / "raw.txt";
    auto compressed = getFixturesPath() / "comma" / "raw.txt.bz2";
    ASSERT_EQ(common::readFile(uncompressed), comma::decompressBZ2(common::readFile(compressed)));
}

TEST(CommaUtils, DecompressBZ2Binary) {
    auto uncompressed = getFixturesPath() / "comma" / "raw.txt";
    auto compressed = getFixturesPath() / "comma" / "raw.txt.bz2";
    ASSERT_EQ(common::readFile(uncompressed, true), comma::decompressBZ2(common::readFile(compressed, true)));
}

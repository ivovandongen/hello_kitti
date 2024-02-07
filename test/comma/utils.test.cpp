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

TEST(CommaUtils, ParseRouteName) {
    const constexpr char *routeName = "a2a0ccea32023010|2023-07-27--13-01-19";
    auto [dongleId, route, success] = comma::parseRouteName(routeName);
    ASSERT_TRUE(success);
    ASSERT_EQ(dongleId, "a2a0ccea32023010");
    ASSERT_EQ(route, "2023-07-27--13-01-19");
}

TEST(CommaUtils, ParseRouteNameInvalid) {
    const constexpr char *routeName = "a2a0ccea32023010-2023-07-27--13-01-19";
    auto [dongleId, route, success] = comma::parseRouteName(routeName);
    ASSERT_FALSE(success);
    ASSERT_EQ(dongleId, "");
    ASSERT_EQ(route, "");
}

TEST(CommaUtils, ParseSegmentName) {
    {
        auto [route, segment, success] = comma::parseSegmentName("2023-07-27--13-01-19--0");
        ASSERT_TRUE(success);
        ASSERT_EQ(segment, 0);
        ASSERT_EQ(route, "2023-07-27--13-01-19");
    }
    {
        auto [route, segment, success] = comma::parseSegmentName("2023-07-27--13-01-19--20");
        ASSERT_TRUE(success);
        ASSERT_EQ(segment, 20);
        ASSERT_EQ(route, "2023-07-27--13-01-19");
    }
}

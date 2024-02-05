#include <test.hpp>

#include <comma/comma.hpp>

using namespace ivd;
using namespace ivd::test;

TEST(Comma, ParseLog) {
    auto logFile = getFixturesPath() / "comma" / "qlog.bz2";
    comma::LogLoader loader{logFile};
    ASSERT_FALSE(loader.isLoaded());

    loader.load();
    ASSERT_TRUE(loader.isLoaded());
    ASSERT_EQ(loader.size(), 14988);
    ASSERT_EQ(loader.events()[0].which, cereal::Event::Which::CAN);
    ASSERT_TRUE(loader.events()[0].event.isCan());
    ASSERT_EQ(loader.events()[0].event.getCan().size(), 41);
}
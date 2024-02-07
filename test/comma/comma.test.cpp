#include <test.hpp>

#include <comma/comma.hpp>

#include <capnp/compat/json.h>

using namespace ivd;
using namespace ivd::test;

TEST(Comma, ParseLog) {
    auto logFile = getFixturesPath() / "comma" / "qlog.bz2";
    comma::LogLoader loader{logFile};
    ASSERT_FALSE(loader.isLoaded());

    loader.load();
    ASSERT_TRUE(loader.isLoaded());
    ASSERT_EQ(loader.size(), 14988);
    ASSERT_EQ(loader.events()[0]->which, cereal::Event::Which::CAN);
    ASSERT_TRUE(loader.events()[0]->event.isCan());
    ASSERT_EQ(loader.events()[0]->event.getCan().size(), 41);
}

TEST(Comma, Player) {
    auto logFile = getFixturesPath() / "comma";
    comma::Player player{logFile, "a2a0ccea32023010", "2023-07-27--13-01-19", true};

    player.preload();
    ASSERT_EQ(player.segments().size(), 1);
    ASSERT_TRUE(player.currentSegment().qLog->isLoaded());
    ASSERT_TRUE(player.currentSegment().qcamera);

    size_t eventCount{0};
    capnp::JsonCodec jsonCodec;
    player.registerEventListener(cereal::Event::Which::RADAR_STATE, [&](const comma::Event &event) {
        ASSERT_EQ(event.which, cereal::Event::Which::RADAR_STATE);
        event.event.getRadarState().getCarStateMonoTime();
        eventCount++;
    });

    while (player.tick()) {
    }

    ASSERT_EQ(eventCount, 232);
}
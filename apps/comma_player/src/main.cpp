#include <comma/comma.hpp>
#include <comma/utils.hpp>

#include <cxxopts.hpp>
#include <opencv2/opencv.hpp>
#include <opendbc/can/common.h>

#include <filesystem>
#include <iostream>

class RadarInterface {
public:
    RadarInterface()  {

    }

private:
    const char * radarDBC = "toyota_tss2_adas";
    const std::array<uint32_t, 16> RADAR_A_MSGS{384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399};
    const std::array<uint32_t, 16> RADAR_B_MSGS{400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415};
    uint32_t triggerMsg = 415;
    std::optional<CANParser> canParser;
};

struct Options {
    std::filesystem::path data;
    std::string route;
    uint32_t wait;
};

Options parseOpts(int argc, char **argv) {
    cxxopts::Options options(argv[0], argv[0]);

    // clang-format off
    options.add_options()
            ("d,data", "Data directory", cxxopts::value<std::string>())
            ("r,route", "Route name. Eg <dongle_id|route_name>", cxxopts::value<std::string>())
            ("w,wait", "Wait time (ms) between frames - 0 == wait indefinitely",
             cxxopts::value<uint32_t>()->default_value("1"))
            ("h,help", "Print usage");
    // clang-format on

    try {
        auto result = options.parse(argc, argv);
        if (result.count("help")) {
            std::cout << options.help().c_str() << std::endl;
            exit(0);
        }

        Options opts{
                result["data"].as<std::string>(),
                result["route"].as<std::string>(),
                result["wait"].as<uint32_t>(),
        };

        return opts;
    } catch (const cxxopts::exceptions::exception &e) {
        std::cerr << "Invalid options: " << e.what() << std::endl;
        std::cout << options.help().c_str() << std::endl;
        exit(1);
    } catch (...) {
        std::cout << options.help().c_str() << std::endl;
        exit(1);
    }
}

int main(int argc, char **argv) {
    using namespace ivd;
    std::cout << "Comma Player" << std::endl;
    auto options = parseOpts(argc, argv);

    if (!exists(options.data)) {
        std::cerr << "Directory: " << options.data << " does not exist" << std::endl;
        exit(1);
    }

    auto [dongle, route, success] = comma::parseRouteName(options.route);
    if (!success) {
        std::cerr << "Invalid route name: " << options.route << std::endl;
        exit(1);
    }

    const constexpr char *roadWindowColor = "road image (Color)";
    const constexpr float windowScale = 0.7;
    cv::namedWindow(roadWindowColor, cv::WINDOW_NORMAL);

    comma::Player player{options.data, dongle, route, false};
    player.preload();
    player.registerEventListener(cereal::Event::Which::CAN, [](const comma::Event &event) {
        std::cout << event.mono_time << std::endl;
//        std::cout << event.event.isRadarState() << std::endl;
//        std::cout << event.event.getRadarState().getCarStateMonoTime() << std::endl;
    });

//    player.registerFrameListener(cereal::Event::Which::ROAD_ENCODE_IDX, [&](const comma::Event &event, cv::Mat frame) {
//        if (!frame.empty()) {
//            cv::imshow(roadWindowColor, frame);
//        } else {
//            std::cerr << "Empty frame for event:\n" << event.json() << std::endl;
//        }
//    });

    while (player.tick()) {
//        cv::waitKey(options.wait);
    }
}
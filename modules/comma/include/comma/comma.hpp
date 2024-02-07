#pragma once

#include <common/string.hpp>

#include <capnp/serialize.h>
#include <cereal/log.capnp.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>

#include <filesystem>
#include <string>
#include <map>

namespace ivd::comma {
    class Event {
    public:
        Event(const kj::ArrayPtr<const capnp::word> &words, bool frame = false);

        struct LessThan {
            inline bool operator()(const Event &l, const Event &r) {
                return l.mono_time < r.mono_time || (l.mono_time == r.mono_time && l.which < r.which);
            }

            inline bool operator()(const std::shared_ptr<Event> &l, const std::shared_ptr<Event> &r) {
                return l->mono_time < r->mono_time || (l->mono_time == r->mono_time && l->which < r->which);
            }
        };

        std::string json() const;

    public:
        cereal::Event::Which which;
        cereal::Event::Reader event;
        uint64_t mono_time;
        bool frame;

        friend class LogLoader;

    private:
        capnp::FlatArrayMessageReader reader;
        kj::ArrayPtr<const capnp::word> words;
    };

    class LogLoader {
    public:
        explicit LogLoader(std::filesystem::path file);

        void load();

        bool isLoaded() const;

        const std::vector<std::shared_ptr<Event>> &events() const;

        size_t size() const;

    private:
        void read();

        void parse();

    private:
        std::filesystem::path path_;
        bool loaded_{false};
        std::string contents_;
        std::vector<std::shared_ptr<Event>> events_;
    };

    class FrameLoader {
    public:
        FrameLoader(std::filesystem::path);

        ~FrameLoader();

        void load();

        cv::Mat get(uint32_t frameIdx);

    private:
        std::filesystem::path file_;
        cv::VideoCapture capture_;
        size_t totalFrames_{};
    };

    class Segment {
    public:
        size_t segmentIdx;
        std::optional<LogLoader> qLog;
        std::optional<LogLoader> rLog;
        std::optional<FrameLoader> qcamera;
        std::optional<FrameLoader> dcamera;
        std::optional<FrameLoader> ecamera;
        std::optional<FrameLoader> fcamera;
    };

    using FrameListener = std::function<void(const Event &, cv::Mat)>;
    using EventListener = std::function<void(const Event &)>;

    class Player {
    public:
        Player(std::filesystem::path dir, std::string dongle, std::string route, bool condensedOnly = false);

        std::map<size_t, Segment> &segments();

        void preload();

        bool tick();

        Segment &currentSegment();

        void registerEventListener(cereal::Event::Which who, EventListener listener);

        void registerFrameListener(cereal::Event::Which who, FrameListener listener);

    private:
        void load();

    private:
        std::filesystem::path dir_;
        std::string dongle_;
        std::string route_;
        bool condensedOnly_;
        size_t currentSegmentIdx_{0};
        uint64_t routeStartMs{};

        std::map<cereal::Event::Which, EventListener> eventListeners;
        std::map<cereal::Event::Which, FrameListener> frameListeners;

        std::map<size_t, Segment> segments_;

    };
}
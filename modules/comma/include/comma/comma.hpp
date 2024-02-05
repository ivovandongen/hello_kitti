#pragma once

#include <cereal/log.capnp.h>
#include <capnp/serialize.h>

#include <filesystem>
#include <string>

namespace ivd::comma {
    class Event {
    public:
        Event(const kj::ArrayPtr<const capnp::word> &words, bool frame = false);

        struct LessThan {
            inline bool operator()(const Event &l, const Event &r) {
                return l.mono_time < r.mono_time || (l.mono_time == r.mono_time && l.which < r.which);
            }
        };

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

        const std::vector<Event> &events() const;

        size_t size() const;

    private:
        void read();

        void parse();

    private:
        std::filesystem::path path_;
        bool loaded_{false};
        std::string contents_;
        std::vector<Event> events_;
    };

}
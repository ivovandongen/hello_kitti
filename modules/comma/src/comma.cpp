#include <comma/comma.hpp>

#include <comma/utils.hpp>
#include <common/file.hpp>

#include <cassert>
#include <utility>

namespace ivd::comma {
    LogLoader::LogLoader(std::filesystem::path dataDir) : path_(std::move(dataDir)) {
    }

    bool LogLoader::isLoaded() const {
        return loaded_;
    }

    void LogLoader::load() {
        if (isLoaded()) {
            return;
        }

        read();
        parse();

        loaded_ = !contents_.empty() && !events_.empty();
    }

    void LogLoader::read() {
        contents_ = common::readFile(path_);

        if (path_.string().find(".bz2") != std::string::npos) {
            contents_ = decompressBZ2(contents_);
        }
    }

    void LogLoader::parse() {
        assert(contents_.size() % sizeof(capnp::word) == 0);

        try {
            assert(contents_.size() % sizeof(capnp::word) == 0);
            kj::ArrayPtr<const capnp::word> allWords((const capnp::word *) contents_.data(),
                                                     contents_.size() / sizeof(capnp::word));
            while (allWords.size() > 0) {
                auto &event = events_.emplace_back(allWords);
                if (event.which == cereal::Event::ROAD_ENCODE_IDX ||
                    event.which == cereal::Event::DRIVER_ENCODE_IDX ||
                    event.which == cereal::Event::WIDE_ROAD_ENCODE_IDX) {
                    // Add encodeIdx packet again as a frame packet for the video stream
                    events_.emplace_back(allWords, true);
                }

                // Forward the array pointer
                allWords = kj::arrayPtr(event.reader.getEnd(), allWords.end());
            }
        } catch (const kj::Exception &e) {
            std::cerr << "Could not parse: " << path_ << " - " << e.getDescription().cStr() << std::endl;
            return;
        }

        if (!events_.empty()) {
            std::sort(events_.begin(), events_.end(), Event::LessThan());
        }
    }

    const std::vector<Event> &LogLoader::events() const {
        return events_;
    }

    size_t LogLoader::size() const {
        return events_.size();
    }

    Event::Event(const kj::ArrayPtr<const capnp::word> &msg, bool frame) : reader(msg), frame(frame) {
        words = kj::ArrayPtr<const capnp::word>(msg.begin(), reader.getEnd());
        event = reader.getRoot<cereal::Event>();
        which = event.which();
        mono_time = event.getLogMonoTime();

        // 1) Send video data at t=timestampEof/timestampSof
        // 2) Send encodeIndex packet at t=logMonoTime
        if (frame) {
            auto idx = capnp::AnyStruct::Reader(event).getPointerSection()[0].getAs<cereal::EncodeIndex>();
            // C2 only has eof set, and some older routes have neither
            uint64_t sof = idx.getTimestampSof();
            uint64_t eof = idx.getTimestampEof();
            if (sof > 0) {
                mono_time = sof;
            } else if (eof > 0) {
                mono_time = eof;
            }
        }
    }

}
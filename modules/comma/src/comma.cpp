#include <comma/comma.hpp>

#include <comma/utils.hpp>
#include <common/file.hpp>
#include <common/string.hpp>

#include <capnp/compat/json.h>
#include <opencv2/opencv.hpp>

#include <cassert>
#include <utility>

namespace {
    using namespace ivd;
    using namespace ivd::comma;

    void populateSegmentFromDataDir(Segment &segment, const std::filesystem::path &segmentDir) {
        for (const auto &segmentFile: std::filesystem::directory_iterator(segmentDir)) {
            if (segmentFile.is_regular_file()) {
                if (segmentFile.path().filename() == "qlog.bz2") {
                    segment.qLog.emplace(segmentFile);
                } else if (segmentFile.path().filename() == "rlog.bz2") {
                    segment.rLog.emplace(segmentFile);
                } else if (segmentFile.path().filename() == "qcamera.ts") {
                    segment.qcamera.emplace(segmentFile);
                } else if (segmentFile.path().filename() == "dcamera.ts") {
                    segment.dcamera.emplace(segmentFile);
                } else if (segmentFile.path().filename() == "ecamera.ts") {
                    segment.ecamera.emplace(segmentFile);
                } else if (segmentFile.path().filename() == "fcamera.ts") {
                    segment.fcamera.emplace(segmentFile);
                }
            }
        }
    }
}

namespace ivd::comma {
    LogLoader::LogLoader(std::filesystem::path dataDir) : path_(std::move(dataDir)) {
    }

    bool LogLoader::isLoaded() const {
        return loaded_;
    }

    void LogLoader::load() {
        std::cout << "Loading: " << path_ << std::endl;
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
                auto event = events_.emplace_back(std::make_shared<Event>(allWords));
                assert(event);
                if (event->which == cereal::Event::ROAD_ENCODE_IDX ||
                    event->which == cereal::Event::DRIVER_ENCODE_IDX ||
                    event->which == cereal::Event::WIDE_ROAD_ENCODE_IDX) {
                    // Add encodeIdx packet again as a frame packet for the video stream
                    events_.emplace_back(std::make_shared<Event>(allWords, true));
                }

                // Forward the array pointer
                allWords = kj::arrayPtr(event->reader.getEnd(), allWords.end());
            }
        } catch (const kj::Exception &e) {
            std::cerr << "Could not parse: " << path_ << " - " << e.getDescription().cStr() << std::endl;
            return;
        }

        if (!events_.empty()) {
            std::sort(events_.begin(), events_.end(), Event::LessThan());
        }
    }

    const std::vector<std::shared_ptr<Event>> &LogLoader::events() const {
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

    std::string Event::json() const {
        return capnp::JsonCodec{}.encode(event).cStr();
    }

    Player::Player(std::filesystem::path dir, std::string dongle, std::string route, bool condensedOnly) :
            dir_(std::move(dir)),
            dongle_(std::move(dongle)),
            route_(std::move(route)),
            condensedOnly_(condensedOnly) {
        load();
    }

    void Player::load() {
        // Find all segments
        auto dongleDir = dir_ / dongle_;
        assert(exists(dongleDir));
        for (const auto &segmentDir: std::filesystem::directory_iterator(dongleDir)) {
            if (segmentDir.is_directory() && common::startsWith(segmentDir.path().filename(), route_)) {
                auto [_, segmentIdx, success] = comma::parseSegmentName(segmentDir.path().filename());
                if (!success) {
                    std::cerr << "Could not parse segment name: " << segmentDir.path().filename() << std::endl;
                    return;
                }
                auto &segment = segments_.emplace(segmentIdx, Segment{}).first->second;
                segment.segmentIdx = segmentIdx;
                populateSegmentFromDataDir(segment, segmentDir);
            }
        }

    }

    std::map<size_t, Segment> &Player::segments() {
        return segments_;
    }

    Segment &Player::currentSegment() {
        return segments_[currentSegmentIdx_];
    }

    bool Player::tick() {
        static size_t index = 0;
        if (currentSegmentIdx_ >= segments_.size()) {
            return false;
        }

        auto &segment = segments_[currentSegmentIdx_];
        auto &loader = !condensedOnly_ && segment.rLog.has_value() ? segment.rLog : segment.qLog;
        if (!loader->isLoaded()) {
            loader->load();
        }

        // Set the start time
        if (index == 0 && currentSegmentIdx_ == 0) {
            routeStartMs = (*loader->events().begin())->mono_time;
        }

        auto begin = loader->events().begin();
        std::advance(begin, index);
        if (begin != loader->events().end()) {
            std::shared_ptr<Event> event = *begin;

            // Check if we need to load a frame
            if (event->frame) {
                auto frameListener = frameListeners.find(event->which);
                if (frameListener != frameListeners.end()) {
                    auto eidx = capnp::AnyStruct::Reader(
                            event->event).getPointerSection()[0].getAs<cereal::EncodeIndex>();
                    if (eidx.getType() == cereal::EncodeIndex::Type::FULL_H_E_V_C) {
                        auto &camSegment = segments_[eidx.getSegmentNum()];
                        frameListener->second(*event, [&]() {
                            // TODO: not precise enough
                            auto frameId = eidx.getFrameId() - currentSegmentIdx_ * 60 * 20; // 60 sec 20fps
                            switch (event->which) {
                                case cereal::Event::ROAD_ENCODE_IDX:
                                    if (!condensedOnly_ && camSegment.fcamera) {
                                        return camSegment.fcamera->get(frameId);
                                    } else {
                                        return camSegment.qcamera->get(frameId);
                                    }
                                case cereal::Event::DRIVER_ENCODE_IDX:
                                    return camSegment.dcamera->get(frameId);
                                case cereal::Event::WIDE_ROAD_ENCODE_IDX:
                                    return camSegment.ecamera->get(frameId);
                            }
                            return cv::Mat{};
                        }());
                    }
                }
            }

            index++;
            auto eventListener = eventListeners.find(event->which);
            if (eventListener != eventListeners.end()) {
                eventListener->second(*event);
            }
            return true;
        } else {
            currentSegmentIdx_++;
            index = 0;
            return tick();
        }
    }

    void Player::preload() {
        auto load = [](auto &loader) {
            if (loader.has_value()) {
                loader->load();
            }
        };

        for (auto &val: segments_) {
            auto &segment = val.second;
            load(segment.qLog);
            load(segment.qcamera);
            if (!condensedOnly_) {
                load(segment.rLog);
                load(segment.ecamera);
                load(segment.fcamera);
                load(segment.dcamera);
            }
        }
    }

    void Player::registerEventListener(cereal::Event::Which who, EventListener listener) {
        eventListeners[who] = std::move(listener);
    }

    void Player::registerFrameListener(cereal::Event::Which who, FrameListener listener) {
        frameListeners[who] = std::move(listener);
    }

    FrameLoader::FrameLoader(std::filesystem::path file) : file_(std::move(file)) {}

    FrameLoader::~FrameLoader() {
        if (capture_.isOpened()) {
            capture_.release();
        }
    }

    cv::Mat FrameLoader::get(uint32_t frameIdx) {
        cv::Mat out;
        if (frameIdx >= totalFrames_) {
            // No-no
            return {};
        } else {
            capture_.set(cv::CAP_PROP_POS_FRAMES, frameIdx);
            capture_.read(out);
        }

        return out;
    }

    void FrameLoader::load() {
        capture_ = cv::VideoCapture(file_);
        totalFrames_ = capture_.get(cv::CAP_PROP_FRAME_COUNT);
    }
}
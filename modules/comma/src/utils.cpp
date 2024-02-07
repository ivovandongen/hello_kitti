#include <comma/utils.hpp>

#include <bzlib.h>

#include <cassert>
#include <iostream>

namespace {
    std::string decompressBZ2(const std::byte *in, size_t in_size) {
        if (in_size == 0) return {};

        bz_stream strm = {};
        int bzerror = BZ2_bzDecompressInit(&strm, 0, 0);
        assert(bzerror == BZ_OK);

        strm.next_in = (char *) in;
        strm.avail_in = in_size;
        std::string out(in_size * 5, '\0');
        do {
            strm.next_out = (char *) (&out[strm.total_out_lo32]);
            strm.avail_out = out.size() - strm.total_out_lo32;

            const char *prev_write_pos = strm.next_out;
            bzerror = BZ2_bzDecompress(&strm);
            if (bzerror == BZ_OK && prev_write_pos == strm.next_out) {
                // content is corrupt
                bzerror = BZ_STREAM_END;
                std::cerr << "decompressBZ2 error : content is corrupt";
                break;
            }

            if (bzerror == BZ_OK && strm.avail_in > 0 && strm.avail_out == 0) {
                out.resize(out.size() * 2);
            }
        } while (bzerror == BZ_OK);

        BZ2_bzDecompressEnd(&strm);
        if (bzerror == BZ_STREAM_END) {
            out.resize(strm.total_out_lo32);
            return out;
        }
        return {};
    }
}

namespace ivd::comma {
    std::string decompressBZ2(const std::string &input) {
        return ::decompressBZ2((std::byte *) input.data(), input.size());
    }

    std::tuple<std::string, std::string, bool> parseRouteName(const std::string &input) {
        auto delimiterIdx = input.find('|');
        if (delimiterIdx == std::string::npos) {
            return {};
        }
        return {input.substr(0, delimiterIdx), input.substr(delimiterIdx + 1), true};
    }

    std::tuple<std::string, size_t, bool> parseSegmentName(const std::string &input) {
        auto delimiterIdx = input.rfind("--");
        if (delimiterIdx == std::string::npos) {
            return {};
        }
        return {input.substr(0, delimiterIdx), stoul(input.substr(delimiterIdx + 2)), true};
    }
}

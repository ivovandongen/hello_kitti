#pragma once

#include <string>

namespace ivd::comma {
    std::string decompressBZ2(const std::string &input);

    /**
     * @return <dongle id, route, success>
     */
    std::tuple<std::string, std::string, bool> parseRouteName(const std::string &input);

    /**
     * @return <route, segment id, success>
     */
    std::tuple<std::string, size_t, bool> parseSegmentName(const std::string &input);
}
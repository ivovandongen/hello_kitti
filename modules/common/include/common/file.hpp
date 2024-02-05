#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <streambuf>
#include <string>

namespace ivd::common {

    inline std::string readFile(const std::filesystem::path &file, bool binary = false) {
        auto flags = std::ios::in;
        if (binary) {
            flags |= std::ios::binary;
        }
        std::ifstream t(file, flags);
        std::string result;

        if (t.is_open()) {
            t.seekg(0, std::ios::end);
            result.reserve(t.tellg());
            t.seekg(0, std::ios::beg);

            result.assign((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
        } else {
            throw std::runtime_error{std::string{"Could not open file: "} + file.string()};
        }

        return result;
    }

}

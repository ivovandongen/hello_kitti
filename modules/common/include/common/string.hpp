#include <string>

namespace ivd::common {

    inline bool startsWith(const std::string &input, const std::string &prefix) {
        if (input.length() >= prefix.length()) {
            return (0 == input.rfind(prefix, 0));
        } else {
            return false;
        }
    }

    inline bool endsWith(const std::string &input, const std::string &suffix) {
        if (input.length() >= suffix.length()) {
            return (0 == input.compare(input.length() - suffix.length(), suffix.length(), suffix));
        } else {
            return false;
        }
    }

}
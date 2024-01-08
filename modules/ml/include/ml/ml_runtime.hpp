#pragma once

#include <onnxruntime_cxx_api.h>

#include <memory>
#include <filesystem>

namespace ivd::ml {
    class MLRuntime {
    public:
        MLRuntime();
        static std::shared_ptr<MLRuntime> Get();

        std::vector<std::string> availableProviders() const;

        Ort::Session createSession(const std::filesystem::path& model) const;

    private:
        static std::weak_ptr<MLRuntime> instance_;

        Ort::Env env_;
    };
}
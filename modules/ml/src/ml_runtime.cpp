#include <ml/ml_runtime.hpp>

#include <onnxruntime_cxx_api.h>
#include <coreml_provider_factory.h>

namespace ivd::ml {

    std::weak_ptr<MLRuntime> MLRuntime::instance_;

    std::shared_ptr<MLRuntime> MLRuntime::Get() {
        if (auto mlRuntime = instance_.lock()) {
            return mlRuntime;
        } else {
            instance_ = mlRuntime = std::make_shared<MLRuntime>();
            return mlRuntime;
        }
    }

    MLRuntime::MLRuntime() {
        env_ = Ort::Env(ORT_LOGGING_LEVEL_VERBOSE, "ML");
    }

    std::vector<std::string> MLRuntime::availableProviders() const {
        return Ort::GetAvailableProviders();
    }

    Ort::Session MLRuntime::createSession(const std::filesystem::path &model) const {
        Ort::SessionOptions session_options;
#ifndef NDEBUG
        session_options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_VERBOSE);
#endif
        session_options.SetExecutionMode(ORT_PARALLEL);
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
#ifdef __APPLE__
        uint32_t coreml_flags = COREML_FLAG_ENABLE_ON_SUBGRAPH;
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(session_options, coreml_flags));
#endif
        return {env_, model.c_str(), session_options};
    }
}
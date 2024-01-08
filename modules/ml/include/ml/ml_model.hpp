#pragma once

#include <ml/ml_runtime.hpp>

#include <filesystem>
#include <vector>

namespace ivd::ml {

    struct Node {
        std::string name;
        std::vector<int64_t> dimensions;
    };

    class MLModel {
    public:
        explicit MLModel(std::filesystem::path modelPath);

        virtual ~MLModel() = default;

        const std::vector<Node>& inputNodes() const {
            return inputs_;
        }
        const std::vector<Node>& outputNodes() const {
            return outputs_;
        }

    protected:
        std::filesystem::path modelPath_;
        std::shared_ptr<MLRuntime> mlRuntime_;
        Ort::Session session_{nullptr};
        std::vector<Node> inputs_;
        std::vector<Node> outputs_;
    };

}
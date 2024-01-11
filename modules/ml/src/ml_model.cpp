#include <ml/ml_model.hpp>

#include <cassert>
#include <iostream>
#include <numeric>

namespace ivd::ml {
    MLModel::MLModel(std::filesystem::path modelPath) : modelPath_(std::move(modelPath)), mlRuntime_(MLRuntime::Get()) {
        assert(mlRuntime_);
        session_ = mlRuntime_->createSession(modelPath_);

        // Register inputs and outputs
        Ort::AllocatorWithDefaultOptions allocator;
        for (std::size_t i = 0; i < session_.GetInputCount(); i++) {
            inputs_.push_back(Node{session_.GetInputNameAllocated(i, allocator).get(),
                                   session_.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape()});
        }

        for (std::size_t i = 0; i < session_.GetOutputCount(); i++) {
            outputs_.push_back(Node{session_.GetOutputNameAllocated(i, allocator).get(),
                                    session_.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape()});
        }

        std::transform(inputs_.begin(), inputs_.end(), std::back_inserter(inputNames_),
                       [](const Node &node) { return node.name.c_str(); });
        std::transform(outputs_.begin(), outputs_.end(), std::back_inserter(outputNames_),
                       [](const Node &node) { return node.name.c_str(); });
#ifndef NDEBUG
        auto printNode = [](const Node &node) {
            auto dimensions = node.dimensions.empty() ? "" : std::accumulate(std::next(node.dimensions.begin()),
                                                                             node.dimensions.end(),
                                                                             std::to_string(node.dimensions[0]),
                                                                             [](const auto &a, auto b) {
                                                                                 return a + "," + std::to_string(b);
                                                                             });
            std::cout << "\t" << node.name << ": [" << dimensions << "]\n";
        };
        std::cout << "Inputs: \n";
        for (auto &node: inputs_) {
            printNode(node);
        }

        std::cout << "\n";
        std::cout << "Outputs: \n";
        for (auto &node: outputs_) {
            printNode(node);
        }
        std::cout << "\n";
#endif
    }
}
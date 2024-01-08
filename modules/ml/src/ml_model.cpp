#include <ml/ml_model.hpp>

#include <cassert>

namespace ivd::ml {
    MLModel::MLModel(std::filesystem::path modelPath) : modelPath_(std::move(modelPath)), mlRuntime_(MLRuntime::Get()) {
        assert(mlRuntime_);
        session_ = mlRuntime_->createSession(modelPath_);

        // print name/shape of inputs
        Ort::AllocatorWithDefaultOptions allocator;
        for (std::size_t i = 0; i < session_.GetInputCount(); i++) {
            inputs_.push_back(Node{session_.GetInputNameAllocated(i, allocator).get(),
                                   session_.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape()});
        }

        for (std::size_t i = 0; i < session_.GetOutputCount(); i++) {
            outputs_.push_back(Node{session_.GetOutputNameAllocated(i, allocator).get(),
                                    session_.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape()});
        }
//        // some models might have negative shape values to indicate dynamic shape, e.g., for variable batch size.
//        for (auto &s: input_shapes) {
//            if (s < 0) {
//                s = 1;
//            }
//        }
    }
}
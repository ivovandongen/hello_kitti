#include <test.hpp>

#include <ml/ml_model.hpp>

TEST(MLModel, load) {
    ivd::ml::MLModel{getModelsPath() / "yolo" / "yolov8n.onnx"};
}
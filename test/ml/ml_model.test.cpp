#include <test.hpp>

#include <ml/ml_model.hpp>

using namespace ivd::test;

TEST(MLModel, load) {
    ivd::ml::MLModel model{getModelsPath() / "yolo" / "yolov8n.onnx"};
    ASSERT_EQ(model.inputNodes().size(), 1);
    ASSERT_STREQ(model.inputNames()[0], "images");
    ASSERT_EQ(model.inputNodes()[0].dimensions.size(), 4);
    ASSERT_EQ(model.inputNodes()[0].dimensions[0], 1);
    ASSERT_EQ(model.inputNodes()[0].dimensions[1], 3);
    ASSERT_EQ(model.inputNodes()[0].dimensions[2], 640);
    ASSERT_EQ(model.inputNodes()[0].dimensions[3], 640);

    ASSERT_EQ(model.outputNodes().size(), 1);
    ASSERT_STREQ(model.outputNames()[0], "output0");
    ASSERT_EQ(model.outputNodes()[0].dimensions.size(), 3);
    ASSERT_EQ(model.outputNodes()[0].dimensions[0], 1);
    ASSERT_EQ(model.outputNodes()[0].dimensions[1], 84);
    ASSERT_EQ(model.outputNodes()[0].dimensions[2], 8400);
}
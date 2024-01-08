#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>

static std::filesystem::path getModelsPath() {
    return MODELS_DIR;
}

static std::filesystem::path getFixturesPath() {
    return FIXTURES_DIR;
}
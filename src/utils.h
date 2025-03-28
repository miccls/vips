#pragma once

#include <format>
#include <iostream>

namespace utils {

void warn(const std::string &msg) {
    std::cout << std::format("Warning: {}", msg) << std::endl;
}

}  // namespace utils

cmake_minimum_required(VERSION 3.14)
project(cpp-grad LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 安全編譯選項
add_compile_options(-Wall -Wextra -Werror -fsanitize=address,undefined)
add_link_options(-fsanitize=address,undefined)

# Header-only library
add_library(cpp-grad INTERFACE)
target_include_directories(cpp-grad INTERFACE include/)

# 測試
enable_testing()
add_executable(tests tests/test_value.cpp)
target_link_libraries(tests cpp-grad)
add_test(NAME all_tests COMMAND tests)

# 範例
add_executable(basic examples/basic.cpp)
target_link_libraries(basic cpp-grad)

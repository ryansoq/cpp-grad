CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -Werror -fsanitize=address,undefined -g -Iinclude

all: test_runner basic

test_runner: tests/test_value.cpp include/cpp_grad.hpp
	$(CXX) $(CXXFLAGS) -o test_runner tests/test_value.cpp

basic: examples/basic.cpp include/cpp_grad.hpp
	$(CXX) $(CXXFLAGS) -o basic examples/basic.cpp

test: test_runner
	./test_runner

clean:
	rm -f test_runner basic

.PHONY: all test clean

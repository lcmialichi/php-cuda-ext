CXX = g++
CXXFLAGS = -fPIC -O2 -std=c++11

src/cuda_wrapper.lo: src/cuda_wrapper.cpp
	$(CXX) $(CXXFLAGS) -I. $(PHP_INCLUDES) -c $< -o $@

src/cuda_array.lo: src/cuda_array.c
	$(CXX) $(CXXFLAGS) -I. $(PHP_INCLUDES) -c $< -o $@
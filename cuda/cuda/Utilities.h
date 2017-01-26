#pragma once

#include <functional>
#include <iostream>
#include "Defs.h"

inline int ceiling_div(const int x, const int y)
{
	return (x + y - 1) / y;
}

template <typename T>
inline void foreach(const T* src, int w, int h, std::function<void(int, int, T)> f) {
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			f(x, y, src[y*w + x]);
		}
	}
}

template <typename T>
inline void printBuffer(const T* src, int w, int h) {
	std::function<void(int, int, float)> print = [](int x, int y, float val) {
		std::cout << x << ", " << y << ": " << val << std::endl;
	};
	foreach<T>(src, w, h, print);
}

template <typename T>
void fill(T* dest, int w, int h, const T value) {
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			const int idx = y * w + x;
			dest[idx] = value;
		}
	}
}

template <typename T>
inline void map_index(T* src, T* dest, int w, int h, std::function<T(int, int)> f) {
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			const int idx = y * w + x;
			dest[idx] = f(x, y);
		}
	}
}

template <typename T>
inline void fill_example(T* src, int w, int h) {
	auto create = [](int x, int y) { return (x + 1)*(y + 1); };
	map_index<T>(src, src, w, h, create);
}

template <typename T>
inline void par_map_index(T* src, T* dest, int w, int h, std::function<T(int, int)> f) {
#pragma omp parallel for
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			const int idx = y * w + x;
			dest[idx] = f(x, y);
		}
	}
}

template <typename T>
inline void par_fill_example(T* src, int w, int h) {
	auto create = [](int x, int y) { return (T(x) + 1)*(T(y) + 1); };
	par_map_index<T>(src, src, w, h, create);
}

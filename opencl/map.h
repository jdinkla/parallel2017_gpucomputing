#pragma once

#include <functional>

using namespace std;

template <typename T>
void map(T* src, T* dest, int w, int h, std::function<T(T)> f) {
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			const int idx = y * w + x;
			dest[idx] = f(src[idx]);
		}
	}
}


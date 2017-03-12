#pragma once

#include "Defs.h"
#include <functional>

template <typename T>
void par_fill(T* dest, int w, int h, const T value) {
#pragma omp parallel for
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			const int idx = y * w + x;
			dest[idx] = value;
		}
	}
}

template <typename T>
void par_map(T* src, T* dest, int w, int h, std::function<T(T)> f) {
#pragma omp parallel for
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			const int idx = y * w + x;
			dest[idx] = f(src[idx]);
		}
	}
}

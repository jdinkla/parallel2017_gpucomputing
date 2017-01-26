#pragma once

#include <functional>
#include <cuda_runtime.h>
#include "defaults_smooth.h"

template <typename T> __host__ __device__
T smooth(const T* src, int w, int h, int x, int y) {
	T sum = 0;
	int c = 0;
	for (int dy = -1; dy <= 1; dy++) {
		for (int dx = -1; dx <= 1; dx++) {
			const int rx = x + dx;
			const int ry = y + dy;
			if (0 <= rx && rx < w && 0 <= ry && ry < h) {
				sum += src[ry * w + rx];
				c++;
			}
		}
	}
	return T(sum / c);
}

template <typename T>
void smooth(const T* src, T* dest, int w, int h) {
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			dest[y * w + x] = smooth<T>(src, w, h, x, y);
		}
	}
}

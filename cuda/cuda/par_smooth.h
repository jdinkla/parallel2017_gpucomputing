#pragma once

#include "smooth.h"
#include "Defs.h"

template <typename T>
void par_smooth(const T* src, T* dest, int w, int h) {
#pragma omp parallel for
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			dest[y * w + x] = smooth<T>(src, w, h, x, y);
		}
	}
}

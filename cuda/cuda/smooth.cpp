#include "smooth.h"
#include "map.h"
#include <assert.h>
#include <iostream>
#include "Utilities.h"
#include "Timer.h"

void example_smooth() {
	cout << "example_smooth" << endl;

	const int w = smooth_tiny_w;
	const int h = smooth_tiny_h;
	const size_t sz = w * h * sizeof(float);

	// Allocate
	float* src = (float*)malloc(sz);
	float* dest = (float*)malloc(sz);

	fill_example(src, w, h);
	printBuffer(src, w, h);

	smooth(src, dest, w, h);
	printBuffer(dest, w, h);

	// Free
	free(src);
	free(dest);
}

void example_smooth_generic(const char* str, const int w, const int h) {
	const size_t sz = w * h * sizeof(float);
	auto t = new Timer();

	// Allocate
	float* src = (float*)malloc(sz);
	float* dest = (float*)malloc(sz);

	cout << "fill" << endl;
	t->start();
	par_fill_example(src, w, h);
	t->stop();
	cout << "duration of par_fill_example " << t->delta() << endl;

	t->reset();

	cout << "smooth" << endl;
	t->start();
	smooth(src, dest, w, h);
	t->stop();
	cout << "duration of " << str << " " << t->delta() << endl;

	// if dest[] is not used, it is not calculated
	cout << "needed on Mac OS X=" << dest[0] << endl;
	
	// Free
	free(src);
	free(dest);
}

void example_smooth_small() {
	example_smooth_generic("example_smooth_small", smooth_small_w, smooth_small_h);
}

void example_smooth_large() {
	example_smooth_generic("example_smooth_large", smooth_large_w, smooth_large_h);
}

void example_smooth_huge() {
	example_smooth_generic("example_smooth_huge", smooth_huge_w, smooth_huge_h);
}

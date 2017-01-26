#include "gpu_smooth.cuh"
#include "map.h"
#include <assert.h>
#include <iostream>
#include "Utilities.h"
#include "CudaUtilities.cuh"
#include "Timer.h"
#include "smooth.h"
#include <list>

using namespace std;

void example_gpu_smooth() {
	const int w = smooth_tiny_w;
	const int h = smooth_tiny_h;
	const size_t sz = w * h * sizeof(float);

	float* src = nullptr;
	cudaMallocManaged(&src, sz);

	float* dest = nullptr;
	cudaMallocManaged(&dest, sz);

	fill_example(src, w, h);
	printBuffer(src, w, h);

	gpu_smooth(src, dest, w, h);
	printBuffer(dest, w, h);

	cudaFree(src);
	cudaFree(dest);
}

void example_gpu_smooth_generic(const char* str, const int w, const int h) {
	const size_t sz = w * h * sizeof(float);
	auto t = new Timer();

	float* src = nullptr;
	cudaMallocManaged(&src, sz);
	check("cudaMallocManaged src");

	float* dest = nullptr;
	cudaMallocManaged(&dest, sz);
	check("cudaMallocManaged dest");

	cout << "fill" << endl;
	t->start();
	gpu_fill_example(src, w, h);
	t->stop();
	cout << "duration of gpu_fill_example " << t->delta() << endl;
	check("gpu_fill_example");

	cout << "smooth" << endl;
	t->start();
	gpu_smooth(src, dest, w, h);
	t->stop();
	cout << "duration of " << str << " " << t->delta() << endl;
	check("gpu_smooth");

	cudaFree(src);
	check("cudaFree src");
	cudaFree(dest);
	check("cudaFree dest");
}

void example_gpu_smooth_small() {
	example_gpu_smooth_generic("example_gpu_smooth_small", smooth_small_w, smooth_small_h);
}

void example_gpu_smooth_large() {
	example_gpu_smooth_generic("example_gpu_smooth_large", smooth_large_w, smooth_large_h);
}

void example_gpu_smooth_huge() {
	example_gpu_smooth_generic("example_gpu_smooth_huge", smooth_huge_w, smooth_huge_h);
}


void example_gpu_smooth_generic(std::list<dim3> blocks, int w, int h) {

	const size_t sz = w * h * sizeof(float);
	auto t = new Timer();

	float* src = nullptr;
	cudaMallocManaged(&src, sz);
	check("cudaMallocManaged src");

	float* dest = nullptr;
	cudaMallocManaged(&dest, sz);
	check("cudaMallocManaged dest");

	gpu_fill_example(src, w, h);
	check("gpu_fill_example");

	for (auto block : blocks) {
		cout << "(" << block.x << "," << block.y << "," << block.z << "): ";
		t->start();
		gpu_smooth(block, src, dest, w, h);
		t->stop();
		cout << t->delta() << endl;
		check("gpu_smooth");
	}

	cudaFree(src);
	check("cudaFree src");
	cudaFree(dest);
	check("cudaFree dest");
}

void bench_smooth_blocks() {
	int w = smooth_small_w;
	int h = smooth_small_h;
	std::list<dim3> ls {
		{128, 1, 1},
		{64, 2, 1},
		{32, 4, 1},
		{16, 8, 1},
		{8, 16, 1},
		{4, 32, 1},
		{2, 64, 1},
		{1, 128, 1}
	};
	example_gpu_smooth_generic(ls, w, h);
}

void bench_smooth_blocks2() {
	int w = smooth_large_w;
	int h = smooth_small_h;
	std::list<dim3> ls {
		{128, 1, 1},
		{64, 2, 1},
		{32, 4, 1},
		{16, 8, 1},
		{8, 16, 1},
		{4, 32, 1},
		{2, 64, 1},
		{1, 128, 1}
	};
	example_gpu_smooth_generic(ls, w, h);
}

#include "gpu_smooth.cuh"
#include "map.h"
#include <assert.h>
#include <iostream>
#include "Utilities.h"
#include "CudaUtilities.cuh"
#include "Timer.h"
#include "smooth.h"

void example_gpu_smooth_expl_generic(const char* str, const int w, const int h) {
	const size_t sz = w * h * sizeof(float);
	auto t = new Timer();

	cout << "Allocating " << sz/(1024*1024) << " megabytes for src" << endl;
	float* src = nullptr;
	cudaMalloc(&src, sz);
	check("cudaMalloc src");

	cout << "Allocating " << sz / (1024 * 1024) << " megabytes for dest" << endl;
	float* dest = nullptr;
	cudaMalloc(&dest, sz);
	check("cudaMalloc dest");

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

	cout << "copy d2h" << endl;
	t->start();
	float* destH = nullptr;
	cudaMallocHost(&destH, sz);
	check("cudaMallocHost destH");
	cudaMemcpy(destH, dest, sz, cudaMemcpyDeviceToHost);
	t->stop();
	cout << "duration of cudaMemcpy d2h " << t->delta() << endl;
	check("cudaMemcpy");

	cudaFree(src);
	check("cudaFree src");
	cudaFree(dest);
	check("cudaFree dest");
	cudaFreeHost(destH);
	check("cudaFreeHost destH");

}

void example_gpu_smooth_expl() {
	example_gpu_smooth_expl_generic("example_gpu_smooth_expl", smooth_tiny_w, smooth_tiny_h);
}

void example_gpu_smooth_expl_small() {
	example_gpu_smooth_expl_generic("example_gpu_smooth_expl_small", smooth_small_w, smooth_small_h);
}

void example_gpu_smooth_expl_large() {
	example_gpu_smooth_expl_generic("example_gpu_smooth_expl_large", smooth_large_w, smooth_large_h);
}

void example_gpu_smooth_expl_huge() {
	example_gpu_smooth_expl_generic("example_gpu_smooth_expl_huge", smooth_huge_w, smooth_huge_h);
}

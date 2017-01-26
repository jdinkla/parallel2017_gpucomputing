#include "cuda_runtime.h"
#include "Utilities.h"
#include <functional>
#include "gpu_map.cuh"
#include "par_map.h"
#include "CudaUtilities.cuh"
#include <assert.h>
#include <iostream>
#include "Timer.h"

using namespace std;

void example_gpu_map() {
	const int w = 100;
	const int h = 80;
	const size_t sz = w * h * sizeof(int);
	int* src = nullptr;
	cudaMallocManaged(&src, sz);
	int* dest = nullptr;
	cudaMallocManaged(&dest, sz);

	par_fill(src, w, h, 12);
	auto f = [] __host__ __device__ (int x) { return x + x; };
	gpu_map(src, dest, w, h, f);

	// dest should contain f(12) and not 12 anymore
	std::cout << "Val " << dest[0] << std::endl;
	assert(dest[0] == f(12));
	assert(dest[(w - 1)*(h - 1) - 1] == f(12));

	cudaFree(src);
	cudaFree(dest);
}

void example_gpu_map_large() {
	const int w = 10 * 1024;
	const int h = 80 * 1024;
	const size_t sz = w * h * sizeof(int);

	int* src = nullptr;
	cudaMallocManaged(&src, sz);
	int* dest = nullptr;
	cudaMallocManaged(&dest, sz);

	par_fill(src, w, h, 12);
	auto f = [] __host__ __device__(int x) { return x + x; };
	gpu_map(src, dest, w, h, f);

	// dest should contain f(12) and not 12 anymore
	std::cout << "Val " << dest[0] << std::endl;
	assert(dest[0] == f(12));
	assert(dest[(w - 1)*(h - 1) - 1] == f(12));

	cudaFree(src);
	cudaFree(dest);
}

void example_gpu_map_generic(dim3 block, int w, int h) {
	cout << "(" << block.x << "," << block.y << "," << block.z << "): ";
	const size_t sz = w * h * sizeof(int);

	auto t = new Timer();

	int* src = nullptr;
	cudaMallocManaged(&src, sz);
	int* dest = nullptr;
	cudaMallocManaged(&dest, sz);

	par_fill(src, w, h, 12);
	auto f = [] __host__ __device__(int x) { return x + x; };

	t->start();
	gpu_map(block, src, dest, w, h, f);
	t->stop();
	cout << "duration: " << t->delta() << endl;

	// if dest[] is not used, it is not calculated
	cerr << "needed on Mac OS X=" << dest[0] << endl;
	//assert(dest[0] != 12345);

	cudaFree(src);
	cudaFree(dest);
}

void bench_map_blocks() {
	int w = 1024 * 8;
	int h = 1024 * 8;
	example_gpu_map_generic(dim3(128, 1, 1), w, h);
	example_gpu_map_generic(dim3(32, 4, 1), w, h);
	example_gpu_map_generic(dim3(16, 8, 1), w, h);
	example_gpu_map_generic(dim3(8, 16, 1), w, h);
	example_gpu_map_generic(dim3(4, 32, 1), w, h);
	example_gpu_map_generic(dim3(1, 128, 1), w, h);
}

#pragma once

#include "cuda_runtime.h"
#include <stdexcept>
#include <sstream>
#include "Utilities.h"

// so blockIdx, blockDim and threadIdx are known
#include <device_launch_parameters.h>

inline void check_rc(cudaError_t rc, const char* msg)
{
	if (rc != cudaSuccess)
	{
		throw std::runtime_error(msg);
	}
}

inline void check(const char* msg)
{
	cudaError_t rc = cudaGetLastError();
	if (msg)
	{
		std::ostringstream buf;
		buf << msg << " - " << cudaGetErrorString(rc);
		check_rc(rc, buf.str().c_str());
	}
	else
	{
		check_rc(rc, cudaGetErrorString(rc));
	}
}

template <typename T, typename F>
__global__
void gpu_map_index_kernel(T* dest, int w, int h, F f) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < w && y < h) {
		dest[y * w + x] = f(x, y);
	}
}

template <typename T, typename F>
inline void gpu_map_index(T* dest, int w, int h, F f) {
	dim3 block(128, 1, 1);
	dim3 grid(ceiling_div(w, block.x),
		ceiling_div(h, block.y), 1);
	gpu_map_index_kernel<<<grid, block>>>(dest, w, h, f);
	cudaDeviceSynchronize();
}

template <typename T>
inline void gpu_fill_example(T* src, int w, int h) {
	auto create = [] __device__ (int x, int y) { return (T(x) + 1)*(T(y) + 1); };
	gpu_map_index<T>(src, w, h, create);

}

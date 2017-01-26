#pragma once

#include "cuda_runtime.h"
#include "Utilities.h"
#include "smooth.h"

template <typename T>
__global__
void gpu_smooth_kernel(const T* src, T* dest, int w, int h) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < w && y < h) {
		dest[y * w + x] = smooth<T>(src, w, h, x, y);
	}
}

template <typename T>
void gpu_smooth(const T* src, T* dest, int w, int h) {
	dim3 block(blockSizeX, blockSizeY, 1);
	dim3 grid(ceiling_div(w, block.x), ceiling_div(h, block.y), 1);
	gpu_smooth_kernel<<<grid, block>>>(src, dest, w, h);
	cudaDeviceSynchronize();
}

template <typename T>
void gpu_smooth(dim3 block, const T* src, T* dest, int w, int h) {
	dim3 grid(ceiling_div(w, block.x), ceiling_div(h, block.y), 1);
	gpu_smooth_kernel<<<grid, block>>>(src, dest, w, h);
	cudaDeviceSynchronize();
}

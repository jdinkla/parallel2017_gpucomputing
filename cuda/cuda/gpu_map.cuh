#include "cuda_runtime.h"
#include "Utilities.h"

template <typename T, typename F>
__global__
void gpu_map_kernel(T* src, T* dest, int w, int h, F f) {
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < w && y < h) {
		const int idx = y * w + x;
		dest[idx] = f(src[idx]);
	}
}

template <typename T, typename F>
void gpu_map(T* src, T* dest, int w, int h, F f) {
	dim3 block(64, 1, 1);
	dim3 grid(ceiling_div(w, block.x),
			  ceiling_div(h, block.y), 1);
	gpu_map_kernel<<<grid, block>>>(src, dest, w, h, f);
	cudaDeviceSynchronize();
}

template <typename T, typename F>
void gpu_map(dim3 block, T* src, T* dest, int w, int h, F f) {
	dim3 grid(ceiling_div(w, block.x), ceiling_div(h, block.y), 1);
	gpu_map_kernel<<<grid, block>>>(src, dest, w, h, f);
	cudaDeviceSynchronize();
}

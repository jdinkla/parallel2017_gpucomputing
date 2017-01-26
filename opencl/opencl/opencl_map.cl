
__kernel void opencl_mask(__global float* src, __global float* dest, int width, int height) {
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	if (x < width && y < height) {
		const int idx = y * width + x;
		//dest[idx] = f(src[idx]);
		dest[idx] = (src[idx]);
	}
}

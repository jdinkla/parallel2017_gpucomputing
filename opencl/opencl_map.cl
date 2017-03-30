
__kernel void opencl_map(__global float* src, __global float* dest, int width, int height) {
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	if (x < width && y < height) {
		const int i = y * width + x;
		dest[i] = 2*src[i];
	}
}

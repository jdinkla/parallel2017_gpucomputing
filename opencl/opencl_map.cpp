#include "Defs.h"
#include "par_map.h"
#include <string>
#ifdef MAC
#define __CL_ENABLE_EXCEPTIONS
#include "cl_1_2.hpp"
#else
#include <cl/cl.hpp>
#endif
#include "OpenCLUtilities.h"
#include "Utilities.h"
#include <iostream>
#include <assert.h>

using namespace std;

const int platform_id = 1;
const int device_id = 0;

void example_opencl_map_for_slide() {
	const int w = 100;
	const int h = 80;
	const size_t sz = w * h * sizeof(int);
	int* src = (int*)malloc(sz);
	int* dest = (int*)malloc(sz);
	par_fill(src, w, h, 12);


	cl::Platform platform = get_platforms()[platform_id];
	cl::Device device = get_devices(platform)[device_id];
	cl::Context context = get_context(platform, device);
	cl::CommandQueue queue = cl::CommandQueue(context, device, 
		CL_QUEUE_PROFILING_ENABLE);

	const string kernel_filename = "../opencl/opencl/opencl_map.cl";
	const string code = read_file(kernel_filename);
	cl::Program::Sources sources(1, { code.c_str(), code.length() + 1 });
	cl::Program program = cl::Program(context, sources);
	program.build({ device });

	cl::Buffer buf_src(context, 
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sz, src);
	cl::Buffer buf_dest(context, 
		CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sz, src);

	cl::Kernel opencl_map_kernel(program, "opencl_map");
	opencl_map_kernel.setArg(0, buf_src);
	opencl_map_kernel.setArg(1, buf_dest);
	opencl_map_kernel.setArg(2, w);
	opencl_map_kernel.setArg(3, h);

	queue.enqueueWriteBuffer(buf_src, false, 0, sz, src, NULL);
	queue.enqueueNDRangeKernel(
		opencl_map_kernel,
		cl::NullRange,
		cl::NDRange(w, h),
		cl::NullRange,
		NULL);
	queue.enqueueReadBuffer(buf_dest, false, 0, sz, dest, NULL);
	queue.finish();
}

void example_opencl_map() {
	const int w = 100;
	const int h = 80;
	const size_t sz = w * h * sizeof(int);
	const string kernel_filename = "../opencl/opencl/opencl_map.cl";

	// Allocate
	int* src = (int*)malloc(sz);
	int* dest = (int*)malloc(sz);

	par_fill(src, w, h, 12);

	cl::Platform platform = get_platforms()[platform_id];
	cout << "Using platform " << endl;
	info_platform(platform);

	cl::Device device = get_devices(platform)[device_id];
	cout << "Using device " << endl;
	info_device(device);

	cl::Context context = get_context(platform, device);
	cl::CommandQueue queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

	const string code = read_file(kernel_filename);
	cl::Program::Sources sources(1, { code.c_str(), code.length() + 1 });
	cl::Program program = cl::Program(context, sources);
	program.build({ device });

	cl::Buffer buf_src(context, 
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
		sz, src);

	cl::Buffer buf_dest(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sz, src);

	cl::Kernel opencl_map_kernel(program, "opencl_map");
	opencl_map_kernel.setArg(0, buf_src);
	opencl_map_kernel.setArg(1, buf_dest);
	opencl_map_kernel.setArg(2, w);
	opencl_map_kernel.setArg(3, h);

	queue.enqueueWriteBuffer(
		buf_src,
		false,
		0,
		sz,
		src,
		NULL);

	cl::Event event_kernel;

	queue.enqueueNDRangeKernel(
		opencl_map_kernel,
		cl::NullRange,
		cl::NDRange(w, h),
		cl::NullRange,
		NULL);

	queue.enqueueReadBuffer(
		buf_dest,
		false,
		0,
		sz,
		dest,
		NULL);

	queue.finish(); 
//	event_kernel.wait();

	// dest should contain 24s now
	// cout << "Val " << dest[0] << endl;
	assert(dest[0] == 24);
	assert(dest[(w - 1)*(h - 1) - 1] == 24);
}

void example_opencl_map_large() {
}



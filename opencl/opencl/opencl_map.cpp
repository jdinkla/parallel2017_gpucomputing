
#include <string>
#include <cl/cl.hpp>
#include "OpenCLUtilities.h"
#include "Utilities.h"

using namespace std;

void example_opencl_map() {

	const int platform_id = 1;
	const int device_id = 1;
	const string kernel_filename = "";

	cl::Platform platform = get_platforms()[platform_id];
	cl::Device device = get_devices(platform)[device_id];
	cl::Context context = get_context(platform, device);
	cl::CommandQueue queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

	const string code = read_file(kernel_filename);
	cl::Program::Sources sources(1, { code.c_str(), code.length() + 1 });
	cl::Program program = cl::Program(context, sources);
	program.build({ device });


}


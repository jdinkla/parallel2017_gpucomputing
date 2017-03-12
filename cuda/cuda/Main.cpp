#include <iostream>
#include "Defs.h"
#include <cuda_runtime_api.h>

using namespace std;

// map
extern void example_map();
extern void example_par_map();
extern void example_par_map_large();
extern void example_gpu_map();
extern void example_gpu_map_large();

// stencil
extern void example_smooth();
extern void example_smooth_small();
extern void example_smooth_large();
extern void example_smooth_huge();

extern void example_par_smooth();
extern void example_par_smooth_small();
extern void example_par_smooth_large();
extern void example_par_smooth_huge();

extern void example_gpu_smooth();
extern void example_gpu_smooth_small();
extern void example_gpu_smooth_large();
extern void example_gpu_smooth_huge();

extern void example_gpu_smooth_expl();
extern void example_gpu_smooth_expl_small();
extern void example_gpu_smooth_expl_large();
extern void example_gpu_smooth_expl_huge();

int main(int argc, char** argv) {
	int rc = 0;
	int alg = 0;

	if (argc == 2) {
		alg = atoi(argv[1]);
	}
	else {
		cerr << "ERROR: missing argument" << endl;
		return 1;
	}

	try {
		switch (alg)
		{

		// map
		case 1: example_map();
			break;
		case 2: example_par_map();
			break;
		case 3: example_par_map_large();
			break;
		case 4: example_gpu_map();
			break;
		case 5: example_gpu_map_large();
			break;

		// stencil
		case 10: example_smooth();
			break;
		case 11: example_smooth_small();
			break;
		case 12: example_smooth_large();
			break;
		case 13: example_smooth_huge();
			break;

			/*   */
		case 20: example_par_smooth();
			break;
		case 21: example_par_smooth_small();
			break;
		case 22: example_par_smooth_large();
			break;
		case 23: example_par_smooth_huge();
			break;

			/*   */
		case 30: example_gpu_smooth();
			break;
		case 31: example_gpu_smooth_small();
			break;
		case 32: example_gpu_smooth_large();
			break;
		case 33: example_gpu_smooth_huge();
			break;

			/*   */
		case 40: example_gpu_smooth_expl();
			break;
		case 41: example_gpu_smooth_expl_small();
			break;
		case 42: example_gpu_smooth_expl_large();
			break;
		case 43: example_gpu_smooth_expl_huge();
			break;

		default:
			cout << "Unknown id" << endl;
			rc = 1;
			break;
		}
	}
	catch (std::runtime_error& e) {
		cerr << "ERROR: " << e.what() << endl;
		cudaDeviceReset();
	}
	return rc;
}

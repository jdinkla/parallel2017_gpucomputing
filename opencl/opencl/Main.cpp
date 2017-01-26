#include <iostream>
#include "Defs.h"

using namespace std;

// map
extern void example_opencl_map();;

// stencil
//extern void example_opencl_smooth();

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
		case 1: example_opencl_map();
			break;

			// stencil
		case 10: example_opencl_map();
			break;

		default:
			cout << "Unknown id" << endl;
			rc = 1;
			break;
		}
	}
	catch (std::runtime_error& e) {
		cerr << "ERROR: " << e.what() << endl;
	}
	return rc;
}

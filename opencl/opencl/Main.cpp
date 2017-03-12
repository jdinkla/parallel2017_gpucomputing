#include <iostream>
#include "Defs.h"

using namespace std;


extern void info_platforms();
extern void info_devices();

// map
extern void example_map();
extern void example_par_map();
extern void example_par_map_large();
extern void example_opencl_map();;
extern void example_opencl_map_large();;

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

		case 1: info_platforms();
			break;

		case 2: info_devices();
			break;

		case 10: example_map();
			break;

		case 11: example_par_map();
			break;

		case 12: example_par_map_large();
			break;

		case 13: example_opencl_map();
			break;

		case 14: example_opencl_map_large();
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

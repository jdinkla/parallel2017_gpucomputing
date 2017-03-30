#include <iostream>
#include <assert.h>
#include "map.h"
#include "Utilities.h"

using namespace std;

void example_map() {
	const int w = 100;
	const int h = 80;
	const size_t sz = w * h * sizeof(int);

	// Allocate
	int* src = (int*)malloc(sz);
	int* dest = (int*)malloc(sz);

	fill(src, w, h, 12);

	std::function<int(int)> f = [](int x) { return x + x; };
	map(src, dest, w, h, f);

	// dest should contain f(12) and not 12 anymore
	assert(dest[0] == f(12));
	assert(dest[(w - 1)*(h - 1) - 1] == f(12));

	// cout << "Val " << dest[0] << endl;

	// Free
	free(src);
	free(dest);
}

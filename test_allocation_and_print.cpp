#define MODEL_LIB_H_IMPLEMENTATION
#include "model_lib.h"
//test_allocation_and_print.cpp
int main(int argc, char** argv)
{

	size_t topology[] = {2 , 1};
	model m;
	record r;

	allocate_model(&m, sizeof(topology) / sizeof(*topology), topology);
	
	randomize_model(m);
	
	allocate_record(&r, 2);
	r.content[0] = 1.0f;
	r.content[1] = 0.0f;

	print_record(r);
	print_model(m);

	return 0;
}
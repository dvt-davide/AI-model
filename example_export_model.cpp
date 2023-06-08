#define MODEL_LIB_H_IMPLEMENTATION
#include"model_lib.h"
//test_export_model.cpp
int main(int argc, char** argv)
{
	const char* usage = "Export model to file. Usage: \n"
						"XXX.exe <filename> <mode>\n"
						"	-<filename>: file path location\n"
						"	-<mode>: HR (Human readable) | MR (Machine readable)\n";

	const char* program = args_shift(&argc, &argv, usage);
	const char* fname   = args_shift(&argc, &argv, usage);
	const char* mode    = args_shift(&argc, &argv, usage);

	model m;
	size_t t[] = {2, 1};

	allocate_model(&m, 2, t);
	
	export_model(m, fname, mode);

	std::cout << "Export completed" << std::endl;

	return 0;
}
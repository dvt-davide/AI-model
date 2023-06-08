#define MODEL_LIB_H_IMPLEMENTATION
#include "model_lib.h"

int main(int argc, char** argv)
{
	const char* usage = "Import model from binary file. Usage: \n"
						"XXX.exe <filename>\n"
						"	-<filename>: .bin file path location\n";

	const char* program = args_shift(&argc, &argv, usage);
	const char* fname   = args_shift(&argc, &argv, usage);

	model m;

	import_model(&m, fname);

	print_model(m);

	return 0;
}

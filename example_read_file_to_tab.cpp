#define MODEL_LIB_H_IMPLEMENTATION 
#include"model_lib.h"
//test_read_file_to_tab.cpp
int main(int argc, char** argv)
{
	const char* usage = "Read file into table structure. Usage: \n"
						"XXX.exe <filename>\n";
	const char* program = args_shift(&argc, &argv, usage);
	const char* fname   = args_shift(&argc, &argv, usage);
	table tab;

	read_file_to_tab(&tab, fname, '\t');
	
	print_table(tab);

	return 0;
}
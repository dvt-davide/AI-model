#define MODEL_LIB_H_IMPLEMENTATION 
#define RATE  2.0f
#define BCKPRP 2.0f
#define OUTQ (n->o * (1.0f - n->o)) /*O1 * (1 - O1)*/ 
#include"model_lib.h"

int main(int argc, char** argv)
{
	const char* usage = "Read file and use it to train a model. Usage: \n"
						"XXX.exe <filename>\n";

	const char* program = args_shift(&argc, &argv, usage);
	const char* fname   = args_shift(&argc, &argv, usage);
	
    table tset;
	read_file_to_tab(&tset, fname, '\t');
    model m;
    size_t topology[] = { 3, 15, 5 };
    allocate_model(&m, (size_t)3, topology);
    randomize_model(m);
    
    // std::cout << "MODEL---------------------------------------------------------------\n"; 

    // print_model(m);

    for(size_t r = 0; r < tset.size; r++)
    {
        internalize_record(tset.content[r]);
    }

    // std::cout << "TRAINING------------------------------------------------------------\n";

    for(size_t epoch = 0; epoch < 100; epoch++)
    {
        // if(epoch % 100 == 0)
        //     std::cout << "Epoch " << (unsigned int) epoch << std::endl;

        for(size_t r = 0; r < tset.size; r++)
        {
            record input_row = { 3, tset.content[r].content };
            record answer_row = { 5, tset.content[r].content + 3 }; 
            backpropagation(m, input_row, answer_row);
        }
    }

    // std::cout << "TRAINED MODEL---------------------------------------------------\n"; 
    
    // print_model(m);
    
    // for(size_t r = 0; r < tset.size; r++)
    // {
    //     record input_row = { 3, tset.content[r].content };
    //     record output_row;
    //     feed(m, input_row);
    //     forward(m);
    //     guess(m, &output_row);
    //     externalize_record(output_row);
    //     print_record(output_row);
    //     delete[] output_row.content;
    // }

	return 0;
}
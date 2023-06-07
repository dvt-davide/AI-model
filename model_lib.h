#include<fstream>
#include<iostream>
#include<stdlib.h>
#include<math.h>
#include<cstdarg>
#include<cfloat>
#include<time.h>
#include<string>
#include<sstream>

/* #define MODEL_LIB_H_IMPLEMENTATION before include lib.h to include function implementations /
/* #define MODEL_LIB_H_SUPPRESS_CHECKS to disable security checks to improve performance */
/* #define MODEL_LIB_H_SUPPRESS_ERROR_MESSAGES to not print errors */

//*********Constants used in thelearning process**************************/
#ifndef RATE // Learning rate of the model
	#define RATE  0.1f
#endif
#ifndef BCKPRP // Constant used in the backpropagation formula
	#define BCKPRP 1.0f /* OR 2.0f */
#endif
#ifndef OUTQ // Used in the backpropagation formula
	#define OUTQ (n->o * n->o) /*O1 * (1 - O1); | 1 - O1 * O1 ;*/
#endif
/*************************************************************************/

typedef struct {
	size_t size;
	float* content;
} record;													        // Array of floats {0.0f, 0.0f, 0.0f, ...}
															        
typedef struct {                                                    // single Node of the neural network
	record is;                                                      // Values feeded into the node
	record ws;                                                      // Weights to apply to the inputs 
	float b;                                                        // Bias of the node
	float o;                                                        // Output signal of the node ( is[0]*ws[0] + is[1]*ws[1] + ...)
} node; 		                                                    
				                                                    
typedef struct {                                                    //Array of records
	size_t size;											        
	record* content;										        
} table;													        
															        
typedef struct {											        // Neural network model
	record topology;										        // How mant nodes per layer and how many layers { input_layer, hidden_layer1, ... hidden_layerX, output_layer}
	node* network;											        // List all nodes sorted by layer 
} model;													     
															     
/*Utils*/													     
const char* args_shift(int* argc, char*** argv, const char* usage);	// Read input arguments one by one 
void externalize_value(float* v);								    // Map value from -inf/+inf to a value between -1.0f and 1.0f
void internalize_value(float* v);								    // Map value from -1.0f and 1.0f to a value from -inf and +inf
/*Print*/													        
void print_record(record r);								        
void print_table(table t);									        
void print_node(node n);									        
void print_model(model m);									        
															        
/*Allocations*/												        
void allocate_record(record* r, size_t size);                       // Allocate 'size' elements in the structure
void allocate_tab(table* t, size_t rs, size_t cs);                  // Allocate 'rs' records with 'cs' elements 
void allocate_node(node* n, size_t size, record* shared);           // If shared = nullptr allocate 'size' elements for the inputs of the node if more nodes shares the same inputs we pass the same memory location to optimize memory management
void allocate_model(model* m, size_t size, size_t* topology);       // Allocate 'size' layers, and sum all elements of topology to define the total number of nodes
															        
/*Record*/													        
void internalize_record(record r);							        // Map all values from -inf/+inf to a value between -1.0f and 1.0f
void externalize_record(record r);							        // Map all vaues from -1.0f and 1.0f to a value from -inf and +inf
float sum_all(record r);									        // Sum all elements in the record content
float avg(record r);										        // Calculate average value of all elements in the record content
																    
/*Table*/														    
void read_file_to_tab(table* t, const char* fname, char fdelim);    // Read input file (numbers) into table strucuture
																    
/*Node*/														    
void forward(node* n);							                    // Evaluate output value of the node  ( is[0]*ws[0] + is[1]*ws[1] + ...)
void correct(node* n, float answer, record b_answer);               // Apply correction to weights and bias based on the correct answer Generate also the 'correct answer' to be used in the nodes in the previous layer
																    
/*Model*/														    
void feed(model m, record r);                                       // Feed input record to IS of the nodes in the input layer
void forward(model m);                                              // Evaluate output value of all nodes (call forward function passing every single node from the input layer to the output layer)
void guess(model m, record * r);							        // Get the values from the output layer and put them into the record structure
void backpropagation(model m, record input, record answer);	        // Backpropagation function: generates corrections to be applied to the nodes (using correct function)
void randomize_model(model m);								        // Randomize all weights of all nodes in the model (from 0.0f to 1.0f)
void export_model(model m, const char* fname, const char* mode);    // Export model to file (mode = HR -> write a human readable txt file) (mode = MR -> write a machine readable binary file)
void import_model(model *m, const char* fname);					    // Import model from .bin file (written in MR mode)

#ifdef MODEL_LIB_H_IMPLEMENTATION

void import_model(model* m, const char* fname)
{
	std::fstream file;
	size_t size = 0;
	float* topology_f;
	size_t* topology_t;

	file.open(fname, std::ios::binary | std::ios::in);

	if (!file.is_open())
	{
#ifndef MODEL_LIB_H_SUPPRESS_ERROR_MESSAGES
		std::cout << "ERROR:  Failed to open file " << fname << std::endl;
#endif
		exit(1);
	}

	file.read(reinterpret_cast<char*>(&size), sizeof(size));
#ifndef MODEL_LIB_H_SUPPRESS_CHECKS
	if (size == 0)
	{
#ifndef MODEL_LIB_H_SUPPRESS_ERROR_MESSAGES
		std::cout << "ERROR:  file " << fname << " corrupted" << std::endl;
#endif
		file.close();
		exit(1);
	}
#endif

	try {
		topology_f = new float[size];
		topology_t = new size_t[size];
	}
	catch (std::bad_alloc& exception)
	{
#ifndef MODEL_LIB_H_SUPPRESS_ERROR_MESSAGES
		std::cout << "ERROR: Cannot allocate topology: " << size << ": " << exception.what() << std::endl;
#endif
		file.close();
		exit(1);
	}

	for (size_t x = 0; x < size; x++)
	{
		file.read(reinterpret_cast<char*>(&topology_f[x]), sizeof(topology_f[x]));
		topology_t[x] = (size_t)topology_f[x];
	}

	allocate_model(m, size, topology_t);

	delete[] topology_f;
	delete[] topology_t;

	for (size_t x = 0; x < sum_all(m->topology); x++)
	{
		file.read(reinterpret_cast<char*>(&m->network[x].ws.size), sizeof(m->network[x].ws.size));
		for (size_t y = 0; y < m->network[x].ws.size; y++)
			file.read(reinterpret_cast<char*>(&m->network[x].ws.content[y]), sizeof(m->network[x].ws.content[y]));
		file.read(reinterpret_cast<char*>(&m->network[x].b), sizeof(m->network[x].b));
	}

	file.close();
}

void export_model(model m, const char* fname, const char* mode)
{
	std::fstream file;
	std::streambuf* coutbuf = std::cout.rdbuf(); //save old buf

	if (mode == "HR")
	{
		file.open(fname, std::ios::out);
		if (!file.is_open())
		{
#ifndef MODEL_LIB_H_SUPPRESS_ERROR_MESSAGES
			std::cout << "ERROR:  Failed to open file " << fname << std::endl;
#endif
			exit(1);
		}

		std::cout.rdbuf(file.rdbuf());//redirect std::cout to out.txt!
		print_model(m);
		std::cout.rdbuf(coutbuf); //reset to standard output again

		file.close();
	}
	else if (mode == "MR")
	{
		file.open(fname, std::ios::binary | std::ios::out);
		if (!file.is_open())
		{
#ifndef MODEL_LIB_H_SUPPRESS_ERROR_MESSAGES
			std::cout << "ERROR:  Failed to open file " << fname << std::endl;
#endif
			exit(1);
		}

		file.write(reinterpret_cast<const char*>(&m.topology.size), sizeof(m.topology.size));
		for (size_t x = 0; x < m.topology.size; x++)
			file.write(reinterpret_cast<const char*>(&m.topology.content[x]), sizeof(m.topology.content[x]));

		for (size_t x = 0; x < sum_all(m.topology); x++)
		{
			file.write(reinterpret_cast<const char*>(&m.network[x].ws.size), sizeof(m.network[x].ws.size));
			for (size_t y = 0; y < m.network[x].ws.size; y++)
				file.write(reinterpret_cast<const char*>(&m.network[x].ws.content[y]), sizeof(m.network[x].ws.content[y]));
			file.write(reinterpret_cast<const char*>(&m.network[x].b), sizeof(m.network[x].b));
		}

		file.close();
	}
#ifndef MODEL_LIB_H_SUPPRESS_CHECKS
	else
	{
		std::cout << "ERROR:  Mode " << mode << " not supported!!" << std::endl;
		exit(1);
	}
#endif
}

const char* args_shift(int* argc, char*** argv, const char* usage)
{
#ifndef MODEL_LIB_H_SUPPRESS_CHECKS
	if (*argc <= 0)
	{
		std::cout << usage << std::endl;
		exit(1);
	}
#endif
	char* result = **argv;
	(*argc) -= 1;
	(*argv) += 1;
	return result;
}

void randomize_model(model m)
{
	size_t size = (size_t)sum_all(m.topology);

	srand((unsigned int)time(0));

	for (size_t x = 0; x < size; x++)
		for (size_t wx = 0; wx < m.network[x].ws.size; wx++)
			m.network[x].ws.content[wx] = (float)rand() / (float)RAND_MAX;

}

void backpropagation(model m, record input, record answer)
{
	record output;
	record* curr_answer = nullptr;
	record* back_answer = nullptr;
	node* n = nullptr;
	size_t size = (size_t)sum_all(m.topology);
	curr_answer = &answer;
	size_t answer_idx = curr_answer->size - 1;

	feed(m, input);
	forward(m);
	guess(m, &output);
	externalize_record(output);
	internalize_record(answer);

#ifndef MODEL_LIB_H_SUPPRESS_CHECKS
	if (output.size != answer.size)
	{
		std::cout << "ERROR: backprop.:output incompatible with answer {" <<
			output.size <<
			"} answer size {" << answer.size << "}" << std::endl;
		exit(1);
	}
#endif
	for (size_t x = size; x > 0; x--)
	{
		back_answer = &m.network[x-1].is;
		correct(&m.network[x-1], curr_answer->content[answer_idx], *back_answer);
		if (answer_idx == 0) 
		{
			curr_answer = back_answer;      // at the change of the layer
			answer_idx = curr_answer->size; // at the change of the layer and when is < 0
		}
		else
		{
			answer_idx--;
		}
	}
}

void externalize_value(float* v)
{
	(*v) = (*v) * FLT_MAX;
}

void internalize_value(float* v)
{
	(*v) = (*v) / FLT_MAX;
}

void externalize_record(record r)
{
	for (size_t x = 0; x < r.size; x++)
	{
		externalize_value(&r.content[x]);
	}
}

void internalize_record(record r)
{
	for (size_t x = 0; x < r.size; x++)
	{
		internalize_value(&r.content[x]);
	}
}

void guess(model m, record* r)
{
	size_t outlayer = m.topology.size - 1;
	size_t offset   = (size_t) sum_all(m.topology) - (size_t)m.topology.content[outlayer];

	allocate_record(r, (size_t)m.topology.content[outlayer]);

	for (size_t x = 0; x < m.topology.content[outlayer]; x++)
	{
		r->content[x] = m.network[x + offset].o;
	}
}

void feed_next_layer(model m, record* r, size_t current_layer, size_t offset)
{
#ifndef MODEL_LIB_H_SUPPRESS_CHECKS
	if (r != nullptr)
	{
		if (m.topology.content[current_layer] != r->size)
		{
			std::cout << "ERROR: output cannot fit into middle layer, output size {" <<
				m.topology.content[current_layer] <<
				"} middle layer interface size {" << r->size << "}" << std::endl;
			exit(1);
		}
#endif
		for (size_t x = 0; x < m.topology.content[current_layer]; x++)
		{
			r->content[x] = m.network[x + offset].o;
		}
#ifndef MODEL_LIB_H_SUPPRESS_CHECKS
	}
#endif
}

void forward(model m)
{
	size_t offset = 0;
	size_t offset_next = (size_t)m.topology.content[0];
	size_t prev_l = 1;
	size_t network_size = (size_t)sum_all(m.topology);
	record* next_l = nullptr;
	size_t next_layer = 0;

	for (size_t t = 0; t < m.topology.size; t++)
	{
		next_layer = offset_next;
		next_l = nullptr;
		
		for (size_t x = 0; x < m.topology.content[t]; x++)
		{
			forward(&m.network[x+offset]);
			if (next_layer < network_size && x == 0)
				next_l = &m.network[next_layer].is;
		} 
		
		feed_next_layer(m, next_l, t, offset);
		offset += (size_t)m.topology.content[t];
		prev_l = (size_t)m.topology.content[t];

		if(t < m.topology.size -1)
			offset_next += (size_t)m.topology.content[t + 1];
	}

}

void feed(model m, record r)
{
#ifndef MODEL_LIB_H_SUPPRESS_CHECKS
	if (r.size != m.topology.content[0])
	{
		std::cout << "ERROR: inptut cannot fit into model, input size {" <<
			r.size <<
			"} netwok interface size {" << m.topology.content[0] << "}" << std::endl;
		exit(1);
	}
#endif
	for (size_t x = 0; x < r.size; x++) 
	{ // non si pu� usare il feed del nodo in questo caso
		if (m.network[x].is.size == 1)
			m.network[x].is.content[0] = r.content[x];
#ifndef MODEL_LIB_H_SUPPRESS_CHECKS
		else 
		{
			std::cout << "ERROR:  Feeded input does not fit into inptut node" << std::endl;
			exit(1);
		}
#endif
	}
}

void print_record(record r)
{
	std::cout.precision(10);
	std::cout << "    ~size(" << r.size << ")->{ ";

	for (size_t x = 0; x < r.size; x++)
		std::cout << std::fixed << r.content[x] << " ";

	std::cout << "}" << std::endl;
}

void print_node(node n) 
{
	std::cout << "    WS:";
	print_record(n.ws);
	std::cout << "    B:    (" << n.b <<")" << std::endl;
}

void print_model(model m)
{
	for (size_t x = 0; x < sum_all(m.topology); x++)
	{
		std::cout << "N" << x << "[" << std::endl;
		print_node(m.network[x]);
		std::cout << "]" << std::endl << std::endl;
	}

	std::cout << "Topology:";
	print_record(m.topology);
}

float sum_all(record r)
{
	float ret = 0.0f;

	for (size_t x = 0; x < r.size; x++)
		ret += r.content[x];

	return ret;
}

void allocate_model(model* m, size_t size, size_t* topology)
{
	size_t network_size = 0;
	size_t offset = 0;
	size_t prev_l = 1;
	record* share = nullptr;

	allocate_record(&m->topology, size);
	
	for (size_t x = 0; x < size; x++)
		m->topology.content[x] = (float)topology[x];
	
	network_size = (size_t)sum_all(m->topology);

	try {
		m->network = new node[network_size];
	}
	catch (std::bad_alloc& exception)
	{
#ifndef MODEL_LIB_H_SUPPRESS_ERROR_MESSAGES
		std::cout << "ERROR: Cannot allocate network of size: " << network_size << ": " << exception.what() << std::endl;
#endif
		exit(1);
	}

	for (size_t t = 0; t < m->topology.size; t++)
	{	
		share = nullptr;

		for (size_t x = 0; x < m->topology.content[t]; x++)
		{
			allocate_node(&m->network[x + offset], prev_l, share);
			if (t > 0)
				share = &m->network[x + offset].is;
		}

		offset += (size_t)m->topology.content[t];
		prev_l = (size_t)m->topology.content[t];
	}
}

void forward(node* n)
{
#ifndef MODEL_LIB_H_SUPPRESS_CHECKS
	if (n->is.size != n->ws.size)
	{
		std::cout << "ERROR: input size != weigths size: " << n->is.size << " != " << n->ws.size << std::endl;
		exit(1);
	}
#endif
	n->o = 0.0f;

	for (size_t x = 0; x < n->is.size; x++)
	{
		n->o += n->is.content[x] * n->ws.content[x];
	}
	n->o += n->b;

	internalize_value(&n->o);

	if (isnan(n->o))
		n->o = 0.0f;
	if (isinf(n->o) && n->o > 0.0f)
		n->o = 1.0f;
	if (isinf(n->o) && n->o < 0.0f)
		n->o = 0.0f;
}

void allocate_node(node* n, size_t size, record* shared = nullptr)
{
	if (shared == nullptr)
		allocate_record(&n->is, size);
	else
		n->is = *shared;
	
	allocate_record(&n->ws, size);

	n->b = 1.0f;
	n->o = 0.0f;
}

void correct(node* n, float answer, record b_answer)
{
#ifndef MODEL_LIB_H_SUPPRESS_CHECKS
	if (n->is.size != n->ws.size)
	{
		std::cout << "ERROR: input size != weigths size: " << n->is.size << " != " << n->ws.size << std::endl;
		exit(1);
	}

	if (n->is.size != b_answer.size)
	{
		std::cout << "ERROR: input size != b_answer size: " << n->is.size << " != " << b_answer.size << std::endl;
		exit(1);
	}
#endif
	float error = answer - n->o;
	float db = BCKPRP * BCKPRP * error * OUTQ * RATE;

	n->b += db;

	for (size_t x = 0; x < n->is.size; x++)
	{
		n->ws.content[x] += db * n->is.content[x];
		b_answer.content[x] = n->is.content[x] + (db * n->ws.content[x]);
	}
}

float avg(record r)
{
#ifndef MODEL_LIB_H_SUPPRESS_CHECKS
	if (r.size == 0)
	{
		std::cout << "ERROR: Cannot evaluate average: size = " << r.size  << std::endl;
		exit(1);
	}
#endif
	float ret = 0;
	for (size_t x = 0; x < r.size; x++)
		ret += r.content[x];
	return ret / r.size;
}

void allocate_record(record* r, size_t size)
{
	try {
		r->size = size;
		r->content = new float[size]();
	}
	catch (std::bad_alloc& exception)
	{
#ifndef MODEL_LIB_H_SUPPRESS_ERROR_MESSAGES
		std::cout << "ERROR: Cannot allocate record of size: " << size << ": " << exception.what() << std::endl;
#endif
		exit(1);
	}
}

void allocate_tab(table* t, size_t rs, size_t cs)
{
	t->size = rs;
	try {
		t->content = new record[t->size];
	}
	catch (std::bad_alloc& exception)
	{
#ifndef MODEL_LIB_H_SUPPRESS_ERROR_MESSAGES
		std::cout << "ERROR: Cannot allocate Table of size: " << t->size << ": " << exception.what() << std::endl;
#endif
		exit(1);
	}

	for (size_t x = 0; x < t->size; x++)
	{
		allocate_record(&t->content[x], cs);
	}
}

void print_table(table t)
{ 
	std::cout << "T" << t.size << "[" << std::endl;

	for (size_t x = 0; x < t.size; x++)
	{
		print_record(t.content[x]);
	}

	std::cout << "]" << std::endl << std::endl;
}

void read_file_to_tab(table* t, const char* fname, char fdelim)
{

	std::string buffer = "";
	std::fstream file;
	char x = ' ';
	std::istringstream ss;
	std::string line;
	size_t size = 0;
	size_t idx = 0;
	size_t cols_size = 0;
	size_t col = 0;
	std::string token;
	std::istringstream iss;

	file.open(fname, std::ios::in);

	if (file.is_open())
	{
		while (!file.eof())
		{	
			file.read(&x, 1);
			buffer.push_back(x);
		}
	}
#ifndef MODEL_LIB_H_SUPPRESS_ERROR_MESSAGES
	else
	{
		std::cout << "ERROR: file " << fname << " not found\n";
	}
#endif
	file.close();

	ss.str(std::string());
	ss.clear();
	ss.str(buffer);

	while (std::getline(ss, line))
	{
		size++;
	}

	ss.str(std::string());
	ss.clear();
	ss.str(buffer);

	while (std::getline(ss, line)) {

		iss.str(std::string());
		iss.clear();
		iss.str(line);
		
		if (idx == 0)
		{
			while (std::getline(iss, token, fdelim))
			{
				cols_size++;
			}

			allocate_tab(t, size, cols_size);

			iss.str(std::string());
			iss.clear();
			iss.str(line);
		}

		col = 0;

		while (std::getline(iss, token, fdelim))
		{
			t->content[idx].content[col] = std::stof(token);
			col++;
		}

		idx++;
	}
}
#endif
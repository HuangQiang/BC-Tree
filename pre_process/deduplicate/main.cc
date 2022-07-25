#include "headers.h"


// -----------------------------------------------------------------------------
int main(int nargs, char **args)
{
	srand(1);

	int    n           = atoi(args[1]);
	int    d           = atoi(args[2]);
	string data_set    = args[3];
	string output_path = args[4];

	char path[200]; strcpy(path, output_path.c_str());
	create_dir(path);

	printf("--------------------------------------------------------------\n");
	printf("n    = %d\n", n);
	printf("d    = %d\n", d);
	printf("data = %s\n", data_set.c_str());
	printf("\n");
	
	de_duplicate(n, d, data_set, output_path);

	// float *data = new float[(u64)n*d];
	// read_bin_data(n, d, data_set.c_str(), data);
	// for (int i = 0; i < 10; ++i) printf("%f ", data[i]);
	// printf("\n");
	// delete[] data;

	return 0;
}

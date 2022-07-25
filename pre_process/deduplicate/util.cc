#include "headers.h"

// -------------------------------------------------------------------------
int create_dir(						// create directory
	char *path)							// input path
{
	// ---------------------------------------------------------------------
	//  check whether the directory exists. if the directory does not 
	//  exist, we create the directory for each folder)
	// ---------------------------------------------------------------------
	int len = (int)strlen(path);
	for (int i = 0; i < len; ++i) {
		if (path[i] == '/') {
			char ch = path[i+1]; path[i + 1] = '\0';
			if (access(path, F_OK) != 0) {
				if (mkdir(path, 0755) != 0) {
					printf("Could not create directory %s\n", path); 
					return 1;
				}
			}
			path[i + 1] = ch;
		}
	}
	return 0;
}

// -----------------------------------------------------------------------------
int read_data(						// read data set from disk
	int    n,							// cardinality
	int    d,							// dimensionality
	string fname,						// file name of data set
	int    &min,						// min value of data set
	int    &max,						// max value of data set
	vector<vector<int> > &data) 		// data objects (return)
{
	FILE *fp = fopen(fname.c_str(), "r");
	if (!fp) { printf("Could not open %s.\n", fname.c_str()); return 1; }

	int i = 0, tmp = -1; min = MAXINT; max = MININT;
	while (!feof(fp) && i < n) {
		fscanf(fp, "%d", &tmp);
		for (int j = 0; j < d; ++j) {
			fscanf(fp, " %d", &tmp);

			data[i][j] = tmp;
			if (tmp < min) min = tmp;
			else if (tmp > max) max = tmp;
		}
		fscanf(fp, "\n");
		++i;
	}
	assert(feof(fp) && i == n);
	fclose(fp);

	return 0;
}

// -----------------------------------------------------------------------------
int read_bin_data(					// read data set from disk
	int    n,							// cardinality
	int    d,							// dimensionality
	string fname,						// file name of data set
	int    &min,						// min value of data set
	int    &max,						// max value of data set
	vector<vector<int> > &data) 		// data objects (return)
{
	FILE *fp = fopen(fname.c_str(), "rb");
	if (!fp) { printf("Could not open %s.\n", fname.c_str()); return 1; }

	int i = 0;
	int *tmp = new int[d];
	
	min = MAXINT; max = MININT;
	while (!feof(fp) && i < n) {
		fread(&tmp[0], sizeof(int), d, fp);
		for (int j = 0; j < d; ++j) {
			data[i][j] = tmp[j];
			if (tmp[j] < min) min = tmp[j];
			else if (tmp[j] > max) max = tmp[j];
		}
		if (i == 0 || i == n-1) {
			for (int j = 0; j < d; ++j) printf("%d ", tmp[j]);
			printf("\n\n");
		}
		++i;
	}
	printf("i=%d, n=%d, min=%d, max=%d\n", i, n, min, max);
	assert(i == n);
	fclose(fp);
	delete[] tmp;
	
	return 0;
}

// -----------------------------------------------------------------------------
int read_bin_data(                  // read bin data from disk
    int   n,                            // number of data points
    int   d,                            // data dimension
    const char *fname,                  // address of data set
    float *data)                        // data (return)
{
    // read bin data
    FILE *fp = fopen(fname, "rb");
    if (!fp) { printf("Could not open %s\n", fname); return 1; }
    int i = 0;
    while (!feof(fp) && i < n) {
        u64 shift = (u64)i*d;
        fread(&data[shift], sizeof(float), d, fp);
        ++i;
    }
    assert(i == n);
    fclose(fp);
    return 0;
}

// -----------------------------------------------------------------------------
u32 uniform_u32(                    // generate uniform unsigned r.v.
	u32 min,                            // min value
	u32 max)                            // max value
{
	u32 r = 0;
	if (RAND_MAX >= max - min) {
		r = min + (u32) floor((max-min+1.0f) * rand() / (RAND_MAX+1.0f));
	} 
	else {
		r = min + (u32) floor((max-min+1.0f) * 
			((u64) rand() * ((u64) RAND_MAX+1) + (u64) rand()) / 
			((u64) RAND_MAX * ((u64) RAND_MAX+1) + (u64) RAND_MAX + 1.0f));
	}
	assert(r >= min && r <= max);
	
	return r;
}
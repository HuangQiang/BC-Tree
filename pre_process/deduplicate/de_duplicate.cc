#include "headers.h"

// -----------------------------------------------------------------------------
void de_duplicate(					// de-duplicate data objects
	int    n,							// cardinality
	int    d,							// dimensionality
	string data_set,					// address of data set
	string output_path)					// output path with data name
{
	timeval start_time, end_time;

	// -------------------------------------------------------------------------
	//  step 1: read data set
	// -------------------------------------------------------------------------
	gettimeofday(&start_time, NULL);
	int min = MAXINT;
	int max = MININT;
	vector<vector<int> > data(n, vector<int>(d, 0));
	// read bin data
	if (read_bin_data(n, d, data_set, min, max, data)) {
		printf("Reading Dataset Error!\n"); exit(1);
	}
	// read txt data
	// if (read_data(n, d, data_set, min, max, data)) {
	// 	printf("Reading Dataset Error!\n"); exit(1);
	// }

	gettimeofday(&end_time, NULL);
	float read_file_time = end_time.tv_sec - start_time.tv_sec + 
		(end_time.tv_usec - start_time.tv_usec) / 1000000.0f;
	printf("Read Dataset: %f Seconds\n", read_file_time);

	// -------------------------------------------------------------------------
	//  step 2: perfect hashing to de-duplicate data objects
	// -------------------------------------------------------------------------
	gettimeofday(&start_time, NULL);
	vector<int> distinct_id;
	perfect_hashing(data, distinct_id);
	
	gettimeofday(&end_time, NULL);
	float hashing_time = end_time.tv_sec - start_time.tv_sec + 
		(end_time.tv_usec - start_time.tv_usec) / 1000000.0f;
	printf("Perfect Hashing: %f Seconds\n\n", hashing_time);

	// -------------------------------------------------------------------------
	//  step 3: write de-duplicated results to disk
	// -------------------------------------------------------------------------
	gettimeofday(&start_time, NULL);
	write_results(min, max, data, distinct_id, output_path);

	gettimeofday(&end_time, NULL);
	float io_time = end_time.tv_sec - start_time.tv_sec + 
		(end_time.tv_usec - start_time.tv_usec) / 1000000.0f;
	printf("Write Results: %f Seconds\n\n", io_time);
}

// -----------------------------------------------------------------------------
void write_results(					// write de-duplicated results to disk
	int    min, 						// min value of data objects
	int    max, 						// max value of data objects
	const  vector<vector<int> > &data, 	// data objects
	const  vector<int> &distinct_id, 	// distinct data objects id
	string output_path)					// output path with data name
{
	FILE *fp = NULL;
	int N = (int) data.size();
	int n = (int) distinct_id.size();
	int d = (int) data[0].size();

	printf("n   = %d\n", n);
	printf("d   = %d\n", d);
	printf("min = %d\n", min);
	printf("max = %d\n", max);
	printf("\n");
	
	// -------------------------------------------------------------------------
	//  write statistics of new data set to disk
	// -------------------------------------------------------------------------
	string stat_fname = output_path + ".stat";
	fp = fopen(stat_fname.c_str(), "w");
	if (!fp) { printf("Could not open %s\n", stat_fname.c_str()); exit(1); }
	
	fprintf(fp, "n   = %d\n", n);
	fprintf(fp, "d   = %d\n", d);
	fprintf(fp, "min = %d\n", min);
	fprintf(fp, "max = %d\n", max);
	fclose(fp);

	// -------------------------------------------------------------------------
	//  write original data objects id of new data set to disk
	// -------------------------------------------------------------------------
	string id_fname = output_path + ".id";
	fp = fopen(id_fname.c_str(), "w");
	if (!fp) { printf("Could not open %s\n", id_fname.c_str()); exit(1); }
	
	for (int i = 0; i < n; ++i) {
		int id = distinct_id[i]; assert(id >= 0 && id < N);
		fprintf(fp, "%d\n", id + 1);
	}
	fclose(fp);

	// -------------------------------------------------------------------------
	//  write new data set to disk
	// -------------------------------------------------------------------------
	string data_fname = output_path + ".bin";
	fp = fopen(data_fname.c_str(), "wb");
	if (!fp) { printf("Could not open %s\n", data_fname.c_str()); exit(1); }
	
	float *record = new float[d];
	for (int i = 0; i < n; ++i) {
		int id = distinct_id[i];
		for (int j = 0; j < d; ++j) record[j] = (float) data[id][j];

		fwrite(record, sizeof(float), d, fp);
	}
	fclose(fp);
	delete[] record;
}

// -----------------------------------------------------------------------------
void perfect_hashing(				// perfect hashing
	const vector<vector<int> > &data, 	// data objects
	vector<int> &distinct_id)			// distinct data objects id (return)
{
	int n = (int) data.size();		// number of data objects
	int d = (int) data[0].size();	// dimensionality
	int N = n * 10;					// number of buckets

	vector<u32> a_arr(d, 0);
	u32 b = 0;

	// -------------------------------------------------------------------------
	//  first level universal hashing
	// -------------------------------------------------------------------------
	umap table;
	gen_universal_hash_func(a_arr, b);
	
	for (int i = 0; i < n; ++i) {
		int bid = calc_hash_value(N, b, a_arr, data[i]);
		table[bid].push_back(i);
	}

	// -------------------------------------------------------------------------
	//  second level universal hashing
	// -------------------------------------------------------------------------
	distinct_id.clear();
	for (umap::iterator iter = table.begin(); iter != table.end(); ++iter) {
		const vector<int> &id_list = iter->second;
		int sub_n = (int) id_list.size(); // number of data objects in one bucket
		
		if (sub_n == 1) {
			distinct_id.push_back(id_list[0]);
		}
		else {
			umap sub_table;
			gen_universal_hash_func(a_arr, b);
			
			for (int j = 0; j < sub_n; ++j) {
				int id  = id_list[j];
				int bid = calc_hash_value(N, b, a_arr, data[id]);
				sub_table[bid].push_back(id);
			}

			add_distinct_id(sub_table, distinct_id);
		}
	}

	// -------------------------------------------------------------------------
	//  sort distinct data objects id in ascending order
	// -------------------------------------------------------------------------
	sort(distinct_id.begin(), distinct_id.end());
}

// -----------------------------------------------------------------------------
void gen_universal_hash_func(		// generate universal hash function 
	vector<u32> &a_arr,					// an array of multiplicator
	u32 &b)								// shift value
{
	int size = (int) a_arr.size();
	for (int i = 0; i < size; ++i) {
		a_arr[i] = uniform_u32(1, PRIME - 1);
	}

	b = uniform_u32(0, PRIME - 1);
}

// -----------------------------------------------------------------------------
int calc_hash_value(				// calculate hash value
	int n,								// number of buckets
	u32 b,								// shift value (hash func)
	const vector<u32> &a_arr,			// an array of multiplicator (hash func)
	const vector<int> &data)			// input data object
{
	int d = (int) data.size();
	int ret = 0;
	u64 h = 0;
	u32 TWO_TO_32_MINUS_1 = 4294967295U; // 2^32 - 1

	// -------------------------------------------------------------------------
	//  compute h = (a_arr * data) % PRIME
	// -------------------------------------------------------------------------
	for (int i = 0; i < d; ++i) {
		h = h + (u64) data[i] * (u64) a_arr[i];
		
		// ---------------------------------------------------------------------
		//  h & TWO_TO_32_MINUS_1 = low-32-bit of h
		//  h >> 32 = high-32-bit of h
		// ---------------------------------------------------------------------
		h = (h & TWO_TO_32_MINUS_1) + 5 * (h >> 32);

		// ---------------------------------------------------------------------
		//  fast compute "mod" function
		// ---------------------------------------------------------------------
		if (h >= PRIME) h = h - PRIME;
	}
	// -------------------------------------------------------------------------
	//  compute h = (a_arr * data + b) % PRIME
	// -------------------------------------------------------------------------
	h = h + (u64) b;
	h = (h & TWO_TO_32_MINUS_1) + 5 * (h >> 32);
	if (h >= PRIME) h = h - PRIME;

	// -------------------------------------------------------------------------
	//  compute h = ((a_arr * data + b) % PRIME) % n
	// -------------------------------------------------------------------------
	ret = (int) (h % (u32) 120001037); // n

	return ret;
}

// -----------------------------------------------------------------------------
void add_distinct_id(				// add distinct data id from hash table
	umap &table, 					// input hash table
	vector<int> &distinct_id)			// distinct data objects id (return)
{
	for (umap::iterator iter = table.begin(); iter != table.end(); ++iter) {
		const vector<int> &id_list = iter->second;
		int size = (int) id_list.size();

		if (size == 1) {
			distinct_id.push_back(id_list[0]);
		}
		else {
			int min = id_list[0];
			for (int j = 1; j < size; ++j) {
				if (id_list[j] < min) min = id_list[j];
			}
			distinct_id.push_back(min);
		}
	}
}

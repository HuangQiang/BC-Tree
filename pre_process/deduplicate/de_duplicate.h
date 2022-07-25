#ifndef __DE_DUPLICATE_H
#define __DE_DUPLICATE_H

struct Pair;

// -----------------------------------------------------------------------------
void de_duplicate(					// de-duplicate data objects
	int    n,							// cardinality
	int    d,							// dimensionality
	string data_set,					// address of data set
	string output_path);				// output path with data name

// -----------------------------------------------------------------------------
void write_results(					// write de-duplicated results to disk
	int    min, 						// min value of data objects
	int    max, 						// max value of data objects
	const  vector<vector<int> > &data,	// data objects
	const  vector<int> &distinct_id, 	// distinct data objects id
	string output_path);				// output path with data name
	
// -----------------------------------------------------------------------------
void perfect_hashing(				// perfect hashing
	const vector<vector<int> > &data, 	// data objects
	vector<int> &distinct_id);			// distinct data objects id (return)

// -----------------------------------------------------------------------------
void gen_universal_hash_func(		// generate universal hash function 
	vector<u32> &a_arr,					// an array of multiplicator
	u32 &b);							// shift value

// -----------------------------------------------------------------------------
int calc_hash_value(				// calculate hash value
	int n,								// number of buckets
	u32 b,								// shift value (hash func)
	const vector<u32> &a_arr,			// an array of multiplicator (hash func)
	const vector<int> &data);			// input data object

// -----------------------------------------------------------------------------
void add_distinct_id(				// add distinct data id from hash table
	umap &table, 						// input hash table
	vector<int> &distinct_id);			// distinct data objects id (return)

#endif // __DE_DUPLICATE_H
#ifndef __DEF_H
#define __DEF_H

#include <unordered_map>
using namespace std;

// -----------------------------------------------------------------------------
//  typedef
// -----------------------------------------------------------------------------
typedef unsigned int u32;
typedef unsigned long long u64; 
typedef unordered_map<int, vector<int> > umap; 

// -----------------------------------------------------------------------------
//  macros
// -----------------------------------------------------------------------------
#define MIN(a, b)	(((a) < (b)) ? (a) : (b))
#define MAX(a, b)	(((a) > (b)) ? (a) : (b))
#define POW(x)		((x) * (x))
#define SUM(x, y)	((x) + (y))
#define DIFF(x, y)	((y) - (x))
#define SWAP(x, y)	{int tmp=x; x=y; y=tmp;}

#define SEEK_SET 0
#define SEEK_CUR 1
#define SEEK_END 2

// -----------------------------------------------------------------------------
//  constants
// -----------------------------------------------------------------------------
const float MAXREAL    = 3.402823466e+38F;
const float MINREAL    = -MAXREAL;
const int   MAXINT     = 2147483647;
const int   MININT     = -MAXINT;

const int   SIZEBOOL   = (int) sizeof(bool);
const int   SIZEINT    = (int) sizeof(int);
const int   SIZECHAR   = (int) sizeof(char);
const int   SIZEFLOAT  = (int) sizeof(float);
const int   SIZEDOUBLE = (int) sizeof(double);

const int   SCALE      = 1000000;
const u32   PRIME      = 4294967291U; // PRIME = 2^32 - 5
const float E          = 2.7182818F;
const float PI         = 3.141592654F;
const float FLOATZERO  = 1e-6F;

#endif // __DEF_H
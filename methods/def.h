#pragma once

#include <iostream>
#include <cstdlib>
#include <vector>
#include <stdint.h>

namespace p2h {

// -----------------------------------------------------------------------------
//  macros
// -----------------------------------------------------------------------------
#define MIN(a, b)                   (((a) < (b)) ? (a) : (b))
#define MAX(a, b)                   (((a) > (b)) ? (a) : (b))
#define SQR(x)                      ((x) * (x))
#define SUM(x, y)                   ((x) + (y))
#define DIFF(x, y)                  ((y) - (x))
#define SWAP(x, y)                  {int tmp=x; x=y; y=tmp;}

// -----------------------------------------------------------------------------
//  typedef
// -----------------------------------------------------------------------------
typedef uint8_t  u08;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef float    f32;
typedef double   f64;

// -----------------------------------------------------------------------------
//  general constants
// -----------------------------------------------------------------------------
const f32 E                  = 2.7182818F;
const f32 PI                 = 3.141592654F;
const f32 CHECK_ERROR        = 1e-6F;

const f32 MAXREAL            = 3.402823466e+38F;
const f32 MINREAL            = -MAXREAL;
const int MAXINT             = 2147483647;
const int MININT             = -MAXINT;

const int RANDOM_SEED        = 6;
const int M                  = 100;   // statistics
const int SCAN_SIZE          = 64;    // RQALSH and QALSH
const int N_PTS              = 100;   // threshold for indexing (FH)
const f32 LEAF_RATIO         = 0.005; // ratio     for indexing (metric-tree)
const int MIN_LEAF_PTS       = 1000;  // threshold for indexing (metric-tree)
const int MAX_BLOCK_NUM      = 1000000; // FH (25,000 for 1,000,000; 1,000,000 for 100,000,000) 
const std::vector<f32> Cs    = { 10.0f };
const std::vector<int> Ls    = { 1,2,3 };
const std::vector<int> TOPKs = { 1, 10, }; // { 1,10,20,40 }; { 1,10,20,40,60,80,100 };

const int MAXK               = 100; // TOPKs.back(); 
const int THREAD_NUM         = 50;

} // end namespace p2h

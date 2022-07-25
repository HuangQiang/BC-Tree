#pragma once

#include <iostream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

#include "def.h"
#include "util.h"
#include "pri_queue.h"

namespace p2h {

// -----------------------------------------------------------------------------
//  Query-Aware Locality-Sensitive Hashing (QALSH) is used to solve the problem 
//  of c-Approximate Nearest Neighbor (c-ANN) search. 
//  
//  Qiang Huang, Jianlin Feng, Yikai Zhang, Qiong Fang, and Wilfred Ng. 
//  Query-aware locality-sensitive hashing for approximate nearest neighbor 
//  search, Proceedings of the VLDB Endowment (PVLDB), 9(1): 1–12, 2015.
// -----------------------------------------------------------------------------
class QALSH {
public:
    int    n_;                      // cardinality
    int    dim_;                    // dimensionality
    int    m_;                      // number of hash tables
    float  *a_;                     // hash functions
    Result *tables_;                // hash tables

    // assistant parameters for fast k-NN search
    int   *freq_;                   // collision frequency for n data points
    bool  *checked_;                // checked or not for n data points
    bool  *flag_;                   // flag for m hash tables
    int   *l_pos_;                  // left  positions for m hash tables
    int   *r_pos_;                  // right positions for m hash tables
    float *q_val_;                  // m hash values of query

    // -------------------------------------------------------------------------
    QALSH(                          // constructor
        int n,                          // cardinality
        int d,                          // dimensionality
        int m);                         // number hash tables

    // -------------------------------------------------------------------------
    ~QALSH();                       // destructor

    // -------------------------------------------------------------------------
    u64 get_memory_usage() {        // get memory usage
        u64 ret = 0UL;
        ret += sizeof(*this);
        ret += sizeof(float)*m_*dim_;     // a_
        ret += sizeof(Result)*(u64)m_*n_; // tables_
        return ret;
    }

    // -------------------------------------------------------------------------
    float calc_hash_value(          // calc hash value
        int   d,                        // dimension for calc hash value
        int   tid,                      // hash table id
        const float *data);             // one data/query object
    
    // -------------------------------------------------------------------------
    float calc_hash_value(          // calc hash value
        int   d,                        // dimension for calc hash value
        int   tid,                      // hash table id
        const Result *data);            // sample data

    // -------------------------------------------------------------------------
    int nns(                        // nearest neighbor search
        int   l,                        // collision threshold
        int   cand,                     // number of candidates
        const float *query,             // query object
        std::vector<int> &cand_list);   // candidates (return)

    // -------------------------------------------------------------------------
    int nns(                        // nearest neighbor search
        int   l,                        // collision threshold
        int   cand,                     // number of candidates
        int   sample_dim,               // sample dimension
        const Result *query,            // query object
        std::vector<int> &cand_list);   // candidates (return)

    // -------------------------------------------------------------------------
    void init(                      // init assistant parameters
        const float *query);            // query object
    
    // -------------------------------------------------------------------------
    void init(                      // init assistant parameters
        int   sample_dim,               // sample dimension
        const Result *query);           // query object
    
    // -------------------------------------------------------------------------
    void alloc();                   // alloc space for assistant parameters
    
    void init_position();           // init left/right positions

    // -------------------------------------------------------------------------
    int dynamic_collsion_counting(  // dynamic collision counting
        int   l,                        // collision threshold
        int   cand,                     // number of candidates
        std::vector<int> &cand_list);   // candidates (return)

    // -------------------------------------------------------------------------
    void free();                    // free space for assistant parameters
};

} // end namespace p2h

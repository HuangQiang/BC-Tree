#pragma once

#include <cstdint>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <vector>

#include "def.h"
#include "util.h"
#include "pri_queue.h"

namespace p2h {

// -----------------------------------------------------------------------------
//  RQALSH: Reverse Query-Aware Locality-Sensitive Hashing for High-Dimensional 
//  Furthest Neighbor Search
//
//  Qiang Huang, Jianlin Feng, Qiong Fang, Wilfred Ng. Two Efficient Hashing 
//  Schemes for High-Dimensional Furthest Neighbor Search. IEEE Transactions 
//  on Knowledge and Data Engineering (TKDE) 29 (12), 2772 - 2785, 2017.
// -----------------------------------------------------------------------------
class RQALSH {
public:
    int   n_;                       // number of input data
    int   dim_;                     // dimension of input data
    int   m_;                       // #hash tables
    const int *index_;              // index of input data
    float *a_;                      // hash functions
    Result *tables_;                // hash tables

    // assistant parameters for fast k-NN search
    int   *freq_;                   // separation frequency for n data points
    bool  *checked_;                // checked or not for n data points
    bool  *b_flag_;                 // flag for m hash tables
    bool  *r_flag_;                 // flag for m hash tables
    int   *l_pos_;                  // left  positions for m hash tables
    int   *r_pos_;                  // right positions for m hash tables
    float *q_val_;                  // m hash values of query

    // -------------------------------------------------------------------------
    RQALSH(                         // constructor
        int   n,                        // number of input data
        int   d,                        // dimension of input data
        int   m,                        // #hash tables
        const int *index);              // index of input data

    // -------------------------------------------------------------------------
    ~RQALSH();                      // destructor

    // -------------------------------------------------------------------------
    u64 get_memory_usage() {        // get memory usage
        u64 ret = 0UL;
        ret += sizeof(*this);
        if (n_ > N_PTS) {
            ret += sizeof(float)*m_*dim_;     // a_
            ret += sizeof(Result)*(u64)m_*n_; // tables_
        }
        return ret;
    }

    // -------------------------------------------------------------------------
    float calc_hash_value(          // calc hash value
        int   d,                        // dimension for calc hash value
        int   tid,                      // hash table id
        const float *data);             // one data object o'

    // -------------------------------------------------------------------------
    float calc_hash_value(          // calc hash value
        int   d,                        // dimension for calc hash value
        int   tid,                      // hash table id
        float last,                     // the last coordinate of input data
        const Result *data);            // sample data

    // -------------------------------------------------------------------------
    int fns(                        // furthest neighbor search
        int   l,                        // separation threshold
        int   cand,                     // number of candidates
        float R,                        // limited search range
        const float *query,             // query object
        std::vector<int> &cand_list);   // candidates (return)

    // -------------------------------------------------------------------------
    int fns(                        // furthest neighbor search
        int   l,                        // separation threshold
        int   cand,                     // number of candidates
        float R,                        // limited search range
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

    // -------------------------------------------------------------------------
    int dynamic_separation_counting(// dynamic separation counting
        int   l,                        // separation threshold
        int   cand,                     // number of candidates
        float R,                        // limited search range
        std::vector<int> &cand_list);   // candidates (return)

    // -------------------------------------------------------------------------
    float find_radius(              // find proper radius
        float w);                       // grid width

    // -------------------------------------------------------------------------
    void free();                    // free space for assistant parameters
};

} // end namespace p2h

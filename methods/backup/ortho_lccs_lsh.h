#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>
#include "../lccs_bucket/bucketAlg/lcs_int.h"

#include "def.h"
#include "util.h"
#include "pri_queue.h"
#include "kl_bucketing.h"

namespace p2h {

// -----------------------------------------------------------------------------
//  Ortho-LCCS-LSH: Locality-Sensitive Hashing based on 2D Plane Projection
//  
//  1. Project all data points into m 2D-Planes
//  2. Use LCCS-LSH for Point-to-Hyperplane Nearest Neighbor Search 
// -----------------------------------------------------------------------------
template<class DType>
class Ortho_LCCS_LSH {
public:
    using LCCS = mylccs::LCCS_SORT_INT;
    using SigType = int32_t;
    
    // -------------------------------------------------------------------------
    Ortho_LCCS_LSH(                 // constructor
        int   n,                        // number of data points
        int   d,                        // dimension of data points
        int   m,                        // number of random 2-dim ortho planes
        float w,                        // bucket width
        const DType *data);             // input data
    
    // -------------------------------------------------------------------------	
    ~Ortho_LCCS_LSH();              // destructor
    
    // -------------------------------------------------------------------------
    void display();                 // display parameters
    
    // -------------------------------------------------------------------------
    int nns(                        // point-to-hyperplane nns
        int   top_k,                    // top-k value
        int   cand,                     // number of candidates
        const float *query,             // input query
        MinK_List *list);               // top-k results (return)
    
    // -------------------------------------------------------------------------
    u64 get_memory_usage() {        // get memory usage
        u64 ret = 0UL;
        ret += sizeof(*this);
        ret += sizeof(float)*m_*(dim_*2+1);   // a1_, a2_, b_
        ret += bucketerp_.get_memory_usage(); // bucketerp_
        ret += data_sigs_.get_memory_usage(); // data_sigs_
        return ret;
    }

protected:
    int   n_;                       // number of data points
    int   dim_;                     // dimension of data points
    int   m_;                       // number of 2-dim ortho planes
    float w_;                       // bucket width
    const DType *data_;             // original data points
    
    float *a1_;                     // lsh func, O(m*d)
    float *a2_;                     // lsh func, O(m*d)
    float *b_;                      // random shift b, O(m)
    LCCS bucketerp_;                // hash tables
    NDArray<2, int32_t> data_sigs_; // data signatures

    // -------------------------------------------------------------------------
    void gen_ortho_proj(            // generate orthogonal random projections
        float *a1,                      // 1st random projection (return)
        float *a2);                     // 2nd random projection (return)
    
    // -------------------------------------------------------------------------
    float calc_data_hash(           // calc hash value for data point
        int   tid,                      // hash table id
        const DType *data);             // input data point

    // -------------------------------------------------------------------------
    float calc_query_hash(          // calc hash value for query
        int   tid,                      // hash table id
        const float *query);            // input query
};

// -----------------------------------------------------------------------------
template<class DType>
Ortho_LCCS_LSH<DType>::Ortho_LCCS_LSH(// constructor
    int   n,                            // number of data points
    int   d,                            // dimension of data points
    int   m,                            // number of random 2-dim ortho planes
    float w,                            // bucket width
    const DType *data)                  // input data
    : n_(n), dim_(d), m_(m), w_(w), data_(data), bucketerp_(m, 1)
{
    // init hash functions: a1_, a2_, b_
    int size = m*d;
    a1_ = new float[size];
    a2_ = new float[size];
    b_  = new float[m];
    for (int i = 0; i < m; ++i) {
        gen_ortho_proj(a1_+i*d, a2_+i*d);
        b_[i] = uniform(0, w);
    }
    
    // build hash tables for the hash values of data points
    data_sigs_.resize({n, m});
    SigType **data_sigs_ptr = data_sigs_.to_ptr();
    for (int i = 0; i < n; ++i) {
        const DType *point = data + (u64) i*d;
        
        for (int j = 0; j < m; ++j) {
            float val = calc_data_hash(j, point);
            data_sigs_ptr[i][j] = SigType((val + b_[j]) / w_);
        }
    }
    bucketerp_.build(data_sigs_);
}

// -----------------------------------------------------------------------------
template<class DType>
void Ortho_LCCS_LSH<DType>::gen_ortho_proj(// generate orthogonal random projs
    float *a1,                          // 1st random projection (return)
    float *a2)                          // 2nd random projection (return)
{
    int   last = dim_-1;
    float v1, v2, ip = 0.0f;
    for (int j = 0; j < last; ++j) {
        v1 = gaussian(0.0f, 1.0f);
        v2 = gaussian(0.0f, 1.0f);
        a1[j] = v1; a2[j] = v2; ip += v1*v2;
    }
    a1[last] = gaussian(0.0f, 1.0f);
    a2[last] = -ip / a1[last];
}

// -----------------------------------------------------------------------------
template<class DType>
float Ortho_LCCS_LSH<DType>::calc_data_hash(// calc hash value for data point
    int   tid,                          // hash table id
    const DType *data)                  // input data point
{
    int shift = tid * dim_;
    float p1 = calc_inner_product2<DType>(dim_, data, a1_+shift);
    float p2 = calc_inner_product2<DType>(dim_, data, a2_+shift);

    return p1 / p2;
}

// -----------------------------------------------------------------------------
template<class DType>
Ortho_LCCS_LSH<DType>::~Ortho_LCCS_LSH()// destructor
{
    delete[] a1_;
    delete[] a2_;
    delete[] b_;
}

// -----------------------------------------------------------------------------
template<class DType>
void Ortho_LCCS_LSH<DType>::display()// display parameters
{
    printf("Parameters of Ortho_LCCS_LSH:\n");
    printf("n = %d\n", n_);
    printf("d = %d\n", dim_);
    printf("m = %d\n", m_);
    printf("w = %f\n", w_);
    printf("\n");
}

// -----------------------------------------------------------------------------
template<class DType>
int Ortho_LCCS_LSH<DType>::nns(     // point-to-hyperplane nns
    int   top_k,                        // top-k value
    int   cand,                         // number of candidates
    const float *query,                 // input query
    MinK_List *list)                    // top-k results (return) 
{
    // calc the hash values of query
    std::vector<SigType> sigs(m_);
    for (int i = 0; i < m_; ++i) {
        float val = calc_query_hash(i, query);
        sigs[i] = SigType((val + b_[i]) / w_);
    }
    
    // calc lccs-lsh to find nearest neighbors
    int   verif_cnt  = 0;
    int   step = (10*cand+top_k-1 + m_-1) / m_;
    float dist = -1.0f;
    bucketerp_.for_candidates(step, sigs, [&](int idx) {
        // verify the true distance of idx
        const DType *data = data_ + (u64) idx*dim_;
        dist = calc_p2h_dist<DType>(dim_, data, query);
        list->insert(dist, idx+1);
        ++verif_cnt;
    });
    return verif_cnt;
}

// -----------------------------------------------------------------------------
template<class DType>
float Ortho_LCCS_LSH<DType>::calc_query_hash(// calc hash value for query
    int   tid,                          // hash table id
    const float *query)                 // input query
{
    int shift = tid * dim_;
    float q1 = calc_inner_product<float>(dim_, query, a1_+shift);
    float q2 = calc_inner_product<float>(dim_, query, a2_+shift);

    return -q2 / q1;
}

} // end namespace p2h

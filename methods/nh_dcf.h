#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

#include "def.h"
#include "util.h"
#include "pri_queue.h"
#include "qalsh.h"

namespace p2h {

// -----------------------------------------------------------------------------
//  NH_DCF: Nearest Hyperplane Hashing based on Dynamic Counting Framework
//
//  Description:
//  1. Convert P2HNNS to P2PNNS, with Randomized Sampling
//  2  Use QALSH (based on Dynamic Collision Counting Framework) for P2PNNS
// -----------------------------------------------------------------------------
template<class DType>
class NH_DCF {
public:
    NH_DCF(                         // constructor
        int   n,                        // number of data objects
        int   d,                        // dimension of data objects
        int   m,                        // #hash tables
        int   s,                        // scale factor of dimension
        const DType *data);             // input data

    // -------------------------------------------------------------------------
    ~NH_DCF();                      // destructor

    // -------------------------------------------------------------------------
    void display();                 // display parameters

    // -------------------------------------------------------------------------
    int nns(                        // point-to-hyperplane NNS
        int   top_k,                    // top-k value
        int   l,                        // collision threshold
        int   cand,                     // #candidates
        const float *query,             // input query
        MinK_List *list);               // top-k results (return)

    // -------------------------------------------------------------------------
    u64 get_memory_usage() {        // get memory usage
        u64 ret = 0UL;
        ret += sizeof(*this);
        ret += lsh_->get_memory_usage();
        return ret;
    }

protected:
    int   n_pts_;                   // number of data objects
    int   dim_;                     // dimension of data objects
    int   sample_dim_;              // sample dimension
    int   nh_dim_;                  // new data dimension after transformation
    float M_;                       // max l2-norm of o' after transformation
    const DType *data_;             // original data objects
    QALSH *lsh_;                    // QALSH for nh_data with sampling

    // -------------------------------------------------------------------------
    void transform_data(            // data transformation
        const  DType *data,             // input data
        int    &sample_d,               // number of sample dimension (return)
        float  &norm,                   // l2-norm-sqr of sample_data (return)
        Result *sample_data);           // sample data (return)

    // -------------------------------------------------------------------------
    void transform_query(           // query transformation
        const  float *query,            // input query
        int    &sample_d,               // number of sample dimension (return)
        Result *sample_query);          // sample query (return)
};

// -----------------------------------------------------------------------------
template<class DType>
NH_DCF<DType>::NH_DCF(              // constructor
    int   n,                            // number of data objects
    int   d,                            // dimension of data objects
    int   m,                            // #hash tables
    int   s,                            // scale factor of dimension
    const DType *data)                  // input data
    : n_pts_(n), dim_(d), sample_dim_(d*s), nh_dim_(d*(d+1)/2+1), data_(data)
{
    assert(sample_dim_ <= nh_dim_-1);
    M_   = MINREAL;
    lsh_ = new QALSH(n, nh_dim_, m);
    
    // build hash tables for rqalsh
    int    sample_d = -1;           // number of sample dimensions
    float  *norms = new float[n];
    Result *sample_data = new Result[sample_dim_];

    for (int i = 0; i < n; ++i) {
        // data transformation
        transform_data(data + (u64) i*d, sample_d, norms[i], sample_data);
        if (M_ < norms[i]) M_ = norms[i];

        // calc partial hash values
        for (int j = 0; j < m; ++j) {
            float val = lsh_->calc_hash_value(sample_d, j, sample_data);
            lsh_->tables_[(u64) j*n+i].id_  = i;
            lsh_->tables_[(u64) j*n+i].key_ = val;
        }
    }
    // calc the final hash values
    for (int i = 0; i < n; ++i) {
        float last = sqrt(M_ - norms[i]);
        for (int j = 0; j < m; ++j) {
            int pos = (j+1)*nh_dim_ - 1;
            lsh_->tables_[(u64) j*n+i].key_ += lsh_->a_[pos]*last;
        }
    }
    // sort hash tables in ascending order by hash values
    for (int i = 0; i < m; ++i) {
        qsort(&lsh_->tables_[(u64) i*n], n, sizeof(Result), ResultComp);
    }

    // release space
    delete[] sample_data;
    delete[] norms;
}

// -----------------------------------------------------------------------------
template<class DType>
void NH_DCF<DType>::transform_data( // data transformation
    const  DType *data,                 // input data
    int    &sample_d,                   // number of sample dimension (return)
    float  &norm,                       // l2-norm-sqr of sample_data (return)
    Result *sample_data)                // sample data (return)
{
    // 1: calc probability vector and the l2-norm-square of data
    float *prob = new float[dim_]; // probability vector
    init_prob_vector<DType>(dim_, data, prob);

    // 2: randomly sample coordinate of data as the coordinate of sample_data
    sample_d = 0; norm = 0.0f;
    bool *checked = new bool[nh_dim_];
    memset(checked, false, sizeof(bool)*nh_dim_);

    // 2.1: first consider the largest coordinate
    int sid = dim_-1;

    checked[sid] = true;
    float key = (float) SQR(data[sid]);
    sample_data[sample_d].id_  = sid;
    sample_data[sample_d].key_ = key;
    norm += SQR(key); ++sample_d;
    
    // 2.2: consider the combination of the left coordinates
    for (int i = 1; i < sample_dim_; ++i) {
        int idx = coord_sampling(dim_-1, prob);
        int idy = coord_sampling(dim_, prob);
        if (idx > idy) std::swap(idx, idy);

        if (idx == idy) {
            sid = idx;
            if (checked[sid]) continue;

            // calc the square coordinates of sample_data
            checked[sid] = true;
            key = (float) SQR(data[idx]);
            sample_data[sample_d].id_  = sid;
            sample_data[sample_d].key_ = key;
            norm += SQR(key); ++sample_d;
        }
        else {
            sid = dim_ + (idx*dim_-idx*(idx+1)/2) + (idy-idx-1);
            if (checked[sid]) continue; 

            // calc the differential coordinates of sample_data
            checked[sid] = true;
            key = (float) data[idx] * data[idy];
            sample_data[sample_d].id_  = sid;
            sample_data[sample_d].key_ = key;
            norm += SQR(key); ++sample_d;
        }
    }
    delete[] prob;
    delete[] checked;
}

// -----------------------------------------------------------------------------
template<class DType>
NH_DCF<DType>::~NH_DCF()            // destructor
{
    delete lsh_;
}

// -----------------------------------------------------------------------------
template<class DType>
void NH_DCF<DType>::display()       // display parameters
{
    printf("Parameters of NH_DCF:\n");
    printf("n          = %d\n", n_pts_);
    printf("dim        = %d\n", dim_);
    printf("nh_dim     = %d\n", nh_dim_);
    printf("sample_dim = %d\n", sample_dim_);
    printf("m          = %d\n", lsh_->m_);
    printf("max_norm   = %f\n", sqrt(M_));
    printf("\n");
}

// -----------------------------------------------------------------------------
template<class DType>
int NH_DCF<DType>::nns(             // point-to-hyperplane NNS
    int   top_k,                        // top-k value
    int   l,                            // collision threshold
    int   cand,                         // #candidates
    const float *query,                 // input query
    MinK_List *list)                    // top-k results (return)
{
    // query transformation
    int    sample_d = -1;
    Result *sample_query = new Result[sample_dim_];
    transform_query(query, sample_d, sample_query);

    // conduct nearest neighbor search by qalsh
    std::vector<int> cand_list;
    int verif_cnt = lsh_->nns(l, cand+top_k-1, sample_d, sample_query, 
        cand_list);

    int   idx  = -1;
    float dist = -1.0f;
    for (int i = 0; i < verif_cnt; ++i) {
        idx  = cand_list[i];
        dist = calc_p2h_dist<DType>(dim_, data_+(u64)idx*dim_, query);
        list->insert(dist, idx+1);
    }
    delete[] sample_query;

    return verif_cnt;
}

// -----------------------------------------------------------------------------
template<class DType>
void NH_DCF<DType>::transform_query(// query transformation
    const  float *query,                // input query
    int    &sample_d,                   // number of sample dimension (return)
    Result *sample_query)               // sample query (return)
{
    // 1: calc probability vector
    float *prob = new float[dim_];
    init_prob_vector<float>(dim_, query, prob);

    // 2: randomly sample the coordinates for sample_query
    sample_d = 0;
    bool *checked = new bool[nh_dim_];
    memset(checked, false, sizeof(bool)*nh_dim_);

    int   sid = -1;
    float key, norm_sample_q = 0.0f;
    for (int i = 0; i < sample_dim_; ++i) {
        int idx = coord_sampling(dim_, prob);
        int idy = coord_sampling(dim_, prob);
        if (idx > idy) std::swap(idx, idy);

        if (idx == idy) {
            sid = idx;
            if (checked[sid]) continue; 

            // calc the square coordinates
            checked[sid] = true;
            key = -query[idx] * query[idx];
            sample_query[sample_d].id_  = sid;
            sample_query[sample_d].key_ = key;
            norm_sample_q += SQR(key); ++sample_d;
        }
        else {
            sid = dim_ + (idx*dim_-idx*(idx+1)/2) + (idy-idx-1);
            if (checked[sid]) continue;

            // calc the differential coordinates
            checked[sid] = true;
            key = -2 * query[idx] * query[idy];
            sample_query[sample_d].id_  = sid;
            sample_query[sample_d].key_ = key;
            norm_sample_q += SQR(key); ++sample_d;
        }
    }    
    // multiply lambda
    float lambda = sqrt(M_ / norm_sample_q);
    for (int i = 0; i < sample_d; ++i) sample_query[i].key_ *= lambda;

    delete[] prob;
    delete[] checked;
}

} // end namespace p2h

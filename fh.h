#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

#include "def.h"
#include "util.h"
#include "pri_queue.h"
#include "rqalsh.h"

namespace p2h {

// -----------------------------------------------------------------------------
//  FH: Furthest Hyperplane Hashing
//
//  Description:
//  1. Convert P2HNNS to P2PFNS, with Randomized Sampling
//  2. Use Dynamic Counting framework (RQALSH) for P2PFNS
//  3. With Data Dependent Multi-Partitioning for P2PFNS
// -----------------------------------------------------------------------------
template<class DType>
class FH {
public:
    FH(                             // constructor
        int   n,                        // number of data objects
        int   d,                        // dimension of data objects
        int   m,                        // #hash tables
        int   s,                        // scale factor of dimension
        float b,                        // interval ratio
        const DType *data);             // input data

    // -------------------------------------------------------------------------
    ~FH();                          // destructor

    // -------------------------------------------------------------------------
    void display();                 // display parameters

    // -------------------------------------------------------------------------
    int nns(                        // point-to-hyperplane NNS
        int   top_k,                    // top-k value
        int   l,                        // separation threshold
        int   cand,                     // #candidates
        const float *query,             // input query
        MinK_List *list);               // top-k results (return)

    // -------------------------------------------------------------------------
    u64 get_memory_usage() {        // get memory usage
        u64 ret = 0UL;
        ret += sizeof(*this);
        ret += sizeof(float)*fh_dim_; // centroid_
        ret += sizeof(int)*n_pts_;    // shift_id_
        for (auto hash : hash_) {     // blocks_
            ret += hash->get_memory_usage();
        }
        return ret;
    }

protected:
    int   n_pts_;                   // number of data objects
    int   dim_;                     // dimension of data objects
    int   sample_dim_;              // sample dimension
    int   fh_dim_;                  // new data dimension after transformation
    float M_;                       // max l2-norm sqr of o'
    const DType *data_;             // original data objects

    int *shift_id_;                 // shift data id
    std::vector<RQALSH*> hash_;     // blocks

    // -------------------------------------------------------------------------
    void transform_data(            // data transformation
        const  DType *data,             // input data
        int    &sample_d,               // sample dimension (return)
        float  &norm,                   // l2-norm-sqr of sample_data (return)
        Result *sample_data,            // sample data (return)
        float  *centroid);              // centroid (return)

    // -------------------------------------------------------------------------
    float calc_transform_dist(      // calc l2-dist after transformation
        int   sample_d,                 // dimension of sample data
        float last,                     // the last coordinate of sample data
        float norm_sqr_ctrd,            // the l2-norm-sqr of centroid
        const Result *sample_data,      // sample data
        const float *centroid);         // centroid after data transformation

    // -------------------------------------------------------------------------
    void transform_query(           // query transformation
        const  float *query,            // input query
        int    &sample_d,               // dimension of sample query (return)
        Result *sample_query);          // sample query after transform (return)
};

// -----------------------------------------------------------------------------
template<class DType>
FH<DType>::FH(                      // constructor
    int   n,                            // number of data objects
    int   d,                            // dimension of data objects
    int   m,                            // number of hash tables
    int   s,                            // scale factor of dimension
    float b,                            // interval ratio
    const DType *data)                  // input data
    : n_pts_(n), dim_(d), sample_dim_(d*s), fh_dim_(d*(d+1)/2+1), data_(data)
{
    // -------------------------------------------------------------------------
    //  calc centroid, l2-norm, and max l2-norm
    // -------------------------------------------------------------------------
    int    fh_dim_1     = fh_dim_ - 1;
    float  *norm        = new float[n];       // l2-norm of sample_data
    float  *centroid    = new float[fh_dim_]; // centroid of sample_data
    int    *sample_d    = new int[n];         // number of sample dimensions
    Result *sample_data = new Result[(u64) n*sample_dim_]; // sample data

    // calc l2-norm & update max l2-norm-sqr
    memset(centroid, 0.0f, sizeof(float)*fh_dim_);
    M_ = MINREAL;
    for (int i = 0; i < n; ++i) {
        transform_data(data + (u64) i*d, sample_d[i], norm[i], 
            sample_data + (u64) i*sample_dim_, centroid);
        if (M_ < norm[i]) M_ = norm[i];
    }
    printf("determine M=%f\n", sqrt(M_));

    // calc centroid and its l2-norm-sqr
    float norm_sqr_ctrd = 0.0f;
    for (int i = 0; i < fh_dim_1; ++i) {
        centroid[i] /= n;
        norm_sqr_ctrd += SQR(centroid[i]);
    }
    float last = 0.0f;
    for (int i = 0; i < n; ++i) {
        norm[i] = sqrt(M_ - norm[i]);
        last += norm[i];
    }
    last /= n;
    centroid[fh_dim_1] = last;
    norm_sqr_ctrd += SQR(last);
    printf("calc centroid\n");

    // -------------------------------------------------------------------------
    //  determine shift_id after shifting data objects to centroid
    // -------------------------------------------------------------------------
    Result *arr = new Result[n];
    for (int i = 0; i < n; ++i) {
        arr[i].id_  = i;
        arr[i].key_ = calc_transform_dist(sample_d[i], norm[i], norm_sqr_ctrd, 
            sample_data + (u64) i*sample_dim_, centroid);
    }
    qsort(arr, n, sizeof(Result), ResultCompDesc);

    shift_id_ = new int[n];
    for (int i = 0; i < n; ++i) shift_id_[i] = arr[i].id_;
    printf("determine shift_id\n");

    // -------------------------------------------------------------------------
    //  divide datasets into blocks and build hash tables for each block
    // -------------------------------------------------------------------------
    int start = 0;
    while (start < n) {
        // partition block
        float min_radius  = b*arr[start].key_;
        int   block_index = start, cnt = 0;
        while (block_index < n && arr[block_index].key_ > min_radius) {
            ++block_index;
            if (++cnt >= MAX_BLOCK_NUM) break;
        }

        // add block
        const int *index = (const int*) shift_id_ + start;
        RQALSH *hash = new RQALSH(cnt, fh_dim_, m, index);

        if (cnt > N_PTS) {
            for (int i = 0; i < cnt; ++i) {
                // calc the hash values of P*f(o)
                int idx = index[i];
                for (int j = 0; j < m; ++j) {
                    float val = hash->calc_hash_value(sample_d[idx], j, 
                        norm[idx], sample_data + (u64) idx*sample_dim_);
                    hash->tables_[(u64) j*cnt+i].id_  = i;
                    hash->tables_[(u64) j*cnt+i].key_ = val;
                }
            }
            // sort hash tables in ascending order of hash values
            for (int i = 0; i < m; ++i) {
                u64 shift = (u64) i*cnt;
                qsort(hash->tables_+shift, cnt, sizeof(Result), ResultComp);
            }
        }
        hash_.push_back(hash);
        start += cnt;
        printf("cnt=%d, total=%d\n", cnt, start);
    }
    assert(start == n);
    printf("\n");
    
    // -------------------------------------------------------------------------
    //  release space
    // -------------------------------------------------------------------------
    delete[] arr;
    delete[] sample_data;
    delete[] sample_d;
    delete[] centroid;
    delete[] norm;
}

// -----------------------------------------------------------------------------
template<class DType>
void FH<DType>::transform_data(     // data transformation
    const  DType *data,                 // input data
    int    &sample_d,                   // sample dimension (return)
    float  &norm,                       // l2-norm-sqr of sample_data (return)
    Result *sample_data,                // sample data (return)
    float  *centroid)                   // centroid (return)
{
    // 1: calc probability vector and the l2-norm-square of data
    float *prob = new float[dim_]; // probability vector
    init_prob_vector<DType>(dim_, data, prob);

    // 2: randomly sample the coordinates for sample_data
    sample_d = 0; norm = 0.0f;
    bool *checked = new bool[fh_dim_];
    memset(checked, false, sizeof(bool)*fh_dim_);
    
    // 2.1: first select the largest coordinate
    int sid = dim_-1; // sample id

    checked[sid] = true;
    float key = (float) SQR(data[sid]);
    sample_data[sample_d].id_  = sid;
    sample_data[sample_d].key_ = key;
    centroid[sid] += key; norm += SQR(key); ++sample_d;

    // 2.2: consider the combination of the remain coordinates
    for (int i = 1; i < sample_dim_; ++i) {
        int idx = coord_sampling(dim_-1, prob); // lower  dim
        int idy = coord_sampling(dim_, prob);   // higher dim
        if (idx > idy) std::swap(idx, idy);

        if (idx == idy) {
            sid = idx;
            if (checked[sid]) continue;

            // calc the square coordinates of sample_data
            checked[sid] = true;
            key = (float) SQR(data[idx]);
            sample_data[sample_d].id_  = sid;
            sample_data[sample_d].key_ = key;
            centroid[sid] += key; norm += SQR(key); ++sample_d;
        }
        else {
            sid = dim_ + (idx*dim_-idx*(idx+1)/2) + (idy-idx-1);
            if (checked[sid]) continue;

            // calc the differential coordinates of sample_data
            checked[sid] = true;
            key = (float) data[idx] * data[idy];
            sample_data[sample_d].id_  = sid;
            sample_data[sample_d].key_ = key;
            centroid[sid] += key; norm += SQR(key); ++sample_d;
        }
    }
    delete[] checked;
    delete[] prob;
}

// -----------------------------------------------------------------------------
template<class DType>
float FH<DType>::calc_transform_dist(// calc l2-dist after transform
    int   sample_d,                     // dimension of sample data
    float last,                         // the last coordinate of sample data
    float norm_sqr_ctrd,                // the l2-norm-sqr of centroid
    const Result *sample_data,          // sample data
    const float *centroid)              // centroid after data transformation
{
    int   idx  = -1;
    float dist = norm_sqr_ctrd, tmp, diff;
    
    // calc the distance for the sample dimension
    for (int i = 0; i < sample_d; ++i) {
        idx  = sample_data[i].id_; tmp = centroid[idx];
        diff = sample_data[i].key_ - tmp;

        dist -= SQR(tmp);
        dist += SQR(diff);
    }
    // calc the distance for the last coordinate
    tmp  = centroid[fh_dim_-1];
    dist -= SQR(tmp);
    dist += SQR(last - tmp);

    return sqrt(dist);
}

// -----------------------------------------------------------------------------
template<class DType>
FH<DType>::~FH()                    // destructor
{
    delete[] shift_id_;
    if (!hash_.empty()) {
        for (auto hash : hash_) delete hash;
        std::vector<RQALSH*>().swap(hash_);
    }
}

// -----------------------------------------------------------------------------
template<class DType>
void FH<DType>::display()           // display parameters
{
    printf("Parameters of FH:\n");
    printf("n          = %d\n", n_pts_);
    printf("dim        = %d\n", dim_);
    printf("sample_dim = %d\n", sample_dim_);
    printf("fh_dim     = %d\n", fh_dim_);
    printf("max_norm   = %f\n", sqrt(M_));
    printf("#blocks    = %d\n", (int) hash_.size());
    printf("\n");
}

// -----------------------------------------------------------------------------
template<class DType>
int FH<DType>::nns(                 // point-to-hyperplane NNS
    int   top_k,                        // top-k value
    int   l,                            // separation threshold
    int   cand,                         // #candidates
    const float *query,                 // input query
    MinK_List *list)                    // top-k results (return)
{
    // query transformation
    int    sample_d = -1;
    Result *sample_query = new Result[sample_dim_];
    transform_query(query, sample_d, sample_query);
    
    // point-to-hyperplane NNS
    int   idx, size, verif_cnt = 0, n_cand = cand+top_k-1;
    float kfn_dist, kdist, dist, fix_val = 2*M_;
    std::vector<int> cand_list;
    
    for (auto hash : hash_) {
        // check candidates returned by rqalsh
        kfn_dist = -1.0f;
        if (list->isFull()) {
            kdist = list->max_key();
            kfn_dist = sqrt(fix_val - 2*kdist*kdist);
        }
        size = hash->fns(l, n_cand, kfn_dist, sample_d, sample_query, cand_list);
        for (int j = 0; j < size; ++j) {
            idx  = cand_list[j];
            dist = calc_p2h_dist<DType>(dim_, data_+(u64)idx*dim_, query);
            list->insert(dist, idx + 1);
        }
        // update info
        verif_cnt += size; n_cand -= size;
        if (n_cand <= 0) break;
    }
    delete[] sample_query;

    return verif_cnt;
}

// -----------------------------------------------------------------------------
template<class DType>
void FH<DType>::transform_query(    // query transformation
    const  float *query,                // input query
    int    &sample_d,                   // dimension of sample query (return)
    Result *sample_q)                   // sample query after transform (return)
{
    // 1: calc probability vector
    float *prob = new float[dim_];
    init_prob_vector<float>(dim_, query, prob);

    // 2: randomly sample the coordinates for sample_query
    sample_d = 0;
    bool *checked = new bool[fh_dim_];
    memset(checked, false, sizeof(bool)*fh_dim_);

    int   sid = -1;
    float key, norm_sample_q = 0.0f;
    for (int i = 0; i < sample_dim_; ++i) {
        int idx = coord_sampling(dim_-1, prob); // lower  dim
        int idy = coord_sampling(dim_, prob);   // higher dim
        if (idx > idy) std::swap(idx, idy);

        if (idx == idy) {
            sid = idx;
            if (checked[sid]) continue;

            // calc the square coordinates
            checked[sid] = true;
            key = query[idx] * query[idx];
            sample_q[sample_d].id_  = sid;
            sample_q[sample_d].key_ = key;
            norm_sample_q += SQR(key); ++sample_d;
        }
        else {
            sid = dim_ + (idx*dim_-idx*(idx+1)/2) + (idy-idx-1);
            if (checked[sid]) continue;

            // calc the differential coordinates
            checked[sid] = true;
            key = 2 * query[idx] * query[idy];
            sample_q[sample_d].id_  = sid;
            sample_q[sample_d].key_ = key;
            norm_sample_q += SQR(key); ++sample_d;
        }
    }
    // multiply lambda
    float lambda = sqrt(M_ / norm_sample_q);
    for (int i = 0; i < sample_d; ++i) sample_q[i].key_ *= lambda;

    delete[] prob;
    delete[] checked;
}

} // end namespace p2h

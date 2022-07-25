#pragma once

#include <iostream>
#include <algorithm>
#include <vector>
#include <cstdint>
#include <numeric>
#include <cstddef>

#include "def.h"
#include "util.h"
#include "pri_queue.h"
#include "ortho_lsh_lite.h"

namespace p2h {

// -----------------------------------------------------------------------------
//  Metric_Node: leaf node and internal nodea data structure of Ball_Tree
// -----------------------------------------------------------------------------
template<class DType>
class Metric_Node {
public:
    int   n_;                       // number of data points
    int   d_;                       // dimension of data points
    int   leaf_pts_;                // leaf pts for indexing
    Metric_Node<DType> *lc_;        // left  child
    Metric_Node<DType> *rc_;        // right child
    float radius_;                  // radius of data points
    float *center_;                 // centroid of data points
    
    int   *index_;                  // data index (re-order for leaf node only)
    DType *data_;                   // data points (for leaf node only)
    float *local_r_;                // local radius of data (for leaf node only)
    Ortho_LSH_Lite<DType> *lsh_;    // ortho_lsh (for leaf node only)
    
    // -------------------------------------------------------------------------
    Metric_Node(                    // constructor
        int   n,                        // number of data points
        int   d,                        // dimension of data points
        int   m,                        // number of hash tables for leaf node
        int   leaf_pts,                 // leaf pts for indexing
        bool  is_leaf,                  // is leaf node
        Metric_Node<DType> *lc,         // left  child
        Metric_Node<DType> *rc,         // right child
        int   *index,                   // data index
        const float *a1,                // lsh func
        const float *a2,                // lsh func
        const DType *data);             // data points

    // -------------------------------------------------------------------------
    ~Metric_Node();                 // desctructor

    // -------------------------------------------------------------------------
    void nns(                       // point-to-hyperplane nns on metric node
        int   top_k,                    // top_k value
        int   l,                        // collision threshold for leaf node
        float tau,                      // candidate ratio for leaf node
        float c,                        // approximate ratio
        float ip,                       // inner product for query and centroid
        float norm_q,                   // l2-norm of query for d dim
        const float *query,             // query point
        const float *q_val,             // hash values of query
        int   &cand,                    // # candidates to check (return)
        MinK_List *list);               // top-k results (return)

    // -------------------------------------------------------------------------
    float est_lower_bound(          // estimate lower bound
        float c,                        // approximation ratio
        float ip,                       // inner product for query and centroid
        float norm_q,                   // the norm of query for d dim
        float radius);                  // radius

    // -------------------------------------------------------------------------
    void linear_scan(               // linear scan
        float c,                        // approximate ratio
        float ip,                       // inner product for query and centroid
        float norm_q,                   // the norm of query for d dim
        const float *query,             // input query
        int   &cand,                    // # candidates to check (return)
        MinK_List *list);               // top-k PNN results (return)

    // -------------------------------------------------------------------------
    void traversal(                 // traversal metric-tree
        std::vector<int> &leaf_size);   // leaf size (return)
    
    // -------------------------------------------------------------------------
    u64 get_memory_usage() {
        u64 ret = 0UL;
        ret += sizeof(*this);
        ret += sizeof(float)*d_; // center_
        
        if (data_ != nullptr) { // leaf node
            ret += sizeof(float)*n_; // local_r_ (ignore data as all need it)
            if (n_ > leaf_pts_) ret += lsh_->get_memory_usage(); // lsh_
            return ret;
        } else { // internal node
            return ret + lc_->get_memory_usage() + rc_->get_memory_usage();
        }
    }
};

// -----------------------------------------------------------------------------
template<class DType>
Metric_Node<DType>::Metric_Node(    // constructor
    int   n,                            // number of data points
    int   d,                            // data dimension
    int   m,                            // number of hash tables for leaf node
    int   leaf_pts,                     // leaf pts for indexing
    bool  is_leaf,                      // is leaf node
    Metric_Node *lc,                    // left  child
    Metric_Node *rc,                    // right child
    int   *index,                       // data index
    const float *a1,                    // lsh func
    const float *a2,                    // lsh func
    const DType *data)                  // data points
    : n_(n), d_(d), leaf_pts_(leaf_pts), lc_(lc), rc_(rc), index_(index), 
    local_r_(nullptr), data_(nullptr), lsh_(nullptr)
{
    center_ = new float[d];
    if (is_leaf) {
        // calc the centroid and radius based on data
        calc_centroid<DType>(n, d, index, data, center_);

        Result *result = new Result[n];
        for (int i = 0; i < n; ++i) {
            const DType *point = data + (u64) index[i]*d;
            result[i].id_  = index[i];
            result[i].key_ = calc_l2_dist2<DType>(d, point, center_);
        }
        qsort(result, n, sizeof(Result), ResultCompDesc);
        radius_ = result[0].key_;
        
        // re-order index_ & data_ and init local_r_ (only for leaf node)
        local_r_ = new float[n];
        data_    = new DType[(u64) n*d];
        for (int i = 0; i < n; ++i) {
            index_[i]   = result[i].id_;
            local_r_[i] = result[i].key_;
            
            const DType *point = data + (u64) index_[i]*d;
            std::copy(point, point+d, data_+(u64)i*d);
        }
        delete[] result;
        
        // init ortho_lsh (only for leaf node with large size)
        if (n > leaf_pts_) {
            lsh_ = new Ortho_LSH_Lite<DType>(n, d, m, index_, a1, a2, data_);
        }
    }
    else {
        // calc the centroid based on its two leaves
        int   ln = lc_->n_, rn = rc_->n_;
        float *l_ctrd = lc_->center_, *r_ctrd = rc_->center_;
        for (int i = 0; i < d; ++i) {
            center_[i] = (ln*l_ctrd[i] + rn*r_ctrd[i]) / n;
        }
        // calc the radius
        radius_ = -1.0f;
        for (int i = 0; i < n; ++i) {
            const DType *point = data + (u64) index[i]*d;
            float dist = calc_l2_sqr2<DType>(d, point, center_);
            if (dist > radius_) radius_ = dist;
        }
        radius_ = sqrt(radius_);
    }
}

// -----------------------------------------------------------------------------
template<class DType>
Metric_Node<DType>::~Metric_Node()  // desctructor
{
    if (lc_  != nullptr) { delete lc_;  lc_  = nullptr; }
    if (rc_  != nullptr) { delete rc_;  rc_  = nullptr; }
    if (lsh_ != nullptr) { delete lsh_; lsh_ = nullptr; }
    
    if (center_  != nullptr) { delete[] center_;  center_  = nullptr; }
    if (local_r_ != nullptr) { delete[] local_r_; local_r_ = nullptr; }
    if (data_    != nullptr) { delete[] data_;    data_    = nullptr; }
}

// -----------------------------------------------------------------------------
template<class DType>
void Metric_Node<DType>::nns(       // point-to-hyperplane nns on metric node
    int   top_k,                        // top_k value
    int   l,                            // collision threshold for leaf node
    float tau,                          // candidate ratio for leaf node
    float c,                            // approximate ratio
    float ip,                           // inner product for query and centroid
    float norm_q,                       // the norm of query for d dim
    const float *query,                 // query point
    const float *q_val,                 // hash values of query
    int   &cand,                        // # candidates to check (return)
    MinK_List *list)                    // top-k results (return)
{
    // early stop 1
    if (cand <= 0) return;
    
    // early stop 2
    float lower_bound = est_lower_bound(c, fabs(ip), norm_q, radius_);
    if (lower_bound >= list->max_key()) return;
    
    // traversal the tree
    if (data_ != nullptr) { // leaf node
        if (n_ > leaf_pts_ && tau < 0.05f) { // speedup by ortho-lsh
            int this_cand = (int) ceil(tau*n_) + top_k;
            this_cand = std::min(std::min(this_cand, cand), n_);
            
            lsh_->nns(l, this_cand, query, q_val, list); 
            cand -= this_cand;
        }
        else { // directly linear scan (even for the last, can stop early)
            linear_scan(c, ip, norm_q, query, cand, list);
        }
    }
    else { // internal node
        float lc_ip = calc_inner_product<float>(d_, lc_->center_, query);
        float rc_ip = (ip*n_ - lc_ip*lc_->n_) / rc_->n_;
        float lc_dp = fabs(lc_ip), rc_dp = fabs(rc_ip);
        
        if (lc_dp < rc_dp) {
            lc_->nns(top_k, l, tau, c, lc_ip, norm_q, query, q_val, cand, list);
            rc_->nns(top_k, l, tau, c, rc_ip, norm_q, query, q_val, cand, list);
        }
        else {
            rc_->nns(top_k, l, tau, c, rc_ip, norm_q, query, q_val, cand, list);
            lc_->nns(top_k, l, tau, c, lc_ip, norm_q, query, q_val, cand, list);
        }
    }
}

// -----------------------------------------------------------------------------
template<class DType>
float Metric_Node<DType>::est_lower_bound(// estimate lower bound
    float c,                            // approximation ratio
    float ip,                           // inner product for query and centroid
    float norm_q,                       // the norm of query for d dim
    float radius)                       // radius
{
    float threshold = radius * norm_q;
    
    if (ip > threshold) return c*(ip-threshold);
    else return 0.0f;
}

// -----------------------------------------------------------------------------
template<class DType>
void Metric_Node<DType>::linear_scan(// linear scan
    float c,                            // approximate ratio
    float ip,                           // inner product for query and centroid
    float norm_q,                       // the norm of query for d dim
    const float *query,                 // input query
    int   &cand,                        // # candidates to check (return)
    MinK_List *list)                    // top-k results (return)
{
    float abso_ip = fabs(ip);
    float lambda = list->max_key();
    for (int i = 0; i < n_; ++i) {
        // early pruning
        float lower_bound = est_lower_bound(c, abso_ip, norm_q, local_r_[i]);
        if (lower_bound >= lambda) return;

        // compute the actual distance
        const DType *point = data_ + (u64) i*d_;
        float dist = calc_p2h_dist<DType>(d_, point, query);
        
        lambda = list->insert(dist, index_[i]+1); 
        --cand;
        if (cand <= 0) return;
    }
}

// -----------------------------------------------------------------------------
template<class DType>
void Metric_Node<DType>::traversal( // traversal metric-tree
    std::vector<int> &leaf_size)        // leaf size (return)
{
    if (data_ != nullptr) { // leaf node
        leaf_size.push_back(n_);
    }
    else { // internal node
        lc_->traversal(leaf_size);
        rc_->traversal(leaf_size);
    }
}

} // end namespace p2h

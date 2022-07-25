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
#include "metric_node.h"

namespace p2h {

// -----------------------------------------------------------------------------
//  Metric_Tree is a data structure for Point-to-Hyperplane NNS 
// -----------------------------------------------------------------------------
template<class DType>
class Metric_Tree {
public:
    int   n_;                       // number of data points
    int   d_;                       // dimension of data points
    int   m_;                       // number of hash tables for leaf node
    int   leaf_;                    // leaf size of metric tree
    int   leaf_pts_;                // leaf pts  of metric tree
    const DType *data_;             // data points
    
    int   *index_;                  // data index
    float *a1_;                     // lsh func
    float *a2_;                     // lsh func
    Metric_Node<DType> *root_;      // root node of metric tree
    
    // -------------------------------------------------------------------------
    Metric_Tree(                    // constructor
        int   n,                        // number of data points
        int   d,                        // dimension of data points
        int   m,                        // number of hash tables for leaf node
        int   leaf,                     // leaf size of metric tree
        const DType *data);             // data points
    
    // -------------------------------------------------------------------------
    ~Metric_Tree();                 // destructor
    
    // -------------------------------------------------------------------------
    void display();                 // display metric tree
    
    // -------------------------------------------------------------------------
    int nns(                        // point-to-hyperplane search on metric tree
        int   top_k,                    // top_k value
        int   l,                        // number of collisions for leaf node
        int   cand,                     // number of candidates
        float c,                        // approximation ratio
        const float *query,             // input query
        MinK_List *list);               // top-k PNN results (return)
    
    // -------------------------------------------------------------------------
    void traversal(                 // traversal ball-tree to get leaf info
        std::vector<int> &leaf_size,    // leaf size (return)
        std::vector<int> &index);       // data index with leaf order (return)
    
    // -------------------------------------------------------------------------
    u64 get_memory_usage() {
        u64 ret = 0UL;
        ret += sizeof(*this);
        ret += (sizeof(int)*n_+sizeof(float)*m_*d_*2); // index_ + a1_ + a2_
        ret += root_->get_memory_usage(); // memory of metric nodes
        return ret;
    }

protected:
    // -------------------------------------------------------------------------
    void gen_ortho_proj(            // generate orthogonal random projections
        float *a1,                      // 1st random projection (return)
        float *a2);                     // 2nd random projection (return)
    
    // -------------------------------------------------------------------------
    Metric_Node<DType>* build(      // build a metric node
        int n,                          // number of data points
        int *index);                    // data index (return)

    // -------------------------------------------------------------------------
    int find_furthest_id(           // find furthest data id
        int from,                       // input data id
        int n,                          // number of data index
        int *index);                    // data index
    
    // -------------------------------------------------------------------------
    float calc_query_hash(          // calc hash value for query
        int   tid,                      // hash table id
        const float *query);            // input query
};

// -----------------------------------------------------------------------------
template<class DType>
Metric_Tree<DType>::Metric_Tree(    // constructor
    int   n,                            // number of data points
    int   d,                            // dimension of data points
    int   m,                            // number of hash tables for leaf node
    int   leaf,                         // leaf size of metric tree
    const DType *data)                  // data points
    : n_(n), d_(d), m_(m), leaf_(leaf), data_(data)
{
    // calc leaf_pts_
    leaf_pts_ = (int) ceil(LEAF_RATIO * n);
    if (leaf_pts_ < MIN_LEAF_PTS) leaf_pts_ = MIN_LEAF_PTS;
    
    // generate lsh functions
    int size = m*d;
    a1_ = new float[size]; a2_ = new float[size];
    for (int i = 0; i < m; ++i) {
        // generate two orthogonal random projection lines
        gen_ortho_proj(a1_+i*d, a2_+i*d);
    }
    
    // init index_
    int i = 0;
    index_ = new int[n];
    std::iota(index_, index_+n, i++);
    
    // build metric tree
    root_ = build(n, index_);
}

// -----------------------------------------------------------------------------
template<class DType>
void Metric_Tree<DType>::gen_ortho_proj(// generate orthogonal random projs
    float *a1,                          // 1st random projection (return)
    float *a2)                          // 2nd random projection (return)
{
    int   last = d_-1;
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
Metric_Node<DType>* Metric_Tree<DType>::build(// build a metric node 
    int n,                              // number of data points
    int *index)                         // data index (return)
{
    srand(1); // fix the metric-tree for different m values
    Metric_Node<DType> *cur = nullptr;
    if (n <= leaf_) {
        // build leaf node
        cur = new Metric_Node<DType>(n, d_, m_, leaf_pts_, true, nullptr, 
            nullptr, index, a1_, a2_, data_);
    }
    else {
        DType *w = new DType[d_];
        int left = 0, right = n-1, cnt=0;
        do {
            // build internal node
            int x_p = rand() % n;
            int l_p = find_furthest_id(x_p, n, index);
            int r_p = find_furthest_id(l_p, n, index);
            assert(l_p != r_p);
            // printf("x=%d, l=%d, r=%d\n", x_p, l_p, r_p);
            
            // note: we use l_p and r_p as two pivots
            const DType *l_pivot = data_ + (u64) index[l_p]*d_;
            const DType *r_pivot = data_ + (u64) index[r_p]*d_;
            float l_sqr = 0.0f, r_sqr = 0.0f;
            for (int i = 0; i < d_; ++i) {
                DType l_v = l_pivot[i], r_v = r_pivot[i];
                w[i] = r_v - l_v; l_sqr += SQR(l_v); r_sqr += SQR(r_v);
            }
            float b = 0.5f * (l_sqr - r_sqr);
    
            left = 0, right = n-1;
            while (left <= right) {
                const DType *x = data_ + (u64) index[left]*d_;
                float val = calc_inner_product<DType>(d_, w, x) + b;
                if (val < 0.0f) ++left;
                else { SWAP(index[left], index[right]); --right; }
            }
            // if (left <= 0 || left >= n) {
            //     printf("n=%d, left=%d, right=%d\n", n, left, right);
            // }
            ++cnt;
        } while ((left <= 0 || left >= n) && (cnt <= 3)); // ensure split into 2
        if (cnt > 3) left = n/2; // meet the case that cannot split, force split
        delete[] w;

        Metric_Node<DType> *lc = build(left,   index);
        Metric_Node<DType> *rc = build(n-left, index+left);
        cur = new Metric_Node<DType>(n, d_, m_, leaf_pts_, false, lc, rc, 
            index, a1_, a2_, data_);
    }
    return cur;
}

// -----------------------------------------------------------------------------
template<class DType>
int Metric_Tree<DType>::find_furthest_id(// find furthest data id
    int from,                           // input id
    int n,                              // number of data index
    int *index)                         // data index
{
    int   far_id   = -1;
    float far_dist = -1.0f;

    const DType *query = data_ + (u64) index[from]*d_;
    for (int i = 0; i < n; ++i) {
        if (i == from) continue;
        
        const DType *point = data_ + (u64) index[i]*d_;
        float dist = calc_l2_sqr<DType>(d_, point, query);
        if (far_dist < dist) { far_dist = dist; far_id = i; }
    }
    return far_id;
}

// -----------------------------------------------------------------------------
template<class DType>
Metric_Tree<DType>::~Metric_Tree()  // desctructor
{
    if (root_  != nullptr) { delete   root_;  root_  = nullptr; }
    if (index_ != nullptr) { delete[] index_; index_ = nullptr; }
    if (a1_    != nullptr) { delete[] a1_;    a1_    = nullptr; }
    if (a2_    != nullptr) { delete[] a2_;    a2_    = nullptr; }
}

// -----------------------------------------------------------------------------
template<class DType>
void Metric_Tree<DType>::display()  // display metric tree
{
    std::vector<int> leaf_size;
    root_->traversal(leaf_size);
        
    printf("Parameters of Metric_Tree:\n");
    printf("n        = %d\n", n_);
    printf("d        = %d\n", d_);
    printf("m        = %d\n", m_);
    printf("leaf     = %d\n", leaf_);
    printf("leaf_pts = %d\n", leaf_pts_);
    printf("# leaves = %d\n", (int) leaf_size.size());
    // for (int leaf : leaf_size) printf("%d\n", leaf);
    printf("\n");
    std::vector<int>().swap(leaf_size);
}

// -----------------------------------------------------------------------------
template<class DType>
int Metric_Tree<DType>::nns(        // point-to-hyperplane search on metric tree
    int   top_k,                        // top_k value
    int   l,                            // collision threshold for leaf node
    int   cand,                         // number of candidates
    float c,                            // approximation ratio
    const float *query,                 // input query
    MinK_List *list)                    // top-k results (return)
{
    // determine some parameters for early pruning and computation saving
    cand = std::min(cand+top_k-1, n_);
    float tau = (float) cand / (float) n_;
    
    float norm_q = sqrt(calc_inner_product<float>(d_, query, query));
    float ip = calc_inner_product<float>(d_, root_->center_, query);
    
    // calc the hash values of query
    float *q_val = new float[m_];
    for (int i = 0; i < m_; ++i) q_val[i] = calc_query_hash(i, query);
    
    // p2hnns for the query
    int total = cand;
    root_->nns(top_k, l, tau, c, ip, norm_q, query, q_val, cand, list);
    delete[] q_val;
    
    return total - cand;
}

// -----------------------------------------------------------------------------
template<class DType>
float Metric_Tree<DType>::calc_query_hash(// calc hash value for query
    int   tid,                          // hash table id
    const float *query)                 // input query
{
    int   shift = tid * d_;
    float q1 = calc_inner_product<float>(d_, query, a1_+shift);
    float q2 = calc_inner_product<float>(d_, query, a2_+shift);

    return -q2 / q1;
}

// -----------------------------------------------------------------------------
template<class DType>
void Metric_Tree<DType>::traversal( // traversal ball-tree to get leaf info
    std::vector<int> &leaf_size,        // leaf size (return)
    std::vector<int> &index)            // data index with leaf order (return)
{
    for (int i = 0; i < n_; ++i) index[i] = index_[i];
    root_->traversal(leaf_size);
}

} // end namespace p2h

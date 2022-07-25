#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

#include "def.h"
#include "util.h"
#include "pri_queue.h"

namespace p2h {

// -----------------------------------------------------------------------------
//  Ortho-LSH: an LSH scheme based on Random Projection on 2D Planes
//  
//  1. Project all data points into m 2D-Planes
//  2. Use QALSH for Point-to-Hyperplane Nearest Neighbor Search 
// -----------------------------------------------------------------------------
template<class DType>
class Ortho_LSH_Lite {
public:
    // -------------------------------------------------------------------------
    Ortho_LSH_Lite(                 // constructor
        int   n,                        // number of data points
        int   d,                        // dimension of data points
        int   m,                        // number of random 2-dim ortho planes
        const int *index,               // index of input data
        const float *a1,                // lsh func
        const float *a2,                // lsh func
        const DType *data);             // input data
    
    // -------------------------------------------------------------------------
    ~Ortho_LSH_Lite();              // destructor
    
    // -------------------------------------------------------------------------
    void display();                 // display parameters
    
    // -------------------------------------------------------------------------
    int nns(                        // point-to-hyperplane nns
        int   l,                        // collision threshold
        int   cand,                     // number of candidates
        const float *query,             // query point
        const float *q_val,             // hash values of query
        MinK_List *list);               // top-k results (return)
    
    // -------------------------------------------------------------------------
    u64 get_memory_usage() {        // get memory usage
        u64 ret = 0UL;
        ret += sizeof(*this);
        ret += sizeof(Result)*(u64)m_*n_; // tables_
        return ret;
    }

protected:
    int   n_;                       // number of data points
    int   d_;                       // dimension of data points
    int   m_;                       // number of 2-dim ortho planes
    const int *index_;              // index of data points
    const float *a1_;               // lsh func
    const float *a2_;               // lsh func
    const DType *data_;             // data points
    Result *tables_;                // hash tables, O(m*n)

    int   *freq_;                   // collision frequency for n data points
    bool  *checked_;                // checked_ or not for n data points
    bool  *flag_;                   // flag_ for m hash tables
    int   *l_pos_;                  // left  positions for m hash tables
    int   *r_pos_;                  // right positions for m hash tables
    const float *q_val_;            // m hash values of query
    
    // -------------------------------------------------------------------------
    float calc_data_hash(           // calc hash value for data point
        const float *a1,                // lsh func
        const float *a2,                // lsh func
        const DType *data);             // input data point

    // -------------------------------------------------------------------------
    void alloc();                   // alloc space for assistant parameters
    
    // -------------------------------------------------------------------------
    float init_position();          // init left/right positions

    // -------------------------------------------------------------------------
    int dcc(                        // dynamic collision counting
        int   l,                        // collision threshold
        int   cand,                     // candidate size
        float width,                    // bucket width
        const float *query,             // input query
        MinK_List *list);               // top-k results (return)

    // -------------------------------------------------------------------------
    void free();                    // free space for assistant parameters
};

// -----------------------------------------------------------------------------
template<class DType>
Ortho_LSH_Lite<DType>::Ortho_LSH_Lite(// constructor
    int   n,                            // number of data points
    int   d,                            // dimension of data points
    int   m,                            // number of random 2-dim ortho planes
    const int *index,                   // index of input data
    const float *a1,                    // lsh func
    const float *a2,                    // lsh func
    const DType *data)                  // input data
    : n_(n), d_(d), m_(m), index_(index), a1_(a1), a2_(a2), data_(data)
{
    // build index
    tables_ = new Result[(u64) m*n];
    for (int i = 0; i < m; ++i) {
        Result *table = tables_ + (u64) i*n;
        const float *tmp_a1 = a1_ + i*d;
        const float *tmp_a2 = a2_ + i*d;
        
        for (int j = 0; j < n; ++j) {
            const DType *point = data + (u64) j*d;
            table[j].id_  = j;
            table[j].key_ = calc_data_hash(tmp_a1, tmp_a2, point);
        }
        qsort(table, n, sizeof(Result), ResultComp);
    }
}

// -----------------------------------------------------------------------------
template<class DType>
float Ortho_LSH_Lite<DType>::calc_data_hash(// calc hash value for data point
    const float *a1,                    // lsh func
    const float *a2,                    // lsh func
    const DType *data)                  // input data point
{
    float p1 = calc_inner_product2<DType>(d_, data, a1);
    float p2 = calc_inner_product2<DType>(d_, data, a2);

    return p1 / p2;
}

// -----------------------------------------------------------------------------
template<class DType>
Ortho_LSH_Lite<DType>::~Ortho_LSH_Lite()// destructor
{
    if (tables_ != nullptr) { delete[] tables_; tables_ = nullptr; }
}

// -----------------------------------------------------------------------------
template<class DType>
void Ortho_LSH_Lite<DType>::display()// display parameters
{
    printf("Parameters of Ortho_LSH_Lite:\n");
    printf("n = %d\n", n_);
    printf("d = %d\n", d_);
    printf("m = %d\n", m_);
    printf("\n");
}

// -----------------------------------------------------------------------------
template<class DType>
int Ortho_LSH_Lite<DType>::nns(     // point-to-hyperplane nns (assist)
    int   l,                            // collision threshold
    int   cand,                         // number of candidates
    const float *query,                 // input query
    const float *q_val,                 // hash values of query
    MinK_List *list)                    // top-k results (return) 
{

    alloc(); // init parameters
    q_val_ = q_val;
    
    // use the largest proj dist as bucket width
    float width = init_position();
    
    // dynamic collision counting
    int cand_cnt = dcc(l, cand, width, query, list);
    free();

    return cand_cnt;
}

// -----------------------------------------------------------------------------
template<class DType>
void Ortho_LSH_Lite<DType>::alloc() // alloc space for assistant parameters
{
    freq_    = new int[n_];  
    checked_ = new bool[n_]; 
    flag_    = new bool[m_]; 
    l_pos_   = new int[m_];
    r_pos_   = new int[m_];

    memset(freq_,    0,     sizeof(int)*n_);
    memset(checked_, false, sizeof(bool)*n_);
    memset(flag_,    true,  sizeof(bool)*m_);
}

// -----------------------------------------------------------------------------
template<class DType>
float Ortho_LSH_Lite<DType>::init_position()// init left/right positions
{
    float  width = MINREAL, p_dist = -1.0f;
    Result tmp;
    for (int i = 0; i < m_; ++i) {
        float q_v = q_val_[i]; tmp.key_ = q_v;
        
        Result *table = tables_ + (u64) i*n_;
        int pos = std::lower_bound(table, table+n_, tmp, cmp) - table;
        if (pos <= 0) {
            l_pos_[i] = -1;  r_pos_[i] = 0;
            
            p_dist = fabs(q_v - table[0].key_);
            if (width < p_dist) width = p_dist;
        }
        else if (pos >= n_-1) {
            l_pos_[i] = n_-1; r_pos_[i] = n_;
            
            p_dist = fabs(q_v - table[n_-1].key_);
            if (width < p_dist) width = p_dist;
        }
        else {
            l_pos_[i] = pos; r_pos_[i] = pos + 1;

            p_dist = q_v - table[pos].key_;
            if (width < p_dist) width = p_dist;
            
            p_dist = table[pos+1].key_ - q_v;
            if (width < p_dist) width = p_dist;
        }
        
    }
    return width;
}

// -----------------------------------------------------------------------------
template<class DType>
int Ortho_LSH_Lite<DType>::dcc(     // dynamic collision counting
    int   l,                            // collision threshold
    int   cand,                         // candidate size
    float width,                        // bucket width
    const float *query,                 // input query
    MinK_List *list)                    // top-k results (return)
{
    int   cand_cnt = 0; // number of candidates for validation so far
    int   num_flag, cnt, pos, id;
    float q_v, dist, ldist, rdist;
    
    Result *table = nullptr;
    while (true) {
        // step 1: initialize the stop condition for current round
        num_flag = 0; memset(flag_, true, sizeof(bool)*m_);
        
        // step 2: (R,c)-NN search
        while (num_flag < m_) {
            for (int i = 0; i < m_; ++i) {
                if (!flag_[i]) continue;
                
                table = tables_ + (u64) i*n_;
                q_v = q_val_[i]; ldist = rdist = -1.0f;
                
                // step 2.1: scan the left part of hash table
                cnt = 0; pos = l_pos_[i];
                while (cnt < SCAN_SIZE) {
                    ldist = MAXREAL;
                    if (pos >= 0) ldist = fabs(q_v - table[pos].key_);
                    else break;
                    if (ldist > width) break;

                    id = table[pos].id_; ++freq_[id];
                    if (freq_[id] >= l && !checked_[id]) {
                        checked_[id] = true;
                        const DType *point = data_ + (u64) id*d_;
                        dist = calc_p2h_dist<DType>(d_, point, query);
                        list->insert(dist, index_[id]+1);
                        if (++cand_cnt >= cand) break;
                    }
                    --pos; ++cnt;
                }
                if (cand_cnt >= cand) break;
                l_pos_[i] = pos;

                // step 2.2: scan right part of hash table
                cnt = 0; pos = r_pos_[i];
                while (cnt < SCAN_SIZE) {
                    rdist = MAXREAL;
                    if (pos < n_) rdist = fabs(q_v - table[pos].key_);
                    else break;
                    if (rdist > width) break;
                    
                    id = table[pos].id_; ++freq_[id];
                    if (freq_[id] >= l && !checked_[id]) {
                        checked_[id] = true;
                        const DType *point = data_ + (u64) id*d_;
                        dist = calc_p2h_dist<DType>(d_, point, query);
                        list->insert(dist, index_[id]+1);
                        if (++cand_cnt >= cand) break;
                    }
                    ++pos; ++cnt;
                }
                if (cand_cnt >= cand) break;
                r_pos_[i] = pos;

                // step 2.3: check whether this width is finished scanned
                if (ldist > width && rdist > width) {
                    flag_[i] = false;
                    if (++num_flag >= m_) break;
                }
            }
            if (num_flag >= m_ || cand_cnt >= cand) break;
        }
        // step 3: stop condition
        if (cand_cnt >= cand) break;

        // step 4: auto-update search width
        width *= 2.0f;
    }
    return cand_cnt;
}

// -----------------------------------------------------------------------------
template<class DType>
void Ortho_LSH_Lite<DType>::free()  // free space for assistant parameters
{
    delete[] freq_;    freq_    = nullptr;
    delete[] l_pos_;   l_pos_   = nullptr;
    delete[] r_pos_;   r_pos_   = nullptr;
    delete[] checked_; checked_ = nullptr;
    delete[] flag_;    flag_    = nullptr;
}

} // end namespace p2h

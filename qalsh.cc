#include "qalsh.h"
#include <cstdint>

namespace p2h {

// -----------------------------------------------------------------------------
QALSH::QALSH(                       // constructor
    int n,                              // cardinality
    int d,                              // dimensionality
    int m)                              // number of hash tables
    : n_(n), dim_(d), m_(m)
{
    // generate hash functions
    a_ = new float[m*d];
    for (int i = 0; i < m*d; ++i) a_[i] = gaussian(0.0f, 1.0f);

    // allocate space for hash tables <tables_>
    tables_ = new Result[(u64) m*n];
}

// -----------------------------------------------------------------------------
QALSH::~QALSH()                     // destructor
{
    delete[] a_;
    delete[] tables_;
}

// -----------------------------------------------------------------------------
float QALSH::calc_hash_value(       // calc hash value
    int   d,                            // dimension for calc hash value
    int   tid,                          // hash table id
    const float *data)                  // one data/query object
{
    return calc_inner_product<float>(d, a_+tid*dim_, data);
}

// -----------------------------------------------------------------------------
float QALSH::calc_hash_value(       // calc hash value
    int   d,                            // dimension for calc hash value
    int   tid,                          // hash table id
    const Result *data)                 // sample data
{
    const float *a = a_ + tid*dim_;
    float val = 0.0f;
    for (int i = 0; i < d; ++i) {
        int idx = data[i].id_;
        val += a[idx] * data[i].key_;
    }
    return val;
}

// -----------------------------------------------------------------------------
int QALSH::nns(                     // nearest neighbor search
    int   l,                            // collision threshold
    int   cand,                         // number of candidates
    const float *query,                 // input query object
    std::vector<int> &cand_list)        // candidates (return)
{
    cand_list.clear();
    cand = std::min(cand, n_);

    // simply check all data if #candidates is equal to the cardinality
    if (cand == n_) {
        cand_list.resize(n_);
        for (int i = 0; i < n_; ++i) cand_list[i] = i;
        return n_;
    }

    // dynamic collsion counting
    init(query);
    int cand_cnt = dynamic_collsion_counting(l, cand, cand_list);
    free();

    return cand_cnt;
}

// -----------------------------------------------------------------------------
int QALSH::nns(                     // nearest neighbor search
    int   l,                            // collision threshold
    int   cand,                         // number of candidates
    int   sample_dim,                   // sample dimension
    const Result *query,                // query object
    std::vector<int> &cand_list)        // candidates (return)
{
    cand_list.clear();
    cand = std::min(cand, n_);
    
    // simply check all data if #candidates is equal to the cardinality
    if (cand == n_) {
        cand_list.resize(n_);
        for (int i = 0; i < n_; ++i) cand_list[i] = i;
        return n_;
    }

    // dynamic collsion counting
    init(sample_dim, query);
    int cand_cnt = dynamic_collsion_counting(l, cand, cand_list);
    free();

    return cand_cnt;
}

// -----------------------------------------------------------------------------
void QALSH::init(                   // init assistant parameters
    const float *query)                 // query object
{
    alloc();
    for (int i = 0; i < m_; ++i) {
        q_val_[i] = calc_hash_value(dim_, i, query);
    }
    init_position();
}

// -----------------------------------------------------------------------------
void QALSH::init(                   // init assistant parameters
    int   sample_dim,                   // sample dimension
    const Result *query)                // query object
{
    alloc();
    for (int i = 0; i < m_; ++i) {
        q_val_[i] = calc_hash_value(sample_dim, i, query);
    }
    init_position();
}

// -----------------------------------------------------------------------------
void QALSH::alloc()                 // alloc space for assistant parameters
{
    freq_    = new int[n_];  
    checked_ = new bool[n_]; 
    flag_    = new bool[m_]; 
    l_pos_   = new int[m_];
    r_pos_   = new int[m_];
    q_val_   = new float[m_];

    memset(freq_,    0,     sizeof(int)*n_);
    memset(checked_, false, sizeof(bool)*n_);
    memset(flag_,    true,  sizeof(bool)*m_);
}

// -----------------------------------------------------------------------------
void QALSH::init_position()         // init left/right positions
{
    Result tmp;
    Result *table = nullptr;
    for (int i = 0; i < m_; ++i) {
        tmp.key_ = q_val_[i];
        table = tables_ + (u64) i*n_;
        
        int pos = std::lower_bound(table, table+n_, tmp, cmp) - table;
        if (pos <= 0) { 
            l_pos_[i] = -1;  r_pos_[i] = 0;
        } 
        else if (pos >= n_-1) {
            l_pos_[i] = n_-1; r_pos_[i] = n_;
        }
        else { 
            l_pos_[i] = pos; r_pos_[i] = pos + 1;
        }
    }
}

// -----------------------------------------------------------------------------
int QALSH::dynamic_collsion_counting(// dynamic collision counting
    int   l,                            // collision threshold
    int   cand,                         // number of candidates
    std::vector<int> &cand_list)        // candidates (return)
{
    int   cand_cnt = 0; // candidate counter
    int   num_flag, cnt, pos, id;

    float w = 1.0f, radius = 1.0f; // grid width & search radius
    float width = radius * w / 2.0f; // bucket width
    float q_v, ldist, rdist;
    
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
                        cand_list.push_back(id);
                        if (++cand_cnt >= cand) break;
                    }
                    --pos; ++cnt;
                }
                if (cand_cnt >= cand) break;
                l_pos_[i] = pos;

                // step 2.2: scan the right part of hash table
                cnt = 0; pos = r_pos_[i];
                while (cnt < SCAN_SIZE) {
                    rdist = MAXREAL;
                    if (pos < n_) rdist = fabs(q_v - table[pos].key_);
                    else break;
                    if (rdist > width) break;

                    id = table[pos].id_; ++freq_[id];
                    if (freq_[id] >= l && !checked_[id]) {
                        checked_[id] = true;
                        cand_list.push_back(id);
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

        // step 4: auto-update radius
        radius = radius * 2.0f;
        width  = radius * w / 2.0f;
    }
    return cand_cnt;
}

// -----------------------------------------------------------------------------
void QALSH::free()                  // free space for assistant parameters
{
    delete[] freq_;    freq_    = nullptr;
    delete[] l_pos_;   l_pos_   = nullptr;
    delete[] r_pos_;   r_pos_   = nullptr;
    delete[] checked_; checked_ = nullptr;
    delete[] flag_;    flag_    = nullptr;
    delete[] q_val_;   q_val_   = nullptr;
}

} // end namespace p2h

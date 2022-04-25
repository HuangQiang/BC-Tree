#include "rqalsh.h"
#include <cstdint>

namespace p2h {

// -----------------------------------------------------------------------------
RQALSH::RQALSH(                     // constructor
    int   n,                            // number of data objects
    int   d,                            // dimension of data objects
    int   m,                            // #hash tables
    const int *index)
    : n_(n), dim_(d), m_(m), index_(index), a_(nullptr), tables_(nullptr)
{
    if (n_ > N_PTS) {
        // generate hash functions
        a_ = new float[m*d];
        for (int i = 0; i < m*d; ++i) a_[i] = gaussian(0.0f, 1.0f);
        
        // allocate space for tables
        tables_ = new Result[(u64) m*n];
    }
}

// -----------------------------------------------------------------------------
RQALSH::~RQALSH()                   // destructor
{
    if (a_      != nullptr) { delete[] a_;      a_      = nullptr; }
    if (tables_ != nullptr) { delete[] tables_; tables_ = nullptr; }
}

// -----------------------------------------------------------------------------
float RQALSH::calc_hash_value(      // calc hash value
    int   d,                            // dimension for calc hash value
    int   tid,                          // hash table ida
    const float *data)                  // input data
{
    const float *a = a_ + tid*dim_;
    return calc_inner_product<float>(d, data, a);
}

// -----------------------------------------------------------------------------
float RQALSH::calc_hash_value(      // calc hash value
    int   d,                            // dimension for calc hash value
    int   tid,                          // hash table id
    float last,                         // the last coordinate of input data
    const Result *data)                 // input data
{
    const float *a = a_ + tid*dim_;
    float val = 0.0f;
    for (int i = 0; i < d; ++i) {
        int idx = data[i].id_;
        val += a[idx] * data[i].key_;
    }
    return val + a[dim_-1] * last;
}

// -----------------------------------------------------------------------------
int RQALSH::fns(                    // furthest neighbor search
    int   l,                            // separation threshold
    int   cand,                         // #candidates
    float R,                            // limited search range
    const float *query,                 // query object
    std::vector<int> &cand_list)        // candidates (return)
{
    cand_list.clear();
    
    // simply check all data if #candidates is equal to the cardinality
    if (n_ <= N_PTS || cand >= n_) {
        cand_list.resize(n_);
        for (int i = 0; i < n_; ++i) cand_list[i] = index_[i];
        return n_;
    }

    // dynamic separation counting
    init(query);
    int cand_cnt = dynamic_separation_counting(l, cand, R, cand_list);
    free();

    return cand_cnt;
}

// -----------------------------------------------------------------------------
int RQALSH::fns(                    // furthest neighbor search
    int   l,                            // separation threshold
    int   cand,                         // #candidates
    float R,                            // limited search range
    int   sample_dim,                   // sample dimension
    const Result *query,                // query object
    std::vector<int> &cand_list)        // candidates (return)
{
    cand_list.clear();

    // simply check all data if #candidates is equal to the cardinality
    if (n_ <= N_PTS || cand >= n_) {
        cand_list.resize(n_);
        for (int i = 0; i < n_; ++i) cand_list[i] = index_[i];
        return n_;
    }

    // dynamic separation counting
    init(sample_dim, query);
    int cand_cnt = dynamic_separation_counting(l, cand, R, cand_list);
    free();

    return cand_cnt;
}

// -----------------------------------------------------------------------------
void RQALSH::init(                  // init assistant parameters
    const float *query)                 // query object
{
    alloc();
    for (int i = 0; i < m_; ++i) {
        q_val_[i] = calc_hash_value(dim_, i, query);
        l_pos_[i] = 0;  
        r_pos_[i] = n_-1;
    }
}

// -----------------------------------------------------------------------------
void RQALSH::init(                  // init assistant parameters
    int   sample_dim,                   // sample dimension
    const Result *query)                // query object
{
    alloc();
    for (int i = 0; i < m_; ++i) {
        q_val_[i] = calc_hash_value(sample_dim, i, 0.0f, query);
        l_pos_[i] = 0;
        r_pos_[i] = n_-1;
    }
}

// -----------------------------------------------------------------------------
void RQALSH::alloc()                // alloc space for assistant parameters
{
    freq_    = new int[n_];
    checked_ = new bool[n_];
    b_flag_  = new bool[m_];
    r_flag_  = new bool[m_];
    l_pos_   = new int[m_];
    r_pos_   = new int[m_];
    q_val_   = new float[m_];

    memset(freq_,    0,     sizeof(int)*n_);
    memset(checked_, false, sizeof(bool)*n_);
    memset(b_flag_,  true,  sizeof(bool)*m_);
    memset(r_flag_,  true,  sizeof(bool)*m_);
}

// -----------------------------------------------------------------------------
int RQALSH::dynamic_separation_counting(// dynamic separation counting
    int   l,                            // separation threshold
    int   cand,                         // number of candidates
    float R,                            // limited search range
    std::vector<int> &cand_list)        // candidates (return)
{
    int   cand_cnt  = 0; // candidate counter
    int   num_range = 0; // number of search range flag
    int   num_bucket, cnt, lpos, rpos, id;

    float w      = 1.0f; // grid width
    float radius = find_radius(w); // search radius
    float width  = radius * w / 2.0f; // bucket width
    float range  = R < CHECK_ERROR ? 0.0f : R*w/2.0f; // search range
    float q_v, ldist, rdist;

    Result *table = nullptr;
    while (true) {
        // step 1: initialization
        num_bucket = 0; memset(b_flag_, true, sizeof(bool)*m_);

        // step 2: (R,c)-FN search
        while (num_bucket < m_ && num_range < m_) {
            for (int j = 0; j < m_; ++j) {
                // CANNOT add !r_flag_[j] as condition, because the
                // r_flag_[j] for large radius will affect small radius
                if (!b_flag_[j]) continue;

                table = tables_ + (u64) j*n_;
                q_v = q_val_[j]; ldist = rdist = -1.0f;
                
                // step 2.1: scan left part of bucket
                cnt = 0; lpos = l_pos_[j]; rpos = r_pos_[j];
                while (cnt < SCAN_SIZE) {
                    ldist = MINREAL;
                    if (lpos < rpos) ldist = fabs(q_v-table[lpos].key_);
                    else break;
                    if (ldist < width || ldist < range) break;

                    id = table[lpos].id_;
                    if (++freq_[id] >= l && !checked_[id]) {
                        checked_[id] = true;
                        cand_list.push_back(index_[id]);
                        if (++cand_cnt >= cand) break;
                    }
                    ++lpos; ++cnt;
                }
                if (cand_cnt >= cand) break;
                l_pos_[j] = lpos;

                // step 2.2: scan right part of bucket
                cnt = 0;
                while (cnt < SCAN_SIZE) {
                    rdist = MINREAL;
                    if (lpos < rpos) rdist = fabs(q_v-table[rpos].key_);
                    else break;
                    if (rdist < width || rdist < range) break;

                    id = table[rpos].id_;
                    if (++freq_[id] >= l && !checked_[id]) {
                        checked_[id] = true;
                        cand_list.push_back(index_[id]);
                        if (++cand_cnt >= cand) break;
                    }
                    --rpos; ++cnt;
                }
                if (cand_cnt >= cand) break;
                r_pos_[j] = rpos;

                // step 2.3: check whether this bucket is finished scanned
                if (lpos >= rpos || (ldist < width && rdist < width)) {
                    if (b_flag_[j]) { b_flag_[j] = false; ++num_bucket; }
                }
                if (lpos >= rpos || (ldist < range && rdist < range)) {
                    if (b_flag_[j]) { b_flag_[j] = false; ++num_bucket; }
                    if (r_flag_[j]) { r_flag_[j] = false; ++num_range;  }
                }
                // use break after checking both b_flag_ and r_flag_
                if (num_bucket >= m_ || num_range >= m_) break;
            }
            if (num_bucket >= m_ || num_range >= m_) break;
            if (cand_cnt >= cand) break;
        }
        // step 3: stop condition
        if (num_range >= m_ || cand_cnt >= cand) break;

        // step 4: update radius
        radius = radius / 2.0f;
        width  = radius * w / 2.0f;
    }
    return cand_cnt;
}

// -----------------------------------------------------------------------------
float RQALSH::find_radius(          // find proper radius
    float w)                            // grid width
{
    // find projected distance closest to the query in each hash tables 
    std::vector<float> list;
    for (int i = 0; i < m_; ++i) {
        int lpos = l_pos_[i], rpos = r_pos_[i];
        if (lpos < rpos) {
            float q_v = q_val_[i];
            list.push_back(fabs(tables_[(u64)i*n_+lpos].key_-q_v));
            list.push_back(fabs(tables_[(u64)i*n_+rpos].key_-q_v));
        }
    }
    // sort the array in ascending order 
    std::sort(list.begin(), list.end());

    // find the median distance and return the new radius
    int   num  = (int) list.size();
    float dist = -1.0f;
    if (num % 2 == 0) {
        dist = (list[num/2-1] + list[num/2]) / 2.0f;
    } else {
        dist = list[num/2];
    }
    int kappa = (int) ceil(log(2.0f*dist/w) / log(2.0f));
    return pow(2.0f, kappa);
}

// -----------------------------------------------------------------------------
void RQALSH::free()                    // free space for assistant parameters
{
    delete[] freq_;    freq_    = nullptr;
    delete[] l_pos_;   l_pos_   = nullptr;
    delete[] r_pos_;   r_pos_   = nullptr;
    delete[] checked_; checked_ = nullptr;
    delete[] b_flag_;  b_flag_  = nullptr;
    delete[] r_flag_;  r_flag_  = nullptr;
    delete[] q_val_;   q_val_   = nullptr;
}

} // end namespace p2h

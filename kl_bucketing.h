#pragma once

#include <vector>
#include <random>

#include "def.h"
#include "util.h"
#include "pri_queue.h"

namespace p2h {

// -----------------------------------------------------------------------------
template<class UINTT>
struct CountMarkerU {
    std::vector<UINTT> mark_count_;
    UINTT cur_cnt_;
    
    CountMarkerU(int sz = 0) : mark_count_(sz), cur_cnt_(1) {}
    ~CountMarkerU() { mark_count_.clear(); mark_count_.shrink_to_fit(); }

    // -------------------------------------------------------------------------
    void resize(int n) {
        mark_count_.resize(n);
        fill(mark_count_.begin(), mark_count_.end(), 0);
    }

    // -------------------------------------------------------------------------
    void mark(int n) { mark_count_[n] = cur_cnt_; }
    
    // -------------------------------------------------------------------------
    bool is_marked(int n) { return mark_count_[n] >= cur_cnt_; }

    // -------------------------------------------------------------------------
    void clear() {
        if (cur_cnt_ == ~UINTT(0)) { cur_cnt_ = 1; mark_count_.clear(); }
        else { ++cur_cnt_; }
    }

    // -------------------------------------------------------------------------
    uint64_t get_memory_usage() {
        return sizeof(UINTT) * (mark_count_.capacity() + 1);
    }
};
using CountMarker = CountMarkerU<unsigned>;

// -----------------------------------------------------------------------------
template<class SigType=int64_t>
class KLBucketingFlat {
public:
    int n_pts_;
    int l_;
    int64_t mask_;
    CountMarker cm_;
    std::vector<std::vector<std::vector<int> > > buckets_;

    // -------------------------------------------------------------------------
    KLBucketingFlat(int n, int l) : n_pts_(n), l_(l), buckets_(l), cm_(n) {
        mask_ = 1;
        while (mask_ < n) mask_ <<= 1;
        --mask_;
        // using pertubation hash to combine hash signatures
        for (int i = 0; i < l; ++i) buckets_[i].resize(mask_ + 1);
    };

    // -------------------------------------------------------------------------
    void insert(int i, const std::vector<SigType> &dcode) {
        for (int j = 0; j < l_; ++j) {
            SigType hashcode32 = dcode[j] & mask_;
            // if (i==0) printf("j=%d, dcode=%d, mask=%d\n", j,hashcode32,mask_);
            buckets_[j][hashcode32].push_back(i);
            int r = rand() % buckets_[j][hashcode32].size();
            
            std::swap(buckets_[j][hashcode32][r], buckets_[j][hashcode32].back());
        }
    }

    // -------------------------------------------------------------------------
    template<typename F>
    void for_cand(int cand, const std::vector<SigType> &qcode, const F& f) {
        cm_.clear();
        for (int j = 0; j < l_; ++j) {
            SigType hashcode32 = qcode[j] & mask_;
            // printf("j=%d, qcode=%u, cand=%d\n", j, hashcode32, cand);
            for (int idx : buckets_[j][hashcode32]) {
                if (!cm_.is_marked(idx)) {
                    f(idx);
                    cm_.mark(idx);
                    if (!--cand) return;
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    uint64_t get_memory_usage() {
        uint64_t ret = 0;
        ret += sizeof(*this);
        // buckets_
        ret += sizeof(std::vector<std::vector<int> >) * buckets_.capacity();
        ret += sizeof(std::vector<int>) * buckets_.size() * (mask_ + 1);
        for (size_t i = 0; i < buckets_.size(); ++i) {
            for (size_t j = 0; j < buckets_[i].size(); ++j) {
                ret += buckets_[i][j].capacity()*sizeof(int);
            }
        }
        ret += cm_.get_memory_usage(); // cm_
        return ret;
    }
};

} // end namespace p2h

#pragma once

#include <vector>
#include <algorithm>
#include <cassert>
#include <queue>
#include <set>
#include <cinttypes>
#include <array>
#include <functional>
#include <unordered_map>
#include "../myndarray.h"

namespace mylccs {

// -----------------------------------------------------------------------------
//  LCS by sort
// -----------------------------------------------------------------------------
template<class uintt>
struct CountMarkerU {
    std::vector<uintt> markCount;
    uintt curCnt;
    CountMarkerU(int sz=0) : markCount(sz), curCnt(1) {}

    // -------------------------------------------------------------------------
    void resize(int n) {
        markCount.resize(n);
        fill(markCount.begin(), markCount.end(), 0);
    }

    // -------------------------------------------------------------------------
    void mark(int n) { markCount[n] = curCnt; }

    // -------------------------------------------------------------------------
    bool isMarked(int n) { return markCount[n] >= curCnt; }
    
    // -------------------------------------------------------------------------
    void clear() {
        if (curCnt == ~uintt(0)) { curCnt=1; markCount.clear(); } 
        else ++curCnt;
    }
};

using CountMarker = CountMarkerU<unsigned>;

// -----------------------------------------------------------------------------
class LCCS_SORT_INT {
public:
    struct DataIdx {
        int32_t idx;
        DataIdx() {}
        DataIdx(int32_t i): idx(i){}
    };

    int dim;
    int nPts;
    int step;
    int nSearchLoc;
    int logn;
    int32_t **datap;

    std::vector<std::vector<DataIdx> > sorted_idx;
    std::vector<std::vector<int32_t> > next_link;

    std::vector<int> res_curidx;
    std::vector<int> res_lowlen;
    std::vector<int> res_highlen;
    CountMarker checked;

    // -------------------------------------------------------------------------
    LCCS_SORT_INT(int dim, int step) : dim(dim), step(step) {
        // assert(dim%64 == 0);
        nSearchLoc = ((dim-1) / step) + 1;

        res_curidx.resize(dim);
        res_lowlen.resize(dim);
        res_highlen.resize(dim);
    }

    // -------------------------------------------------------------------------
    void build(NDArray<2, int32_t> &data) {
        datap = data.to_ptr();
        // nPts = datap->size();
        nPts = data.lens[0];
        checked.resize(nPts);

        init_sorted_idx();
        init_next_link();
    }

    // -------------------------------------------------------------------------
    inline const int32_t* get_datap(int qloc, int idx) {
        return datap[sorted_idx[qloc][idx].idx];
    }

    // -------------------------------------------------------------------------
    inline const int32_t* get_datap(const DataIdx& idx) {
        // return &datap->at(idx.idx)[0];
        return datap[(idx.idx)];
    }

    // -------------------------------------------------------------------------
    inline int getidx(const DataIdx& idx) { return idx.idx; }

    // -------------------------------------------------------------------------
    inline int getidx(int qloc, int idx) { return sorted_idx[qloc][idx].idx; }

    // -------------------------------------------------------------------------
    inline int dim64() { return dim / 64; }

    // -------------------------------------------------------------------------
    void init_sorted_idx() { // bucket-sort leveraging extents
        // int TableSize = (1<<logn)+1;
        sorted_idx.resize(dim);
        for (int d = 0; d < dim; ++d) {
            sorted_idx[d].resize(nPts);
            for (int i = 0; i < nPts; ++i) {
                sorted_idx[d][i] = {i};
            }
            
            const auto& cmpf = [d, this](const DataIdx& a, const DataIdx& b) -> bool {
                auto [len, isLess] = match_util(a, b, d, 0);
                return isLess;
            };
            sort(sorted_idx[d].begin(), sorted_idx[d].end(), cmpf);
        }
    }

    // -------------------------------------------------------------------------
    void init_next_link() {
        std::unordered_map<int, int> nextMap;

        next_link.resize(dim);
        for (int d = dim-1; d >= 0; --d) {
            nextMap.clear();
            next_link[d].resize(nPts);
            int nextd = (d+step) % dim;
            for (int i = 0; i < nPts; ++i){
                // nextMap[sorted_idx[nextd][i]] = i;
                nextMap[getidx(nextd, i)] = i;
            }

            for (int i = 0; i < nPts; ++i){
                // int dataidxi = sorted_idx[d][i];
                int dataidxi = getidx(d, i);
                next_link[d][i] = nextMap[dataidxi];
            }
        }
    }

    // -------------------------------------------------------------------------
    //  lowidx, lowlen, highlen
    //  assume the length of matching of low and high are lowlen and
    // -------------------------------------------------------------------------
    std::tuple<int, int, int> get_loc_scan(
        const int32_t *query, 
        int   loc, 
        int   low, 
        int   lowlen, 
        int   high, 
        int   highlen)
    {
        int lastlen = lowlen;
        int minlen = std::min(lowlen, highlen);
        for (int i = low+1; i < high; ++i) {
            const int32_t *dpi = get_datap(loc, i);
            auto [ilen, iless] = match_util(query, dpi, loc, minlen);

            if (iless) return std::make_tuple(i-1, lastlen, ilen);
            lastlen = ilen;
        }
        // reach the end
        return std::make_tuple(high-1, lastlen, highlen);
    }

    // -------------------------------------------------------------------------
    //  binary search with linear when interval is small
    //  return (lowidx, lowlen, highlen) that query \in [lowidx, highidx) 
    //         or lowidx == 0 or highidx == N-1
    // -------------------------------------------------------------------------
    std::tuple<int, int, int> get_loc_mixed(
        const int32_t *query, 
        int   qloc, 
        int   low, 
        int   lowlen, 
        int   high, 
        int   highlen)
    {
        static const int SCAN_SIZE = 4;
        // const auto& sorted_idx_qloc = sorted_idx[qloc];

        // return datap->at(idx[qloc][a]) < query
        const auto& cmp = [&](int a, int match=0) -> std::tuple<int, bool> {
            const int32_t* dpa = get_datap(qloc, a);
            return match_util(query, dpa, qloc, match);
        };

        while (low < high - SCAN_SIZE) {
            int curlen = std::min(lowlen, highlen);
            // binary search
            int mid = (low + high) / 2;
            auto [midlen, midisless] = cmp(mid, curlen);
            // f(qloc, mid, midlen, midisless);
            
            // if query < mid:
            if (midisless) {
                // which means mid < query
                high = mid; highlen = midlen;
            } else {
                low = mid; lowlen = midlen;
            }
        }
        return get_loc_scan(query, qloc, low, lowlen, high, highlen);
    }

    // -------------------------------------------------------------------------
    //  return (lowidx, lowlen, highlen) that q \in [lowidx, highidx) 
    //         or lowidx == 0 or highidx == N-1
    // -------------------------------------------------------------------------
    std::tuple<int, int, int> get_loc(const int32_t* query, int qloc) {
        int   low = 0;
        int   high = nPts-1;
        const int32_t* dplow  = get_datap(qloc, low);
        const int32_t* dphigh = get_datap(qloc, high);
        
        auto [lowlen, lowisless]   = match_util(query, dplow, qloc, 0);
        auto [highlen, highisless] = match_util(query, dphigh, qloc, 0);

        return get_loc_mixed(query, qloc, low, lowlen, high, highlen);
    }

    // -------------------------------------------------------------------------
    //  match_util :: [sigs] a-> [sigs] b->int loc-> (int, bool)
    //  return matchlen and bool(x<y)
    // -------------------------------------------------------------------------
    std::tuple<int, bool> match_util(
        const int32_t *x, 
        const int32_t *y, 
        int   loc, 
        int   match) 
    {
        //to make sure the order is correct
        if (match > 0) --match;
        
        int matchloc = (loc + match) % dim;
        int restlen  = dim - match;
        for (int i = 0; i < restlen; ++i) {
            int idx = (matchloc + i) % dim;
            int x_i = x[idx];
            int y_i = y[idx];
            if (x_i!=y_i) return std::make_tuple(match, x_i < y_i);
            ++match;
        }
        // equal
        match = dim;
        return std::make_tuple(match, false);
    } 

    // -------------------------------------------------------------------------
    std::tuple<int, bool> match_util(
        const DataIdx &x,
        const DataIdx &y,
        int   loc,
        int   match)
    {
        //to make sure the order is correct
        const int32_t* xp = datap[x.idx];
        const int32_t* yp = datap[y.idx];

        return match_util(xp, yp, loc, match);
    }

    // -------------------------------------------------------------------------
    uint64_t get_memory_usage() {
        uint64_t ret = sizeof(*this);

        ret += sizeof(uint32_t) * int64_t(nPts); // checked
        ret += sizeof(std::vector<DataIdx>)*sorted_idx.capacity() + 
               sizeof(DataIdx)*int64_t(nPts)*dim;   // sorted_idx
        ret += sizeof(std::vector<int32_t>) * next_link.capacity() + 
               sizeof(int32_t)*int64_t(nPts)*dim;   // next
        
        return ret;
    }

    // -------------------------------------------------------------------------
    //  this function will set res_curidx, res_lowlen and res_highlen
    // -------------------------------------------------------------------------
    void find_matched_loc(
        const std::vector<int32_t> &query, 
        std::vector<int32_t> &res_curidx, 
        std::vector<int32_t> &res_lowlen, 
        std::vector<int32_t> &res_highlen)
    {
        const int32_t* queryp = (const int32_t*)&query[0];
        res_curidx.resize(dim);
        res_lowlen.resize(dim);
        res_highlen.resize(dim);

        // binary search
        auto [curidx, lowlen, highlen] = get_loc(queryp, 0);
        // store the res for multi-probe lsh
        res_curidx[0]  = curidx;
        res_lowlen[0]  = lowlen;
        res_highlen[0] = highlen;

        for (int i = 1; i < nSearchLoc; ++i) {
            int d       = i * step;
            int lowidx  = next_link[d-step][curidx];
            int highidx = next_link[d-step][curidx+1];

            if (lowlen < step) { lowlen = 0; lowidx = 0; }
            else if (lowlen != dim) { lowlen -= step; }

            if (highlen < step) { highlen = 0; highidx = nPts-1; }
            else if (highlen != dim) { highlen -= step; }
            
            // printf("lidx, llen, hidx, hlen=%d, %d, %d, %d\n", lowidx, 
            //     lowlen, highidx, highlen);

            std::tie(curidx, lowlen, highlen) = get_loc_mixed(queryp, d, 
                lowidx, lowlen, highidx, highlen);
            // store the res for multi-probe lsh
            res_curidx[d] = curidx;
            res_lowlen[d] = lowlen;
            res_highlen[d] = highlen;
        }
    }

    // -------------------------------------------------------------------------
    //  simple scan strategy (high data locality)
    // -------------------------------------------------------------------------
    template<typename F> 
    void for_candidates_by_scan(
        int nScanStep, 
        const std::vector<int32_t> &query, 
        const F &f)
    {
        find_matched_loc(query, res_curidx, res_lowlen, res_highlen);

        checked.clear();
        const auto tryCheck = [&](int dataidx) {
            if (!checked.isMarked(dataidx)) {
                checked.mark(dataidx);
                f(dataidx);
            }
        };
        const auto tryCheckLoc = [&](int curidx, int d) {
            for (int i = curidx; i >= 0 && curidx-i < nScanStep; --i) {
                int matchingIdx = getidx(d, i);
                tryCheck(matchingIdx);
            }
            for (int i = curidx+1; i < nPts && i-curidx-1 < nScanStep; ++i) {
                int matchingIdx = getidx(d, i);
                tryCheck(matchingIdx);
            }
        };

        for (int i = 0; i < dim; ++i) tryCheckLoc(res_curidx[i], i);
    }

    // -------------------------------------------------------------------------
    struct CandidateLoc {
        int qloc, idx, len, dir;
        
        CandidateLoc(int qloc, int idx, int len, int dir) : qloc(qloc), 
            idx(idx), len(len), dir(dir) {}

        bool operator<(const CandidateLoc& a) const {
            return len < a.len;
        }
    };

    // -------------------------------------------------------------------------
    template<typename F> 
    void for_candidates_by_heap(
        int nCandidates, 
        const std::vector<int32_t> &query, 
        const F &f)
    {
        const int32_t *queryp = (const int32_t*) &query[0];
        find_matched_loc(query, res_curidx, res_lowlen, res_highlen);

        std::vector<CandidateLoc> pool;
        pool.reserve(nSearchLoc*2 + nCandidates*nSearchLoc);
        std::priority_queue<CandidateLoc, std::vector<CandidateLoc>> candQue(
            std::less<CandidateLoc>(), std::move(pool));

        int checkCnt = 0;
        checked.clear();
        const auto& tryCheck = [&](int dataidx) {
            if (!checked.isMarked(dataidx)) {
                checked.mark(dataidx);
                f(dataidx);

                ++checkCnt;
            }
        };

        for (int i = 0; i < dim; ++i) {
            candQue.emplace(i, res_curidx[i],   res_lowlen[i], -1);
            candQue.emplace(i, res_curidx[i]+1, res_highlen[i], 1);
        }

        while(!candQue.empty() && checkCnt<nCandidates){
            const auto& curCand = candQue.top();
            // printf("%d, %d, %d, %d\n", curCand.qloc, curCand.len, 
            //     curCand.idx, curCand.dir);

            int matchingIdx = getidx(curCand.qloc, curCand.idx);
            tryCheck(matchingIdx);
            if (curCand.idx+curCand.dir>=0 && curCand.idx+curCand.dir<nPts) {
                int  newidx = curCand.idx + curCand.dir;
                int  len;
                bool isless;
                if (curCand.len == dim) {
                    std::tie(len, isless) = match_util(queryp, 
                        get_datap(curCand.qloc, newidx), curCand.qloc, 0);
                } 
                else {
                    // heuristic, use curLen-1 to estimate newlen and avoid 
                    // to re-compute the matched_length
                    len = curCand.len - 1;
                }
                
                // full heap
                // auto [len, isless] = match_util(queryp, get_datap(curCand.qloc,
                //     newidx), curCand.qloc, 0);
                candQue.emplace(curCand.qloc, newidx, len, curCand.dir);
            }
            candQue.pop();
        }
    }

    // -------------------------------------------------------------------------
    template<typename F>
    void for_candidates(
        int nScanStep, 
        const std::vector<int32_t> &query, 
        const F &f) 
    {
        for_candidates_by_scan(nScanStep, query, f);
        // for_candidates_by_heap(nScanStep*dim, query, f);
    }
    
    // -------------------------------------------------------------------------
    //  check between start and end only
    //  will not clear checked, so this should be called only after 
    //  for_candidates is called. 
    // -------------------------------------------------------------------------
    template<typename F>
    void for_candidates_between(
        int   start, 
        int   end, 
        int   nCandidates, 
        const std::vector<int32_t> &query, 
        const F &f) 
    {
        const int32_t* queryp = (const int32_t*)&query[0];

        // std::vector<CandidateLoc> pool;
        // pool.reserve(nSearchLoc * 2 + nCandidates*nSearchLoc);
        // std::priority_queue<CandidateLoc, std::vector<CandidateLoc>> candidates(
        //     std::less<CandidateLoc>() , std::move(pool));

        int startLoc = start /step * step % dim;
        int endLoc   = end   /step * step % dim;
        if (startLoc == endLoc) return; 
        
        // checked.clear();
        int checkCounter = 0;
        // call the callback function if not checked and increase the counter
        const auto& tryCheck = [&](int dataidx) {
            if (!checked.isMarked(dataidx)) {
                checked.mark(dataidx);
                f(dataidx);
                ++checkCounter;
            }
        };

        const auto& tryCheckLoc = [&](int curidx, int d) {
            for (int i = curidx; i >= 0 && curidx-i < nCandidates; --i) {
                int matchingIdx = getidx(d, i);
                tryCheck(matchingIdx);
            }
            for (int i = curidx+1; i < nPts && i-curidx-1 < nCandidates; ++i) {
                int matchingIdx = getidx(d, i);
                tryCheck(matchingIdx);
            }
        };

        // the information is kept before
        auto [curidx, lowlen, highlen] = get_loc(queryp, startLoc);
        // int curidx  = res_curidx[startLoc];
        // int lowlen  = 0;
        // int highlen = 0;
        tryCheckLoc(curidx, startLoc);

        for (int i = 1; i < nSearchLoc; ++i) {
            int d = (startLoc + i*step) % dim;
            if (d == endLoc) return;

            int prevd   = (d-step+dim) % dim;
            int lowidx  = next_link[prevd][curidx];
            int highidx = next_link[prevd][curidx+1];

            if (lowlen < step) { lowlen = 0; lowidx = 0; }
            else if (lowlen != dim) { lowlen -= step; }

            if (highlen < step) { highlen = 0; highidx = nPts-1; }
            else if (highlen != dim) { highlen -= step; }
            
            std::tie(curidx, lowlen, highlen) = get_loc_mixed(queryp, d, 
                lowidx, lowlen, highidx, highlen);
            // printf("qloc=%d, curidx, llen, hlen=%d, %d, %d\n", d, curidx,
            //     lowlen, highlen);
            tryCheckLoc(curidx, d);
        }
    }
};

} // end namespace mylccs

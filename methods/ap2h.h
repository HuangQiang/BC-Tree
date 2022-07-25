#pragma once

#include <cstdint>
#include "fh.h"
#include "nh.h"
#include "nh_dcf.h"
#include "ball_tree.h"
#include "bc_tree.h"

namespace p2h {

// -----------------------------------------------------------------------------
template<class DType>
int linear_scan(                    // Linear Scan
    int   n,                            // number of data  points
    int   qn,                           // number of query points
    int   d,                            // dimension of space
    const char *method_name,            // name of method
    const char *path,                   // output path
    const DType *data,                  // data set
    const float *query,                 // query set
    const Result *R)                    // truth set
{
    char fname[200]; sprintf(fname, "%s%s.out", path, method_name);
    FILE *fp = fopen(fname, "w");
    if (!fp) { printf("Could not create %s\n", fname); return 1; }

    // -------------------------------------------------------------------------
    //  Point-to-Hyperplane NNS
    // -------------------------------------------------------------------------
    fprintf(fp, "%s:\n", method_name);

    head(method_name);
    for (int top_k : TOPKs) {
        init_global_metric();
        gettimeofday(&g_start_time, nullptr);
        
        MinK_List *list = new MinK_List(top_k);
        for (int i = 0; i < qn; ++i) {
            const float *q = query + i*d;
            list->reset();
            for (int j = 0; j < n; ++j) {
                float dist = calc_p2h_dist<DType>(d, data+(u64)j*d, q);
                list->insert(dist, j+1);
            }
            update_global_metric(top_k, n, n, R+i*MAXK, list);
        }
        delete list;
        gettimeofday(&g_end_time, nullptr);
        calc_and_write_global_metric(top_k, qn, fp);
    }
    foot(fp);
    fclose(fp);
    return 0;
}

// -----------------------------------------------------------------------------
template<class DType>
int ball_tree(                      // Ball-Tree
    int   n,                            // number of data  points
    int   qn,                           // number of query points
    int   d,                            // dimension of space
    int   leaf,                         // leaf size of ball-tree
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *path,                   // output folder
    const DType *data,                  // data set
    const float *query,                 // query set
    const Result *R)                    // truth set
{
    char fname[200]; sprintf(fname, "%s%s.out", path, method_name);
    FILE *fp = fopen(fname, "a+");
    if (!fp) { printf("Could not create %s\n", fname); return 1; }

    // -------------------------------------------------------------------------
    //  Preprocessing
    // -------------------------------------------------------------------------
    gettimeofday(&g_start_time, nullptr);
    Ball_Tree<DType> *tree = new Ball_Tree<DType>(n, d, leaf, data);
    tree->display();
    float memory = tree->get_memory_usage() / 1048576.0f;
    gettimeofday(&g_end_time, nullptr);
    
    fprintf(fp, "%s: leaf=%d\n", method_name, leaf);
    write_index_info(memory, fp);
    
    // -------------------------------------------------------------------------
    //  Point-to-Hyperplane NNS
    // -------------------------------------------------------------------------
    std::vector<int> cand_list;
    if (get_conf(conf_name, data_name, method_name, cand_list)) return 1;
    
    float c = 10.0f;
    for (int cand : cand_list) {
        write_params(cand, c, method_name, fp);
        for (int top_k : TOPKs) {
            init_global_metric();
            gettimeofday(&g_start_time, nullptr);
            
            MinK_List *list = new MinK_List(top_k);
            for (int i = 0; i < qn; ++i) {
                list->reset();
                int check_k = tree->nns(top_k, cand, c, query+i*d, list);
                update_global_metric(top_k, check_k, n, R+i*MAXK, list);
            }
            delete list;
            gettimeofday(&g_end_time, nullptr);
            calc_and_write_global_metric(top_k, qn, fp);
        }
        foot(fp);
    }
    fclose(fp);
    delete tree;
    return 0;
}

// -----------------------------------------------------------------------------
template<class DType>
int bc_tree(                        // BC-Tree
    int   n,                            // number of data  points
    int   qn,                           // number of query points
    int   d,                            // dimension of space
    int   leaf,                         // leaf size of ball-tree
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *path,                   // output folder
    const DType *data,                  // data set
    const float *query,                 // query set
    const Result *R)                    // truth set
{
    char fname[200]; sprintf(fname, "%s%s.out", path, method_name);
    FILE *fp = fopen(fname, "a+");
    if (!fp) { printf("Could not create %s\n", fname); return 1; }

    // -------------------------------------------------------------------------
    //  Preprocessing
    // -------------------------------------------------------------------------
    gettimeofday(&g_start_time, nullptr);
    BC_Tree<DType> *tree = new BC_Tree<DType>(n, d, leaf, data);
    tree->display();
    float memory = tree->get_memory_usage() / 1048576.0f;
    gettimeofday(&g_end_time, nullptr);
    
    fprintf(fp, "%s: leaf=%d\n", method_name, leaf);
    write_index_info(memory, fp);
    
    // -------------------------------------------------------------------------
    //  Point-to-Hyperplane NNS
    // -------------------------------------------------------------------------
    std::vector<int> cand_list;
    if (get_conf(conf_name, data_name, method_name, cand_list)) return 1;
    
    float c = 10.0f;
    for (int cand : cand_list) {
        write_params(cand, c, method_name, fp);
        for (int top_k : TOPKs) {
            init_global_metric();
            gettimeofday(&g_start_time, nullptr);
            
            MinK_List *list = new MinK_List(top_k);
            for (int i = 0; i < qn; ++i) {
                list->reset();
                int check_k = tree->nns(top_k, cand, c, query+i*d, list);
                update_global_metric(top_k, check_k, n, R+i*MAXK, list);
            }
            delete list;
            gettimeofday(&g_end_time, nullptr);
            calc_and_write_global_metric(top_k, qn, fp);
        }
        foot(fp);
    }
    fclose(fp);
    delete tree;
    return 0;
}

// -----------------------------------------------------------------------------
template<class DType>
int fh(                             // FH with Multi-Partition RQALSH
    int   n,                            // number of data  points
    int   qn,                           // number of query points
    int   d,                            // dimension of space
    int   m,                            // #hash tables
    int   s,                            // scale factor of dimension
    float b,                            // interval ratio
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *path,                   // output path
    const DType *data,                  // data set
    const float *query,                 // query set
    const Result *R)                    // truth set
{
    char fname[200]; sprintf(fname, "%s%s.out_s=%d", path, method_name, s);
    FILE *fp = fopen(fname, "a+");
    if (!fp) { printf("Could not create %s\n", fname); return 1; }

    // -------------------------------------------------------------------------
    //  Preprocessing
    // -------------------------------------------------------------------------
    gettimeofday(&g_start_time, nullptr);
    FH<DType> *lsh = new FH<DType>(n, d, m, s, b, data);
    lsh->display();
    float memory = lsh->get_memory_usage() / 1048576.0f;
    gettimeofday(&g_end_time, nullptr);
    
    fprintf(fp, "%s: m=%d, s=%d, b=%.2f\n", method_name, m, s, b);
    write_index_info(memory, fp);

    // -------------------------------------------------------------------------
    //  Point-to-Hyperplane NNS
    // -------------------------------------------------------------------------
    std::vector<int> cand_list;
    if (get_conf(conf_name, data_name, method_name, cand_list)) return 1;
    
    for (int l : Ls) {
        if (l >= m) continue;
        for (int cand : cand_list) {
            write_params(l, cand, method_name, fp);
            for (int top_k : TOPKs) {
                init_global_metric();
                gettimeofday(&g_start_time, nullptr);
                
                MinK_List *list = new MinK_List(top_k);
                for (int i = 0; i < qn; ++i) {
                    list->reset();
                    int check_k = lsh->nns(top_k, l, cand, query+i*d, list);
                    update_global_metric(top_k, check_k, n, R+i*MAXK, list);
                }
                delete list;
                gettimeofday(&g_end_time, nullptr);
                calc_and_write_global_metric(top_k, qn, fp);
            }
            foot(fp);
        }
    }
    fclose(fp);
    delete lsh;
    return 0;
}

// -----------------------------------------------------------------------------
template<class DType>
int nh(                             // NH with LCCS-LSH
    int   n,                            // number of data  points
    int   qn,                           // number of query points
    int   d,                            // dimension of space
    int   m,                            // #single hasher of the compond hasher
    int   s,                            // scale factor of dimension
    float w,                            // bucket width
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *path,                   // output path
    const DType *data,                  // data set
    const float *query,                 // query set
    const Result *R)                    // truth set
{
    char fname[200]; sprintf(fname, "%s%s.out", path, method_name);
    FILE *fp = fopen(fname, "a+");
    if (!fp) { printf("Could not create %s\n", fname); return 1; }

    // -------------------------------------------------------------------------
    //  Preprocessing
    // -------------------------------------------------------------------------
    gettimeofday(&g_start_time, nullptr);
    NH<DType> *lsh = new NH<DType>(n, d, m, s, w, data);
    lsh->display();
    float memory = lsh->get_memory_usage() / 1048576.0f;
    gettimeofday(&g_end_time, nullptr);

    fprintf(fp, "%s: m=%d, s=%d, w=%.2f\n", method_name, m, s, w);
    write_index_info(memory, fp);

    // -------------------------------------------------------------------------
    //  Point-to-Hyperplane NNS
    // -------------------------------------------------------------------------
    std::vector<int> cand_list;
    if (get_conf(conf_name, data_name, method_name, cand_list)) return 1;

    for (int cand : cand_list) {
        write_params(cand, method_name, fp);
        for (int top_k : TOPKs) {
            init_global_metric();
            gettimeofday(&g_start_time, nullptr);
            
            MinK_List *list = new MinK_List(top_k);
            for (int i = 0; i < qn; ++i) {
                list->reset();
                int check_k = lsh->nns(top_k, cand, query+i*d, list);
                update_global_metric(top_k, check_k, n, R+i*MAXK, list);
            }
            delete list;
            gettimeofday(&g_end_time, nullptr);
            calc_and_write_global_metric(top_k, qn, fp);
        }
        foot(fp);
    }
    fclose(fp);
    delete lsh;
    return 0;
}

// -----------------------------------------------------------------------------
template<class DType>
int nh_dcf(                         // NH with Dynamic Counting Framework
    int   n,                            // number of data  points
    int   qn,                           // number of query points
    int   d,                            // dimension of space
    int   m,                            // #hash tables
    int   s,                            // scale factor of dimension
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    const char *path,                   // output path
    const DType *data,                  // data set
    const float *query,                 // query set
    const Result *R)                    // truth set
{
    char fname[200]; sprintf(fname, "%s%s.out", path, method_name);
    FILE *fp = fopen(fname, "a+");
    if (!fp) { printf("Could not create %s\n", fname); return 1; }

    // -------------------------------------------------------------------------
    //  Preprocessing
    // -------------------------------------------------------------------------
    gettimeofday(&g_start_time, nullptr);
    NH_DCF<DType> *lsh = new NH_DCF<DType>(n, d, m, s, data);
    lsh->display();
    float memory = lsh->get_memory_usage() / 1048576.0f;
    gettimeofday(&g_end_time, nullptr);

    fprintf(fp, "%s: m=%d, s=%d\n", method_name, m, s);
    write_index_info(memory, fp);

    // -------------------------------------------------------------------------
    //  Point-to-Hyperplane NNS
    // -------------------------------------------------------------------------
    std::vector<int> cand_list;
    if (get_conf(conf_name, data_name, method_name, cand_list)) return 1;

    for (int l : Ls) {
        if (l >= m) continue;
        for (int cand : cand_list) {
            write_params(l, cand, method_name, fp);
            for (int top_k : TOPKs) {
                init_global_metric();
                gettimeofday(&g_start_time, nullptr);
                
                MinK_List *list = new MinK_List(top_k);
                for (int i = 0; i < qn; ++i) {
                    list->reset();
                    int check_k = lsh->nns(top_k, l, cand, query+i*d, list);
                    update_global_metric(top_k, check_k, n, R+i*MAXK, list);
                }
                delete list;
                gettimeofday(&g_end_time, nullptr);
                calc_and_write_global_metric(top_k, qn, fp);
            }
            foot(fp);
        }
    }
    fclose(fp);
    delete lsh;
    return 0;
}

} // end namespace p2h

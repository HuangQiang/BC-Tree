#include <iostream>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <cstring>
#include <sys/types.h>

#include "def.h"
#include "util.h"
#include "ap2h.h"

using namespace p2h;

// -----------------------------------------------------------------------------
void usage()                        // display the usage
{
    printf("\n"
        "-------------------------------------------------------------------\n"
        " The Parameters for Point-to-Hyperplane Nearest Neighbor Search    \n"
        "-------------------------------------------------------------------\n"
        " -alg   {integer}  options of algorithms\n"
        " -n     {integer}  cardinality of the dataset\n"
        " -qn    {integer}  number of queries\n"
        " -d     {integer}  dimensionality of the dataset\n"
        " -m     {integer}  #hash tables (FH, NH)\n"
        " -leaf  {integer}  leaf size ratio (Ball_Tree)\n"
        " -s     {integer}  scale factor of dimension (FH, NH)\n"
        " -b     {float}    interval ratio (FH)\n"
        " -w     {float}    bucket width (NH)\n"
        " -cf    {string}   name of configuration\n"
        " -dt    {string}   data type\n"
        " -dn    {string}   data name\n"
        " -ds    {string}   address of data  set\n"
        " -qs    {string}   address of query set\n"
        " -ts    {string}   address of truth set\n"
        " -op    {string}   output folder\n"
        "\n"
        "-------------------------------------------------------------------\n"
        " Primary Options of Algorithms                                     \n"
        "-------------------------------------------------------------------\n"
        " 0  - Ground Truth & Histogram & Heatmap\n"
        "      Param: -alg 0 -n -qn -d -dt -dn -ds -qs -ts -op\n"
        "\n"
        " 1  - Linear_Scan\n"
        "      Param: -alg 1 -n -qn -d -dt -dn -ds -qs -ts -op\n"
        "\n"
        " 2  - Ball_Tree\n"
        "      Param: -alg 2 -n -qn -d -leaf -cf -dt -dn -ds -qs -ts -op\n"
        "\n"
        " 3  - BC_Tree\n"
        "      Param: -alg 3 -n -qn -d -leaf -cf -dt -dn -ds -qs -ts -op\n"
        "\n"
        " 4  - FH (Furthest Hyperpalne Hash)\n"
        "      Param: -alg 4 -n -qn -d -m -s -b -cf -dt -dn -ds -qs -ts -op\n"
        "\n"
        " 5  - NH (Nearest Hyperpalne Hash with LCCS-LSH)\n"
        "      Param: -alg 5 -n -qn -d -m -s -w -cf -dt -dn -ds -qs -ts -op\n"
        "\n"
        "-------------------------------------------------------------------\n"
        " Author: Qiang Huang (huangq@comp.nus.edu.sg)                      \n"
        "-------------------------------------------------------------------\n"
        "\n\n\n");
}

// -----------------------------------------------------------------------------
template<class DType>
void impl(                          // implementation interface
    int   alg,                          // which algorithm?
    int   n,                            // number of data points
    int   qn,                           // number of query hyperplanes
    int   d,                            // dimensionality
    int   leaf,                         // leaf size of ball-tree/bc-tree
    int   m,                            // #tables (FH,NH)
    int   s,                            // scale factor of dim (s>=1) (FH,NH)
    float b,                            // interval ratio (0 < b < 1) (FH)
    float w,                            // bucket width (NH)
    const char *conf_name,              // configuration name
    const char *data_name,              // data name
    const char *data_set,               // address of data  set
    const char *query_set,              // address of query set
    const char *truth_set,              // address of truth set
    const char *path)                   // output path
{
    // -------------------------------------------------------------------------
    //  read data set and query set
    // -------------------------------------------------------------------------
    gettimeofday(&g_start_time, nullptr);
    DType *data = new DType[(u64) n*d];
    if (read_bin_data<DType>(n, d, data_set, data)) exit(1);

    float *query = new float[qn*d];
    if (read_bin_query<DType>(qn, d, query_set, query)) exit(1);

    Result *R = nullptr; // k-NN ground truth
    if (alg > 0) {
        R = new Result[qn*MAXK];
        if (read_ground_truth(qn, truth_set, R)) exit(1);
    }
    gettimeofday(&g_end_time, nullptr);
    float running_time = g_end_time.tv_sec - g_start_time.tv_sec + 
        (g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;
    printf("Read Data & Query: %f Seconds\n\n", running_time);

    // -------------------------------------------------------------------------
    //  methods
    // -------------------------------------------------------------------------
    switch (alg) {
    case 0:
        ground_truth<DType>(n, qn, d, truth_set, path, (const DType*) data, 
            (const float*) query);
        break;
    case 1:
        linear_scan<DType>(n, qn, d, "Linear_Scan", path, (const DType*) data,
            (const float*) query, (const Result*) R);
        break;
    case 2:
        ball_tree<DType>(n, qn, d, leaf, conf_name, data_name, "Ball_Tree", path, 
            (const DType*) data, (const float*) query, (const Result*) R);
        break;
    case 3:
        bc_tree<DType>(n, qn, d, leaf, conf_name, data_name, "BC_Tree", path, 
            (const DType*) data, (const float*) query, (const Result*) R);
        break;
    case 4:
        fh<DType>(n, qn, d, m, s, b, conf_name, data_name, "FH", path, 
            (const DType*) data, (const float*) query, (const Result*) R);
        break;
    case 5:
        nh<DType>(n, qn, d, m, s, w, conf_name, data_name, "NH", path, 
            (const DType*) data, (const float*) query, (const Result*) R);
        break;
    default:
        printf("Parameters error!\n"); usage();
        break;
    }
    // -------------------------------------------------------------------------
    //  release space
    // -------------------------------------------------------------------------
    delete[] data;
    delete[] query;
    if (alg > 0) delete[] R;
}

// -----------------------------------------------------------------------------
int main(int nargs, char **args)
{
    srand(RANDOM_SEED);             // use a fixed random seed

    int   cnt  = 1;
    int   alg  = -1;                // which algorithm?
    int   n    = -1;                // cardinality
    int   qn   = -1;                // query number
    int   d    = -1;                // dimensionality
    int   leaf = -1;                // leaf size of ball-tree/bc-tree
    int   m    = -1;                // #tables (FH,NH)
    int   s    = -1;                // scale factor of dim (s>=1) (FH,NH)
    float b    = -1.0f;             // interval ratio (0 < b < 1) (FH)
    float w    = -1.0f;             // bucket width (NH)
    char  conf_name[200];           // configuration name
    char  data_type[20];            // data type
    char  data_name[200];           // data set name
    char  data_set[200];            // address of data set
    char  query_set[200];           // address of query set
    char  truth_set[200];           // address of ground truth file
    char  path[200];                // output path
    
    while (cnt < nargs) {
        if (strcmp(args[cnt], "-alg") == 0) {
            alg = atoi(args[++cnt]); assert(alg >= 0);
            printf("alg       = %d\n", alg);
        }
        else if (strcmp(args[cnt], "-n") == 0) {
            n = atoi(args[++cnt]); assert(n > 0);
            printf("n         = %d\n", n);
        }
        else if (strcmp(args[cnt], "-qn") == 0) {
            qn = atoi(args[++cnt]); assert(qn > 0);
            printf("qn        = %d\n", qn);
        }
        else if (strcmp(args[cnt], "-d") == 0) {
            d = atoi(args[++cnt]); assert(d > 1);
            printf("d         = %d\n", d);
        }
        else if (strcmp(args[cnt], "-m") == 0) {
            m = atoi(args[++cnt]); assert(m > 0);
            printf("m         = %d\n", m);
        }
        else if (strcmp(args[cnt], "-s") == 0) {
            s = atoi(args[++cnt]); assert(s > 0);
            printf("s         = %d\n", s);
        }
        else if (strcmp(args[cnt], "-leaf") == 0) {
            leaf = atoi(args[++cnt]); assert(leaf > 0);
            printf("leaf      = %d\n", leaf);
        }
        else if (strcmp(args[cnt], "-b") == 0) {
            b = atof(args[++cnt]); assert(b > 0.0f && b < 1.0f);
            printf("b         = %.2f\n", b);
        }
        else if (strcmp(args[cnt], "-w") == 0) {
            w = atof(args[++cnt]); assert(w > 0.0f);
            printf("w         = %.2f\n", w);
        }
        else if (strcmp(args[cnt], "-cf") == 0) {
            strncpy(conf_name, args[++cnt], sizeof(conf_name));
            printf("conf_name = %s\n", conf_name);
        }
        else if (strcmp(args[cnt], "-dt") == 0) {
            strncpy(data_type, args[++cnt], sizeof(data_type));
            printf("data_type = %s\n", data_type);
        }
        else if (strcmp(args[cnt], "-dn") == 0) {
            strncpy(data_name, args[++cnt], sizeof(data_name));
            printf("data_name = %s\n", data_name);
        }
        else if (strcmp(args[cnt], "-ds") == 0) {
            strncpy(data_set, args[++cnt], sizeof(data_set));
            printf("data_set  = %s\n", data_set);
        }
        else if (strcmp(args[cnt], "-qs") == 0) {
            strncpy(query_set, args[++cnt], sizeof(query_set));
            printf("query_set = %s\n", query_set);
        }
        else if (strcmp(args[cnt], "-ts") == 0) {
            strncpy(truth_set, args[++cnt], sizeof(truth_set));
            printf("truth_set = %s\n", truth_set);
        }
        else if (strcmp(args[cnt], "-op") == 0) {
            strncpy(path, args[++cnt], sizeof(path));
            printf("path      = %s\n", path);
            create_dir(path);
        }
        else {
            usage(); exit(1);
        }
        ++cnt;
    }
    printf("\n");

    // -------------------------------------------------------------------------
    //  methods
    // -------------------------------------------------------------------------
    if (strcmp(data_type, "int32") == 0) {
        impl<int>(alg, n, qn, d, leaf, m, s, b, w, conf_name, data_name, 
            data_set, query_set, truth_set, path);
    }
    else if (strcmp(data_type, "float32") == 0) {
        impl<float>(alg, n, qn, d, leaf, m, s, b, w, conf_name, data_name, 
            data_set, query_set, truth_set, path);
    }
    else {
        printf("Parameters error!\n"); usage();
    }
    return 0;
}

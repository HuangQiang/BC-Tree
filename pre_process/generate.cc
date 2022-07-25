#include <cstdint>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

#include <unistd.h>
#include <stdarg.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>

//                      no limit         one-side limit   two-side limit
const float  MIN_DIST = 0.000000001F; // 0.000000001F; // 0.00005F;
const float  MAX_DIST = 1000000.0F;   // 0.1F;         // 0.1F;
const float  MAX_NORM = 1.0F;
const double PI       = 3.141592654;

typedef uint64_t u64;

timeval g_start_time;
timeval g_end_time;

// -----------------------------------------------------------------------------
void create_dir(                    // create directory
    char *path)                         // input path
{
    int len = (int) strlen(path);
    for (int i = 0; i < len; ++i) {
        if (path[i] != '/') continue;
        
        char ch = path[i+1]; path[i+1] = '\0';
        if (access(path, F_OK) != 0) {
            if (mkdir(path, 0755) != 0) { 
                printf("Could not create %s\n", path); exit(1);
            }
        }
        path[i+1] = ch;
    }
}

// -----------------------------------------------------------------------------
int read_bin_data(                  // read bin data from disk
    int   n,                            // number of data points
    int   d,                            // data dimension
    const char *fname,                  // address of data set
    float &M,                           // max l2-norm
    float *data)                        // data (return)
{
    gettimeofday(&g_start_time, NULL);

    // read bin data
    FILE *fp = fopen(fname, "rb");
    if (!fp) { printf("Could not open %s\n", fname); return 1; }
    int i = 0;
    while (!feof(fp) && i < n) {
        u64 shift = (u64)i*(d+1);
        fread(&data[shift], sizeof(float), d, fp);
        data[shift+d] = 1.0f;
        ++i;
    }
    assert(i == n);
    fclose(fp);

    // find the minimum & maximum coordinates for d dimensions
    float *min_coord = new float[d];
    float *max_coord = new float[d];
    for (int i = 0; i < n; ++i) {
        const float *point = data + (u64)i*(d+1);
        for (int j = 0; j < d; ++j) {
            if (i == 0 || point[j] < min_coord[j]) min_coord[j] = point[j];
            if (i == 0 || point[j] > max_coord[j]) max_coord[j] = point[j];
        }
    }
    // calc the data center
    float *center = new float[d];
    for (int i = 0; i < d; ++i) center[i] = (min_coord[i]+max_coord[i])/2.0f;

    // shift the data by the center & find the max l2-norm to the center
    M = -1.0f;
    for (int i = 0; i < n; ++i) {
        float *point = data + (u64)i*(d+1);
        // shift the data by the center
        float norm = 0.0f;
        for (int j = 0; j < d; ++j) {
            float val = point[j] - center[j];
            point[j] = val; norm += val * val;
        }
        norm = sqrt(norm);
        // find the max l2-norm to the center
        if (M < norm) M = norm;
    }
    delete[] center;
    delete[] min_coord;
    delete[] max_coord;

    gettimeofday(&g_end_time, NULL);
    float running_time = g_end_time.tv_sec - g_start_time.tv_sec + 
        (g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;
    printf("Read Data: %f Seconds\n\n", running_time);
    
    return 0;
}

// -----------------------------------------------------------------------------
int read_bin_data(                  // read bin data from disk
    int   n,                            // number of data points
    int   d,                            // data dimension
    const char *fname,                  // address of data set
    float *data)                        // data (return)
{
    gettimeofday(&g_start_time, NULL);

    // read bin data
    FILE *fp = fopen(fname, "rb");
    if (!fp) { printf("Could not open %s\n", fname); return 1; }
    int i = 0;
    while (!feof(fp) && i < n) {
        u64 shift = (u64)i*(d+1);
        fread(&data[shift], sizeof(float), d, fp);
        data[shift+d] = 1.0f;
        ++i;
    }
    assert(i == n);
    fclose(fp);

    // find the minimum & maximum coordinates for d dimensions
    float *min_coord = new float[d];
    float *max_coord = new float[d];
    for (int i = 0; i < n; ++i) {
        const float *point = data + (u64)i*(d+1);
        for (int j = 0; j < d; ++j) {
            if (i == 0 || point[j] < min_coord[j]) min_coord[j] = point[j];
            if (i == 0 || point[j] > max_coord[j]) max_coord[j] = point[j];
        }
    }
    // calc the data center
    float *center = new float[d];
    for (int i = 0; i < d; ++i) center[i] = (min_coord[i]+max_coord[i])/2.0f;

    // shift the data by the center & find the max l2-norm to the center
    float M = -1.0f;
    for (int i = 0; i < n; ++i) {
        float *point = data + (u64)i*(d+1);
        // shift the data by the center
        float norm = 0.0f;
        for (int j = 0; j < d; ++j) {
            float val = point[j] - center[j];
            point[j] = val; norm += val * val;
        }
        norm = sqrt(norm);
        // find the max l2-norm to the center
        if (M < norm) M = norm;
    }
    
    // rescale the data by the max l2-norm
    float lambda = MAX_NORM / M;
    for (int i = 0; i < n; ++i) {
        float *point = data + (u64)i*(d+1);
        for (int j = 0; j < d; ++j) point[j] *= lambda;
    }
    delete[] center;
    delete[] min_coord;
    delete[] max_coord;

    gettimeofday(&g_end_time, NULL);
    float running_time = g_end_time.tv_sec - g_start_time.tv_sec + 
        (g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;
    printf("Read Data: %f Seconds\n\n", running_time);
    
    return 0;
}

// -----------------------------------------------------------------------------
int write_bin_data(                 // write binary data to disk
    int   n,                            // number of data points
    int   d,                            // data dimension + 1
    bool  sign,                         // data or query
    char  *fname,                       // output file name
    const float *data)                  // output data
{
    gettimeofday(&g_start_time, NULL);

    //  write binary data
    FILE *fp = fopen(fname, "wb");
    if (!fp) { printf("Could not open %s\n", fname); return 1; }
    fwrite(data, sizeof(float), (u64) n*d, fp);
    fclose(fp);

    gettimeofday(&g_end_time, NULL);
    float running_time = g_end_time.tv_sec - g_start_time.tv_sec + 
        (g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;
    printf("Write Bin %s: %f Seconds\n\n", sign?"Data":"Query", running_time);
    
    return 0;
}

// -----------------------------------------------------------------------------
int write_txt_data(                 // write text data to disk
    int   n,                            // number of data points
    int   d,                            // data dimension + 1
    bool  sign,                         // data or query
    char  *fname,                       // output file name
    const float *data)                  // output data
{
    gettimeofday(&g_start_time, NULL);

    // write text data
    std::ofstream fp;
    fp.open(fname, std::ios::trunc);
    for (int i = 0; i < n; ++i) {
        const float *point = data + (u64)i*d;
        fp << i+1;
        for (int j = 0; j < d; ++j) fp << " " << point[j];
        fp << "\n";
    }
    fp.close();

    gettimeofday(&g_end_time, NULL);
    float running_time = g_end_time.tv_sec - g_start_time.tv_sec + 
        (g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;
    printf("Write Text %s: %f Seconds\n\n", sign?"Data":"Query", running_time);

    return 0;
}

// -----------------------------------------------------------------------------
float uniform(                      // r.v. from Uniform(min, max)
    float min,                          // min value
    float max)                          // max value
{
    int   num  = rand();
    float base = (float) RAND_MAX - 1.0F;
    float frac = ((float) num) / base;

    return (max - min) * frac + min;
}

// -----------------------------------------------------------------------------
//  use Box-Muller to generate a random variable from Gaussian(mean, sigma)
//  when mean = 0 and sigma = 1, it is standard Gaussian distr, i.e., Gaussian(0,1)
// -----------------------------------------------------------------------------
float gaussian(                     // r.v. from N(mean, sigma)
    float mu,                           // mean (location)
    float sigma)                        // stanard deviation (scale > 0)
{
    // assert(sigma > 0.0f);
    float u1 = -1.0f;
    float u2 = -1.0f;
    do {
        u1 = uniform(0.0f, 1.0f);
    } while (u1 < 0.000001f);
    u2 = uniform(0.0f, 1.0f);
    
    return mu + sigma * sqrt(-2.0f * log(u1)) * cos(2.0f * PI * u2);
    // return mu + sigma * sqrt(-2.0f * log(u1)) * sin(2.0f * PI * u2);
}

// -----------------------------------------------------------------------------
float calc_dot_plane(               // calc dist from a point to a hyperplane
    int   d,                            // dimension of input data object
    const float *data,                  // input data object
    const float *query)                 // input query
{
    float ip = 0.0f, norm = 0.0f;
    for (int i = 0; i < d; ++i) {
        ip   += data[i]  * query[i];
        norm += query[i] * query[i];
    }
    ip += query[d];

    return fabs(ip) / sqrt(norm);
}

// -----------------------------------------------------------------------------
void generate_query(                // generate query
    int   n,                            // number of data points
    int   qn,                           // number of hyperplane queries
    int   d,                            // data dimension + 1
    float max_norm,                     // max l2-norm
    const char *fname,                  // file name
    const float *data,                  // input data
    float *query)                       // query (return)
{
    gettimeofday(&g_start_time, NULL);
    
    // generate query set
    float *p = new float[d+2]; // random pivot (only consider d dimensions)
    int   total_cnt = 0;
    float total_ip  = 0.0f;
    float max_min_dist = -1.0f;
    for (int i = 0; i < qn; ++i) {
        float *q = query + (u64)i*(d+1);
        int   cnt = 0;
        float min_dist = -1.0f, ip = -1.0f;
        do {
            // generate a normal vector w as q[0:d-1]
            float norm = 0.0f;
            for (int j = 0; j < d; ++j) {
                q[j] = gaussian(0.0f, 1.0f); norm += q[j]*q[j];
            }
            norm = sqrt(norm);
            for (int j = 0; j < d; ++j) q[j] /= norm;
            
            // generate a random pivot from the d-ball as a point confined 
            // on the hyperplane
            norm = 0.0f;
            for (int j = 0; j < d+2; ++j) {
                p[j] = gaussian(0.0f, 1.0f); norm += p[j]*p[j];
            }
            norm = sqrt(norm) / max_norm;
            
            // determine the last coordinate based on p and w
            ip = 0.0f;
            for (int j = 0; j < d; ++j) ip += p[j]*q[j]/norm;
            q[d] = -ip;

            // calc the minimum distance for this query
            min_dist = -1.0f;
            for (int j = 0; j < n; ++j) {
                float dist = calc_dot_plane(d, data+(u64)j*(d+1), q);
                if (min_dist < 0 || dist < min_dist) min_dist = dist;
            }
            printf("%s %3d: cnt=%d, min_dist=%.9f, ip=%f\n", fname, i+1, ++cnt, min_dist, ip);
        } while (min_dist < MIN_DIST || min_dist > MAX_DIST);
        
        total_cnt += cnt;
        total_ip  += fabs(ip);
        if (min_dist > max_min_dist) max_min_dist = min_dist;
    }
    delete[] p;
    
    gettimeofday(&g_end_time, NULL);
    float runtime = g_end_time.tv_sec - g_start_time.tv_sec + 
        (g_end_time.tv_usec - g_start_time.tv_usec) / 1000000.0f;
    printf("\nGenerate Query: %f Seconds\n", runtime);
    printf("cnt=%.2f, max_norm = %.9f, last=%f, max_min_dist=%f\n\n", 
        (float) total_cnt/qn, max_norm, total_ip/qn, max_min_dist);
}

// -----------------------------------------------------------------------------
int main(int nargs, char** args)
{
    srand(666); // set up random seed 

    // read parameters
    int  n    = atoi(args[1]); // cardinality
    int  d    = atoi(args[2]); // dimensionality
    int  qn   = atoi(args[3]); // number of queries
    int  orig = atoi(args[4]);
    char infile[200]; strncpy(infile, args[5], sizeof(infile));
    char folder[200]; strncpy(folder, args[6], sizeof(folder));
    create_dir(folder);

    printf("n           = %d\n", n);
    printf("d           = %d\n", d);
    printf("qn          = %d\n", qn);
    printf("orig        = %d\n", orig);
    printf("input file  = %s\n", infile);
    printf("out folder  = %s\n", folder);
    printf("\n");

    // read dataset
    float *data = new float[(u64) n*(d+1)];
    float M = -1.0f;
    if (orig == 1) {
        read_bin_data(n, d, infile, M, data);
    } else {
        read_bin_data(n, d, infile, data);
    }
    
    // write data to disk
    char bin_fname[200]; sprintf(bin_fname, "%s.ds", folder);
    char txt_fname[200]; sprintf(txt_fname, "%s.dt", folder);
    write_bin_data(n, d+1, true, bin_fname, data);
    // write_txt_data(n, d+1, true, txt_fname, data);
    
    // generate query
    float *query = new float[(u64) qn*(d+1)];
    if (orig == 1) {
        generate_query(n, qn, d, M, infile, data, query);
    } else {
        generate_query(n, qn, d, MAX_NORM, infile, data, query);
    }
    
    // write query to disk
    sprintf(bin_fname, "%s.q",  folder);
    sprintf(txt_fname, "%s.qt", folder);
    write_bin_data(qn, d+1, false, bin_fname, query);
    write_txt_data(qn, d+1, false, txt_fname, query);
    
    // for (int i = 0; i < d+1; ++i) {
    //     printf("%f ", data[i]);
    // }
    // printf("\n\n");
    
    // release space
    delete[] data;
    delete[] query;

    return 0;
}

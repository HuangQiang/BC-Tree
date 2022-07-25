#include "util.h"

namespace p2h {
    
timeval g_start_time;               // global param: start time
timeval g_end_time;                 // global param: end time

float g_memory    = -1.0f;          // global param: memory usage (megabytes)
float g_indextime = -1.0f;          // global param: indexing time (seconds)

float g_runtime   = -1.0f;          // global param: running time (ms)
float g_ratio     = -1.0f;          // global param: overall ratio
float g_recall    = -1.0f;          // global param: recall (%)
float g_precision = -1.0f;          // global param: precision (%)
float g_fraction  = -1.0f;          // global param: fraction (%)

// -----------------------------------------------------------------------------
void create_dir(                    // create dir if the path not exists
    char *path)                         // input path
{
    int len = (int) strlen(path);
    for (int i = 0; i < len; ++i) {
        if (path[i] != '/') continue; 
        
        char ch = path[i+1]; path[i+1] = '\0';
        if (access(path, F_OK) != 0) { // create directory if not exists
            if (mkdir(path, 0755) != 0) {
                printf("Could not create directory %s\n", path); exit(1);
            }
        }
        path[i+1] = ch;
    }
}

// -----------------------------------------------------------------------------
int read_ground_truth(              // read ground truth results from disk
    int qn,                             // number of query objects
    const char *fname,                  // address of truth set
    Result *R)                          // ground truth results (return)
{
    FILE *fp = fopen(fname, "r");
    if (!fp) { printf("Could not open %s\n", fname); return 1; }

    int tmp1 = -1, tmp2 = -1;
    fscanf(fp, "%d,%d\n", &tmp1, &tmp2); assert(tmp1==qn && tmp2==MAXK);

    for (int i = 0; i < qn; ++i) {
        fscanf(fp, "%d", &tmp1);
        for (int j = 0; j < MAXK; ++j) {
            fscanf(fp, ",%d,%f", &R[i*MAXK+j].id_, &R[i*MAXK+j].key_);
        }
        fscanf(fp, "\n");
    }
    fclose(fp);
    return 0;
}

// -----------------------------------------------------------------------------
void get_csv_from_line(             // get an array with csv format from a line
    std::string str_data,               // a string line
    std::vector<int> &csv_data)         // csv data (return)
{
    csv_data.clear();

    std::istringstream ss(str_data);
    while (ss) {
        std::string s;
        if (!getline(ss, s, ',')) break;
        csv_data.push_back(stoi(s));
    }
}

// -----------------------------------------------------------------------------
int get_conf(                       // get cand list from configuration file
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    std::vector<int> &cand)             // candidates list (return)
{
    std::ifstream infile(conf_name);
    if (!infile) { printf("Could not open %s\n", conf_name); return 1; }

    std::string dname, mname, tmp;
    bool stop = false;
    while (infile) {
        getline(infile, dname);
        
        // // skip the first three methods
        // getline(infile, mname); getline(infile, tmp); getline(infile, tmp);
        // getline(infile, mname); getline(infile, tmp); getline(infile, tmp);
        // getline(infile, mname); getline(infile, tmp); getline(infile, tmp);

        // check the remaining methods
        while (true) {
            getline(infile, mname);
            if (mname.length() == 0) break;

            if ((dname.compare(data_name)==0) && (mname.compare(method_name)==0)) {
                getline(infile, tmp); get_csv_from_line(tmp, cand);
                stop = true; break;
            } else {
                getline(infile, tmp);
            }
        }
        if (stop) break;
    }
    infile.close();
    return 0;
}

// -----------------------------------------------------------------------------
int get_conf(                       // get nc and cand list from config file
    const char *conf_name,              // name of configuration
    const char *data_name,              // name of dataset
    const char *method_name,            // name of method
    std::vector<int> &l,                // a list of separation threshold (return)
    std::vector<int> &cand)             // a list of #candidates (return)
{
    std::ifstream infile(conf_name);
    if (!infile) { printf("Could not open %s\n", conf_name); return 1; }

    std::string dname, mname, tmp;
    while (infile) {
        getline(infile, dname);
        
        // check the 1st method
        getline(infile, mname);
        if ((dname.compare(data_name)==0) && (mname.compare(method_name)==0)) {
            getline(infile, tmp); get_csv_from_line(tmp, l);
            getline(infile, tmp); get_csv_from_line(tmp, cand);
            break;
        } else {
            getline(infile, tmp);
            getline(infile, tmp);
        }
        
        // check the 2nd method
        getline(infile, mname);
        if ((dname.compare(data_name)==0) && (mname.compare(method_name)==0)) {
            getline(infile, tmp); get_csv_from_line(tmp, l);
            getline(infile, tmp); get_csv_from_line(tmp, cand);
            break;
        } else {
            getline(infile, tmp);
            getline(infile, tmp);
        }

        // check the 3rd method
        getline(infile, mname);
        if ((dname.compare(data_name)==0) && (mname.compare(method_name)==0)) {
            getline(infile, tmp); get_csv_from_line(tmp, l);
            getline(infile, tmp); get_csv_from_line(tmp, cand);
            break;
        } else {
            getline(infile, tmp);
            getline(infile, tmp);
        }

        // skip the remaining methods
        while (true) {
            getline(infile, mname);
            if (mname.length() == 0) break;
            else getline(infile, tmp); // skip the setting of this method
        }
        if (dname.length() == 0 && mname.length() == 0) break;
    }
    infile.close();
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
//  Given a mean and a standard deviation, gaussian generates a normally 
//  distributed random number.
//
//  Algorithm:  Polar Method, p.104, Knuth, vol. 2
// -----------------------------------------------------------------------------
float gaussian(                     // r.v. from Gaussian(mean, sigma)
    float mean,                         // mean value
    float sigma)                        // std value
{
    float v1 = -1.0f, v2 = -1.0f, s = -1.0f, x = -1.0f;
    do {
        v1 = 2.0f*uniform(0.0f, 1.0f) - 1.0f;
        v2 = 2.0f*uniform(0.0f, 1.0f) - 1.0f;
        s = v1*v1 + v2*v2;
    } while (s >= 1.0f);
    x = v1 * sqrt(-2.0f*log(s) / s);

    // x is distributed from N(0, 1)
    return x * sigma + mean;
}

// -----------------------------------------------------------------------------
int coord_sampling(                 // sampling coordinate based on prob vector
    int   d,                            // dimension
    const float *prob)                  // probability vector
{
    float  end = prob[d-1];
    float  rnd = uniform(0.0f, end);
    return std::lower_bound(prob, prob + d, rnd) - prob;
}

// -----------------------------------------------------------------------------
float calc_ratio(                   // calc overall ratio
    int   k,                            // top-k value
    const Result *R,                    // ground truth results 
    MinK_List *list)                    // results returned by algorithms
{
    // add penalty if list->size() < k
    if (list->size() < k) return sqrt(MAXREAL);

    // consider geometric mean instead of arithmetic mean for overall ratio
    float sum = 0.0f;
    double r = -1.0;
    for (int j = 0; j < k; ++j) {
        float found = list->ith_key(j);
        float truth = R[j].key_;
        float diff  = fabs(found - truth);
        
        if (diff < CHECK_ERROR) r = 1.0;
        else r = sqrt((found + 1e-9) / (truth + 1e-9));
        sum += log(r);
    }
    return pow(E, sum / k);
}

// -----------------------------------------------------------------------------
void calc_pre_recall(               // calc precision and recall (percentage)
    int   top_k,                        // top-k value
    int   check_k,                      // number of checked objects
    const Result *R,                    // ground truth results 
    MinK_List *list,                    // results returned by algorithms
    float &recall,                      // recall value (return)
    float &precision)                   // precision value (return)
{
    int i = list->size() - 1;
    float last = R[top_k-1].key_;
    while (i >= 0 && list->ith_key(i) - last > CHECK_ERROR) {
        --i;
    }
    recall    = (i + 1) * 100.0f / top_k;
    precision = (i + 1) * 100.0f / check_k;
}

// -----------------------------------------------------------------------------
void write_index_info(              // display & write index overhead info
    float memory,                       // memory cost
    FILE  *fp)                          // file pointer (return)
{
    float time = g_end_time.tv_sec-g_start_time.tv_sec + (g_end_time.tv_usec-
        g_start_time.tv_usec)/1000000.0f;

    printf("Indexing Time: %.3f Seconds\n", time);
    printf("Estimated Mem: %.3f MB\n\n", memory);

    fprintf(fp, "Indexing Time: %f Seconds\n", time);
    fprintf(fp, "Estimated Memory: %f MB\n", memory);
}

// -----------------------------------------------------------------------------
void head(                          // display head with method name
    const char *method_name)            // method name
{
    printf("%s for Point-to-Hyperplane NNS:\n", method_name); 
    printf("Top-k\t\tRatio\t\tTime (ms)\tRecall (%%)\tPrecision (%%)\t"
        "Fraction (%%)\n");
}

// -----------------------------------------------------------------------------
void write_params(                  // write parameters
    int   cand,                         // number of candidates
    const char *method_name,            // method name
    FILE  *fp)                          // file pointer (return)
{
    fprintf(fp, "cand=%d\n", cand);
    printf("cand=%d\n", cand);
    head(method_name);
}

// -----------------------------------------------------------------------------
void write_params(                  // write parameters
    int   l,                            // collision (or separation) threshold 
    int   cand,                         // number of candidates
    const char *method_name,            // method name
    FILE  *fp)                          // file pointer (return)
{
    fprintf(fp, "l=%d, cand=%d\n", l, cand);
    printf("l=%d, cand=%d\n", l, cand);
    head(method_name);
}

// -----------------------------------------------------------------------------
void write_params(                  // write parameters
    int   cand,                         // number of candidates
    float c,                            // approximation ratio
    const char *method_name,            // method name
    FILE  *fp)                          // file pointer (return)
{
    fprintf(fp, "cand=%d, c=%.1f\n", cand, c);
    printf("cand=%d, c=%.1f\n", cand, c);
    head(method_name);
}

// -----------------------------------------------------------------------------
void write_params(                  // write parameters
    int   l,                            // collision (or separation) threshold 
    int   cand,                         // number of candidates
    float c,                            // approximation ratio
    const char *method_name,            // method name
    FILE  *fp)                          // file pointer (return)
{
    fprintf(fp, "l=%d, cand=%d, c=%.1f\n", l, cand, c);
    printf("l=%d, cand=%d, c=%.1f\n", l, cand, c);
    head(method_name);
}

// -----------------------------------------------------------------------------
void foot(                          // close file
    FILE *fp)                           // file pointer (return)
{
    printf("\n");
    fprintf(fp,"\n");
}

// -----------------------------------------------------------------------------
void init_global_metric()           // init the global metric
{
    g_ratio     = 0.0f;
    g_recall    = 0.0f;
    g_precision = 0.0f;
    g_fraction  = 0.0f;
}

// -----------------------------------------------------------------------------
void update_global_metric(          // init the global metric
    int   top_k,                        // top-k value
    int   check_k,                      // number of checked objects
    int   n,                            // number of data points
    const Result *R,                    // ground truth results 
    MinK_List *list)                    // results returned by algorithms
{
    float recall = 0.0f;
    float precision = 0.0f;
    calc_pre_recall(top_k, check_k, R, list, recall, precision);

    g_recall    += recall;
    g_precision += precision;
    g_fraction  += check_k * 100.0f / n;
    g_ratio     += calc_ratio(top_k, R, list);
}

// -----------------------------------------------------------------------------
void calc_and_write_global_metric(  // init the global metric
    int   top_k,                         // top-k value
    int   qn,                            // number of queries
    FILE *fp)                      // file pointer
{
    g_runtime = g_end_time.tv_sec - g_start_time.tv_sec + (g_end_time.tv_usec -
        g_start_time.tv_usec) / 1000000.0f;
    
    g_ratio     = g_ratio     / qn;
    g_recall    = g_recall    / qn;
    g_precision = g_precision / qn;
    g_fraction  = g_fraction  / qn;
    g_runtime   = g_runtime * 1000.0f / qn;

    printf("%3d\t\t%.3f\t\t%.2f\t\t%.3f\t\t%.3f\t\t%.3f\n", top_k, 
        g_ratio, g_runtime, g_recall, g_precision, g_fraction);
    fprintf(fp, "%d\t%f\t%f\t%f\t%f\t%f\n", top_k, g_ratio, g_runtime, 
        g_recall, g_precision, g_fraction);
}

} // end namespace p2h

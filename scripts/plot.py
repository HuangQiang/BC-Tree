import os
import re
import glob
import numpy as np
import matplotlib.pylab as plt
import matplotlib

from os import makedirs
from os.path import isdir, isfile, join
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
from itertools import chain, count
from collections import defaultdict

from plot_util  import *
from plot_other import *

# ------------------------------------------------------------------------------
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42

method_labels_map = {
    'P2LSH':       'P2LSH',
    'P2LSH_DCF':   'P2LSH-QALSH',
    'Ortho_LSH':   'Ortho-LSH',
    'Ortho_LSH_Plus': 'Ortho-LSH$^+$',
    'Ortho_LCCS_LSH': 'Ortho-LCCS-LSH',
    'Metric_Tree': 'Ortho-LSH$^+$',
    'FH':          'FH',
    'NH':          'NH',
    'NH_DCF':      'NH$^*$',
    'FH_wo_S':     'FH-wo-S',
    'NH_wo_S':     'NH-wo-S', 
    'NH_DCF_wo_S': 'NH-QALSH-wo-S', 
    'EH':          'EH',
    'BH':          'BH',
    'MH':          'MH',
    'Random_Scan': 'Random-Scan',
    'Sorted_Scan': 'Sorted-Scan',
    'Ball_Tree':   'Ball-Tree',
    'BC_Tree':     'BC-Tree',
    'BC_Tree_Minus':"BC-Tree$^-$",
    'KD_Tree':     'KD-Tree',
    'Linear':      'Linear-Scan',
}

dataset_labels_map = {
    'Yelp':        'Yelp ($d=50$)',
    'GloVe100':    'GloVe ($d=100$)',
    'Music':       'Music ($d=100$)',
    'Sift':        'Sift ($d=128$)',
    'UKBench':     'UKBench ($d=128$)',
    'ImageNet':    'ImageNet ($d=150$)',
    'Audio':       'Audio ($d=192$)',
    'Deep':        'Deep ($d=256$)',
    'Tiny1M':      'Tiny ($d=384$)',
    'Msong':       'Msong ($d=420$)',
    'NUSW':        'NUSW ($d=500$)',
    'Cifar':       'Cifar-10 ($d=512$)',
    'LabelMe':     'LabelMe ($d=512$)',
    'Sun':         'Sun ($d=512$)',
    'Mnist':       'Mnist ($d=784$)',
    'Gist':        'Gist ($d=960$)',
    'Enron':       'Enron ($d=1,369$)',
    'Trevi':       'Trevi ($d=4,096$)',
    'P53':         'P53 ($d=5,408$)',
    'Sift1M':      'Sift1M ($d=128$)',
    'Sift10M':     'Sift10M ($d=128$)',
    'Sift100M':    'Sift100M ($d=128$)',
    'Deep1M':      'Deep1M ($d=96$)',
    'Deep10M':     'Deep10M ($d=96$)',
    'Deep100M':    'Deep100M ($d=96$)',
}

# methods = ['Ortho_LSH', 'FH', 'NH', 'NH_DCF']
method_colors  = ['red', 'blue', 'green', 'purple', 
    'darkorange', 'deeppink', 'deepskyblue', 
    'olive', 'dodgerblue', 'dimgray']
method_markers = ['o', '^', 's', 'd', 'x', '*', 'p', 'v', 'D', '>']


# ------------------------------------------------------------------------------
def calc_width_and_height(n_datasets, n_rows):
    '''
    calc the width and height of figure

    :params n_datasets: number of dataset (integer)
    :params n_rows: number of rows (integer)
    :returns: width and height of figure
    '''
    fig_width  = 0.55 + 3.2 * n_datasets
    fig_height = 0.80 + 2.6 * n_rows
    
    return fig_width, fig_height


# ------------------------------------------------------------------------------
def calc_width_and_height2(n_datasets, n_rows, wide_scale, high_scale):
    '''
    calc the width and height of figure

    :params n_datasets: number of dataset (integer)
    :params n_rows: number of rows (integer)
    :returns: width and height of figure
    '''
    fig_width  = 0.55 + wide_scale * n_datasets
    fig_height = 0.80 + high_scale * n_rows
    
    return fig_width, fig_height

# ------------------------------------------------------------------------------
def get_filename(input_folder, dataset_name, method_name):
    '''
    get the file prefix 'dataset_method'

    :params input_folder: input folder (string)
    :params dataset_name: name of dataset (string)
    :params method_name:  name of method (string)
    :returns: file prefix (string)
    '''
    name = '%s%s/%s.out' % (input_folder, dataset_name, method_name)
    return name

# ------------------------------------------------------------------------------
def get_filename_branch(input_folder, dataset_name, method_name):
    '''
    get the file prefix 'dataset_method'

    :params input_folder: input folder (string)
    :params dataset_name: name of dataset (string)
    :params method_name:  name of method (string)
    :returns: file prefix (string)
    '''
    name = '%s%s/%s.out2' % (input_folder, dataset_name, method_name)
    return name

# ------------------------------------------------------------------------------
def parse_res(filename, chosen_top_k):
    '''
    parse result and get info such as ratio, qtime, recall, index_size, 
    chosen_k, and the setting of different methods
    
    BH: m=2, l=8, b=0.90
    Indexing Time: 2.708386 Seconds
    Estimated Memory: 347.581116 MB
    cand=10000
    1	5.948251	2.960960	0.000000	0.000000	0.844941
    5	4.475743	2.954690	0.400000	0.000200	0.845279
    10	3.891794	2.953910	0.900000	0.000899	0.845703
    20	3.289422	2.963460	0.950000	0.001896	0.846547
    50	2.642880	2.985980	0.900000	0.004478	0.849082
    100	2.244649	3.012860	0.800000	0.007922	0.853307

    cand=50000
    1	3.905541	14.901140	6.000000	0.000120	4.222926
    5	2.863510	14.905370	4.800000	0.000480	4.223249
    10	2.626913	14.910181	5.300000	0.001061	4.223649
    20	2.392440	14.913270	4.850000	0.001941	4.224458
    50	2.081206	14.931760	4.560000	0.004558	4.227065
    100	1.852284	14.964050	4.500000	0.008987	4.231267
    '''
    setting_pattern = re.compile(r'\S+\s+.*=.*')

    setting_m = re.compile(r'.*(m)=(\d+).*')
    setting_l = re.compile(r'.*(l)=(\d+).*')
    setting_M = re.compile(r'.*(M)=(\d+).*')
    setting_s = re.compile(r'.*(s)=(\d+).*')
    setting_b = re.compile(r'.*(b)=(\d+\.\d+).*')
    setting_leaf = re.compile(r'.*(leaf)=(\d+).*')

    param_settings = [setting_m, setting_l, setting_M, setting_s, setting_b, setting_leaf]

    index_time_pattern   = re.compile(r'Indexing Time: (\d+\.\d+).*')
    memory_usage_pattern = re.compile(r'Estimated Memory: (\d+\.\d+).*')
    candidate_pattern    = re.compile(r'.*cand=(\d+).*')
    records_pattern      = re.compile(r'(\d+)\s*(\d+\.\d+)\s*(\d+\.\d+)\s*(\d+\.\d+)\s*(\d+\.\d+)\s*(\d+\.\d+)')

    params = {}
    with open(filename, 'r') as f:
        for line in f:
            res = setting_pattern.match(line)
            if res:
                for param_setting in param_settings:
                    tmp_res = param_setting.match(line)
                    if tmp_res is not None:
                        # print(tmp_res.groups())
                        params[tmp_res.group(1)] = tmp_res.group(2)
                # print("setting=", line)

            res = index_time_pattern.match(line)
            if res:
                chosen_k = float(res.group(1))
                # print('chosen_k=', chosen_k)
            
            res = memory_usage_pattern.match(line)
            if res:
                memory_usage = float(res.group(1))
                # print('memory_usage=', memory_usage)

            res = candidate_pattern.match(line)
            if res:
                cand = int(res.group(1))
                # print('cand=', cand)
            
            res = records_pattern.match(line)
            if res:
                top_k     = int(res.group(1))
                ratio     = float(res.group(2))
                qtime     = float(res.group(3))
                recall    = float(res.group(4))
                precision = float(res.group(5))
                fraction  = float(res.group(6))
                # print(top_k, ratio, qtime, recall, precision, fraction)

                if top_k == chosen_top_k:
                    yield ((cand, params), (top_k, chosen_k, memory_usage, 
                        ratio, qtime, recall, precision, fraction))


# ------------------------------------------------------------------------------
def getindexingtime(res):
    return res[1]
def getindexsize(res):
    return res[2]
def getratio(res):
    return res[3]
def gettime(res):
    return res[4]
def getrecall(res):
    return res[5]
def getprecision(res):
    return res[6]
def getfraction(res):
    return res[7]

def get_cand(res):
    return int(res[0][0])
def get_l(res):
    return int(res[0][1]['l'])
def get_leaf(res):
    return int(res[0][1]['leaf'])
def get_m(res):
    return int(res[0][1]['m'])
def get_s(res):
    return int(res[0][1]['s'])
def get_time(res):
    return float(res[1][4])
def get_recall(res):
    return float(res[1][5])
def get_precision(res):
    return float(res[1][6])
def get_fraction(res):
    return float(res[1][7])


# ------------------------------------------------------------------------------
def lower_bound_curve(xys):
    '''
    get the time-recall curve by convex hull and interpolation

    :params xys: 2-dim array (np.array)
    :returns: time-recall curve with interpolation
    '''
    # add noise and conduct convex hull to find the curve
    eps = np.random.normal(size=xys.shape) * 1e-6
    xys += eps
    # print(xys)
    
    hull = ConvexHull(xys)
    hull_vs = xys[hull.vertices]
    # hull_vs = np.array(sorted(hull_vs, key=lambda x:x[1]))
    # print("hull_vs: ", hull_vs)

    # find max pair (maxv0) and min pairs (v1s) from the convex hull
    v1s = []
    maxv0 = [-1, -1]
    for v0, v1 in zip(hull_vs, chain(hull_vs[1:], hull_vs[:1])):
        # print(v0, v1)
        if v0[1] > v1[1] and v0[0] > v1[0]:
            v1s = np.append(v1s, v1, axis=-1)
            if v0[1] > maxv0[1]:
                maxv0 = v0
    # print(v1s, maxv0)
                
    # interpolation: vs[:, 1] -> recall (x), vs[:, 0] -> time (y)
    vs = np.array(np.append(maxv0, v1s)).reshape(-1, 2) # 2-dim array
    f = interp1d(vs[:, 1], vs[:, 0])

    minx = np.min(vs[:, 1]) + 1e-6
    maxx = np.max(vs[:, 1]) - 1e-6
    x = np.arange(minx, maxx, 1.0) # the interval of interpolation: 1.0
    y = list(map(f, x))          # get time (y) by interpolation

    return x, y


# ------------------------------------------------------------------------------
def upper_bound_curve(xys, interval, is_sorted):
    '''
    get the time-ratio and precision-recall curves by convex hull and interpolation

    :params xys: 2-dim array (np.array)
    :params interval: the interval of interpolation (float)
    :params is_sorted: sort the convex hull or not (boolean)
    :returns: curve with interpolation
    '''
    # add noise and conduct convex hull to find the curve
    eps = np.random.normal(size=xys.shape) * 1e-6
    xys += eps
    # print(xys)
    
    xs = xys[:, 0]
    if len(xs) > 2 and xs[-1] > 0:
        hull = ConvexHull(xys)
        hull_vs = xys[hull.vertices]
        if is_sorted:
            hull_vs = np.array(sorted(hull_vs, key=lambda x:x[1]))
        print("hull_vs: ", hull_vs)

        # find max pair (maxv0) and min pairs (v1s) from the convex hull
        v1s = []
        maxv0 = [-1, -1]
        for v0, v1 in zip(hull_vs, chain(hull_vs[1:], hull_vs[:1])):
            # print(v0, v1)
            if v0[1] > v1[1] and v0[0] < v1[0]:
                v1s = np.append(v1s, v1, axis=-1)
                if v0[1] > maxv0[1]:
                    maxv0 = v0
        print(v1s, maxv0)

        # interpolation: vs[:, 1] -> recall (x), vs[:, 0] -> time (y)
        vs = np.array(np.append(maxv0, v1s)).reshape(-1, 2) # 2-dim array
        if len(vs) >= 2:
            f = interp1d(vs[:, 1], vs[:, 0])

            minx = np.min(vs[:, 1]) + 1e-6
            maxx = np.max(vs[:, 1]) - 1e-6
            x = np.arange(minx, maxx, interval)
            y = list(map(f, x))          # get time (y) by interpolation

            return x, y
        else:
            return xys[:, 0], xys[:, 1]
    else:
        return xys[:, 0], xys[:, 1]


# ------------------------------------------------------------------------------
def lower_bound_curve2(xys):
    '''
    get the querytime-indexsize and querytime-indextime curve by convex hull

    :params xys: 2-dim array (np.array)
    :returns: querytime-indexsize and querytime-indextime curve
    '''
    # add noise and conduct convex hull to find the curve
    eps = np.random.normal(size=xys.shape) * 1e-6
    xys += eps
    # print(xys)

    xs = xys[:, 0]
    if len(xs) > 2 and xs[-1] > 0:
        # conduct convex hull to find the curve
        hull = ConvexHull(xys)
        hull_vs = xys[hull.vertices]
        # print("hull_vs: ", hull_vs)
        
        ret_vs = []
        for v0, v1, v2 in zip(chain(hull_vs[-1:], hull_vs[:-1]), hull_vs, \
            chain(hull_vs[1:], hull_vs[:1])):

            # print(v0, v1, v2)
            if v0[0] < v1[0] or v1[0] < v2[0]:
                ret_vs = np.append(ret_vs, v1, axis=-1)

        # sort the results in ascending order of x without interpolation
        ret_vs = ret_vs.reshape((-1, 2))
        ret_vs = np.array(sorted(ret_vs, key=lambda x:x[0]))

        return ret_vs[:, 0], ret_vs[:, 1]
    else:
        return xys[:, 0], xys[:, 1]


# ------------------------------------------------------------------------------
def plot_time_index_k(chosen_top_k, chosen_top_ks, recall_level, size_x_scales,\
    time_x_scales, methods, input_folder, output_folder):
    '''
    draw the querytime-indexsize curves and querytime-indexingtime curves for 
    all methods on all datasets 

    :params chosen_top_k: top_k value for drawing figure (integer)
    :params chosen_top_ks: a list of op_k values for drawing figure (list)
    :params recall_level: recall value for drawing figure (integer)
    :params size_x_scales: a list of x scales for index size (list)
    :params time_x_scales: a list of x scales for indexing time (list)
    :params methods: a list of method (list)
    :params input_folder: input folder (string)
    :params output_folder: output folder (string)
    :returns: None
    '''
    n_datasets = len(datasets)
    fig_width, fig_height = calc_width_and_height(n_datasets, 3)
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust()       # define a window for a figure

    method_labels = [method_labels_map[method] for method in methods]
    for di, (dataset, dataset_label) in enumerate(zip(datasets, dataset_labels)):
        # set up three sub-figures
        ax_size = plt.subplot(3, n_datasets, di+1)
        plt.title(dataset_label)            # title
        plt.xlabel('Index Size (MB)')       # label of x-axis

        ax_time = plt.subplot(3, n_datasets, n_datasets+di+1)
        plt.xlabel('Indexing Time (Seconds)') # label of x-axis

        ax_k = plt.subplot(3, n_datasets, 2*n_datasets+di+1)
        plt.xlabel('$k$')                   # label of x-axis

        if di == 0:
            ax_size.set_ylabel('Query Time (ms)')
            ax_time.set_ylabel('Query Time (ms)')
            ax_k.set_ylabel('Query Time (ms)')

        min_size_x = 1e9; max_size_x = -1e9
        min_size_y = 1e9; max_size_y = -1e9
        min_time_x = 1e9; max_time_x = -1e9
        min_time_y = 1e9; max_time_y = -1e9
        min_k_y    = 1e9; max_k_y    = -1e9
        for method_idx, method, method_label, method_color, method_marker in \
            zip(count(), methods, method_labels, method_colors, method_markers):

            # get file name for this method on this dataset
            filename = get_filename(input_folder, dataset, method)
            if filename is None: continue
            print(filename)

            # ------------------------------------------------------------------
            #  query time vs. index size and indexing time
            # ------------------------------------------------------------------
            # get all results from disk
            chosen_ks_dict = defaultdict(list)
            for _,res in parse_res(filename, chosen_top_k):
                query_time = gettime(res)
                recall     = getrecall(res)
                index_time = getindexingtime(res)
                index_size = getindexsize(res)
                chosen_ks_dict[(index_time, index_size)] += [[recall, query_time]]

            # get querytime-indexsize and querytime-indexingtime results if its 
            # recall is higher than recall_level 
            index_times, index_sizes, querytimes_at_recall = [], [], []
            for (index_time, index_size), recall_querytimes_ in chosen_ks_dict.items():
                # add [[0, 0]] for interpolation
                recall_querytimes_ = np.array([[0, 0]] + recall_querytimes_)

                recalls, query_times = lower_bound_curve2(recall_querytimes_)
                if np.max(recalls) > recall_level:
                    # get the estimated time at recall level by interpolation 
                    f = interp1d(recalls, query_times)
                    querytime_at_recall = f(recall_level)

                    # update results
                    index_times += [index_time]
                    index_sizes += [index_size]
                    querytimes_at_recall += [querytime_at_recall]
            
            index_times = np.array(index_times)
            index_sizes = np.array(index_sizes)
            querytimes_at_recall = np.array(querytimes_at_recall)
          
            # get the querytime-indexsize curve by convex hull
            isize_qtime = np.zeros(shape=(len(index_sizes), 2))
            isize_qtime[:, 0] = index_sizes
            isize_qtime[:, 1] = querytimes_at_recall

            lower_isizes, lower_qtimes = lower_bound_curve2(isize_qtime)
            if len(lower_isizes) > 0:
                # print(method, lower_isizes, lower_qtimes)
                min_size_x = min(min_size_x, np.min(lower_isizes))
                max_size_x = max(max_size_x, np.max(lower_isizes))
                min_size_y = min(min_size_y, np.min(lower_qtimes))
                max_size_y = max(max_size_y, np.max(lower_qtimes))
                ax_size.semilogy(lower_isizes, lower_qtimes, '-', color=method_color, 
                    marker=method_marker, label=method_label if di==0 else "", 
                    markerfacecolor='none', markersize=7)

                # get the querytime-indextime curve by convex hull
                itime_qtime = np.zeros(shape=(len(index_times), 2))
                itime_qtime[:, 0] = index_times
                itime_qtime[:, 1] = querytimes_at_recall

                lower_itimes, lower_qtimes = lower_bound_curve2(itime_qtime)
                # print(method, lower_itimes, lower_qtimes)
                min_time_x = min(min_time_x, np.min(lower_itimes))
                max_time_x = max(max_time_x, np.max(lower_itimes))
                min_time_y = min(min_time_y, np.min(lower_qtimes))
                max_time_y = max(max_time_y, np.max(lower_qtimes))
                ax_time.semilogy(lower_itimes, lower_qtimes, '-', color=method_color, 
                    marker=method_marker, label="", markerfacecolor='none', 
                    markersize=7, zorder=len(methods)-method_idx)

            # ------------------------------------------------------------------
            #  query time vs. k
            # ------------------------------------------------------------------
            # get all results from disk
            chosen_ks_dict = defaultdict(list)
            for chosen_top_k in chosen_top_ks:
                for _,res in parse_res(filename, chosen_top_k):
                    query_time = gettime(res)
                    recall     = getrecall(res)
                    chosen_ks_dict[chosen_top_k] += [[recall, query_time]]

            # get querytime-indexsize and querytime-indexingtime results if its 
            # recall is higher than recall_level 
            chosen_ks, querytimes_at_recall = [], []
            for chosen_k, recall_querytimes_ in chosen_ks_dict.items():
                # add [[0, 0]] for interpolation
                recall_querytimes_ = np.array([[0, 0]] + recall_querytimes_)

                recalls, query_times = lower_bound_curve2(recall_querytimes_)
                if np.max(recalls) > recall_level:
                    # get the estimated time at recall level by interpolation 
                    f = interp1d(recalls, query_times)
                    querytime_at_recall = f(recall_level)

                    # update results
                    chosen_ks += [chosen_k]
                    querytimes_at_recall += [querytime_at_recall]

            chosen_ks = np.array(chosen_ks)
            querytimes_at_recall = np.array(querytimes_at_recall)

            min_k_y = min(min_k_y, np.min(querytimes_at_recall))
            max_k_y = max(max_k_y, np.max(querytimes_at_recall))
            ax_k.semilogy(chosen_ks, querytimes_at_recall, '-', color=method_color, 
                marker=method_marker, label="", markerfacecolor='none', 
                markersize=7, zorder=len(methods)-method_idx)

        # set up the limit (or range) of y-axis 
        plt_helper.set_x_axis(ax_size, min_size_x, size_x_scales[di]*max_size_x)
        plt_helper.set_y_axis_log10(ax_size, min_size_y, max_size_y)

        plt_helper.set_x_axis(ax_time, min_time_x, time_x_scales[di]*max_time_x)
        plt_helper.set_y_axis_log10(ax_time, min_time_y, max_time_y)
        
        plt_helper.set_y_axis_log10(ax_k, min_k_y, max_k_y)

    # plot legend and save figure
    plt_helper.plot_fig_legend(ncol=len(methods))
    plt_helper.plot_and_save(output_folder, 'time_index_k_%d' % recall_level)


# ------------------------------------------------------------------------------
def plot_time_recall(fname, chosen_top_k, methods, datasets, dataset_labels, 
    input_folder, output_folder):
    '''
    draw the querytime-recall curve for all methods on all datasets 

    :params chosen_top_k: top_k value for drawing figure (integer)
    :params methods: a list of method (list)
    :params input_folder: input folder (string)
    :params output_folder: output folder (string)
    :returns: None
    '''
    num_row = 2
    num_col = len(datasets)/num_row
    fig_width, fig_height = calc_width_and_height(num_col, num_row)
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust()         # define a window for a figure
    
    method_labels = [method_labels_map[method] for method in methods]
    for di, (dataset, dataset_label) in enumerate(zip(datasets, dataset_labels)):
        # set up each sub-figure
        ax = plt.subplot(num_row, num_col, di+1)
        plt.title(dataset_label, fontsize=14) # title
        plt.xlim(0, 100)                      # limit (or range) of x-axis
        plt.xticks(np.arange(0, 101, step=20))
        plt.xlabel('Recall (%)')              # label of x-axis
        if di == 0 or di == num_col:          # add label of y-axis at 1st dataset
            plt.ylabel('Query Time (ms)')

        miny = 1e9
        maxy = -1e9
        for method_idx, method, method_label, method_color, method_marker in \
            zip(count(), methods, method_labels, method_colors, method_markers):
            
            # get file name for this method on this dataset
            filename = get_filename(input_folder, dataset, method)
            if filename is None: continue
            print(filename)

            # get time-recall results
            time_recalls = []
            for _,res in parse_res(filename, chosen_top_k):
                time_recalls += [[gettime(res), getrecall(res)]]

            time_recalls = np.array(time_recalls)
            # print(time_recalls)

            # get the time-recall curve by convex hull and interpolation, where 
            # lower_recalls -> x, lower_times -> y
            lower_recalls, lower_times = lower_bound_curve(time_recalls) 
            miny = min(miny, np.min(lower_times))
            maxy = max(maxy, np.max(lower_times)) 
            ax.semilogy(lower_recalls, lower_times, '-', color=method_color, 
                marker=method_marker, label=method_label if di==0 else "", 
                markevery=10, markerfacecolor='none', markersize=7, 
                zorder=len(methods)-method_idx)

        # set up the limit (or range) of y-axis
        ratio = maxy / miny
        if ratio <= 10:
            plt_helper.set_y_axis_close(ax, miny, maxy)
        else:
            plt_helper.set_y_axis_log10(ax, miny, maxy)
        plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
    
    # plot legend and save figure
    plt_helper.plot_fig_legend(ncol=len(methods), legend_width=0.6)
    plt_helper.plot_and_save(output_folder, '%s_%d' % (fname, chosen_top_k))
    plt.show()


# ------------------------------------------------------------------------------
def plot_fraction_recall(fname, chosen_top_k, methods, datasets, dataset_labels, 
    input_folder, output_folder):
    '''
    draw the fraction-recall curve for all methods on all datasets 

    :params chosen_top_k: top_k value for drawing figure (integer)
    :params methods: a list of method (list)
    :params input_folder: input folder (string)
    :params output_folder: output folder (string)
    :returns: None
    '''
    num_row = 2
    num_col = len(datasets)/num_row
    fig_width, fig_height = calc_width_and_height(num_col, num_row)
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust()       # define a window for a figure
    
    method_labels = [method_labels_map[method] for method in methods]
    for di, (dataset, dataset_label) in enumerate(zip(datasets, dataset_labels)):
        # set up each sub-figure
        ax = plt.subplot(num_row, num_col, di+1)
        plt.title(dataset_label, fontsize=14) # title
        plt.xlim(0, 100)                      # limit (or range) of x-axis
        plt.xticks(np.arange(0, 101, step=20))
        plt.xlabel('Recall (%)')              # label of x-axis
        if di == 0 or di == num_col:          # add label of y-axis at 1st dataset
            plt.ylabel('Fraction (%)')

        miny = 1e9
        maxy = -1e9
        for method_idx, method, method_label, method_color, method_marker in \
            zip(count(), methods, method_labels, method_colors, method_markers):
            
            # get file name for this method on this dataset
            filename = get_filename(input_folder, dataset, method)
            if filename is None: continue
            print(filename)

            # get fraction-recall results
            fraction_recalls = []
            for _,res in parse_res(filename, chosen_top_k):
                fraction_recalls += [[getfraction(res), getrecall(res)]]

            fraction_recalls = np.array(fraction_recalls)
            # print(fraction_recalls)

            # get the fraction-recall curve by convex hull and interpolation, where 
            # lower_recalls -> x, lower_times -> y
            # print('fraction_recall!!!!\n', fraction_recalls)
            lower_recalls, lower_fractions = lower_bound_curve(fraction_recalls) 
            miny = min(miny, np.min(lower_fractions))
            maxy = max(maxy, np.max(lower_fractions))
            ax.semilogy(lower_recalls, lower_fractions, '-', color=method_color, 
                marker=method_marker, label=method_label if di==0 else "", 
                markevery=10, markerfacecolor='none', markersize=7, 
                zorder=len(methods)-method_idx)

        # set up the limit (or range) of y-axis
        ratio = maxy / miny
        if ratio <= 10:
            plt_helper.set_y_axis_close(ax, miny, maxy)
        else:
            plt_helper.set_y_axis_log10(ax, miny, maxy)
        plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
    
    # plot legend and save figure
    plt_helper.plot_fig_legend(ncol=len(methods), legend_width=0.6)
    plt_helper.plot_and_save(output_folder, '%s_%d' % (fname, chosen_top_k))
    plt.show()


# ------------------------------------------------------------------------------
def plot_time_k(fname, chosen_top_ks, recall_level, methods, datasets, 
    dataset_labels, input_folder, output_folder):
    '''
    draw the querytime-indexsize curves and querytime-indexingtime curves for 
    all methods on all datasets 

    :params chosen_top_ks: top_k value for drawing figure (list)
    :params recall_level: recall value for drawing figure (integer)
    :params methods: a list of method (list)
    :params input_folder: input folder (string)
    :params output_folder: output folder (string)
    :returns: None
    '''
    num_row = 2
    num_col = len(datasets)/num_row
    fig_width, fig_height = calc_width_and_height(num_col, num_row)
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust()       # define a window for a figure

    method_labels = [method_labels_map[method] for method in methods]
    for di, (dataset, dataset_label) in enumerate(zip(datasets, dataset_labels)):
        # set up sub-figure
        ax_k = plt.subplot(num_row, num_col, di+1)
        plt.title(dataset_label, fontsize=14) # title
        plt.xlabel('$k$')                     # label of x-axis
        # plt.xlim(0, 100, 20)
        plt.xticks(np.arange(0, 41, step=10))
        if di == 0 or di == num_col:
            ax_k.set_ylabel('Query Time (ms)')

        miny = 1e9; maxy = -1e9
        for method_idx, method, method_label, method_color, method_marker in \
            zip(count(), methods, method_labels, method_colors, method_markers):

            # get file name for this method on this dataset
            filename = get_filename(input_folder, dataset, method)
            if filename is None: continue
            print(filename)

            # get all results from disk
            chosen_ks_dict = defaultdict(list)
            for chosen_top_k in chosen_top_ks:
                for _,res in parse_res(filename, chosen_top_k):
                    query_time = gettime(res)
                    recall     = getrecall(res)
                    chosen_ks_dict[chosen_top_k] += [[recall, query_time]]

            # get querytime-indexsize and querytime-indexingtime results if its 
            # recall is higher than recall_level 
            chosen_ks, querytimes_at_recall = [], []
            for chosen_k, recall_querytimes_ in chosen_ks_dict.items():
                # add [[0, 0]] for interpolation
                recall_querytimes_ = np.array([[0, 0]] + recall_querytimes_)

                recalls, query_times = lower_bound_curve2(recall_querytimes_)
                if np.max(recalls) > recall_level:
                    # get the estimated time at recall level by interpolation 
                    f = interp1d(recalls, query_times)
                    querytime_at_recall = f(recall_level)

                    # update results
                    chosen_ks += [chosen_k]
                    querytimes_at_recall += [querytime_at_recall]

            chosen_ks = np.array(chosen_ks)
            querytimes_at_recall = np.array(querytimes_at_recall)

            miny = min(miny, np.min(querytimes_at_recall))
            maxy = max(maxy, np.max(querytimes_at_recall))
            ax_k.semilogy(chosen_ks, querytimes_at_recall, '-', 
                    color=method_color, marker=method_marker, 
                    label=method_label if di==0 else "", 
                    markerfacecolor='none', markersize=7, 
                    zorder=len(methods)-method_idx)
                    
        # set up the limit (or range) of y-axis 
        ratio = maxy / miny
        if ratio <= 20:
            plt_helper.set_y_axis_close(ax_k, miny, maxy)
        else:
            plt_helper.set_y_axis_log10(ax_k, miny, maxy)
        plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)

    # plot legend and save figure
    plt_helper.plot_fig_legend(ncol=len(methods), legend_width=0.6)
    plt_helper.plot_and_save(output_folder, '%s_%d' % (fname, recall_level))


# ------------------------------------------------------------------------------
def plot_preference_choice(fname, chosen_top_k, methods, datasets, 
    dataset_labels, input_folder, output_folder):
    '''
    draw the querytime-recall curve for all methods on all datasets 

    :params chosen_top_k: top_k value for drawing figure (integer)
    :params methods: a list of method (list)
    :params input_folder: input folder (string)
    :params output_folder: output folder (string)
    :returns: None
    '''
    num_row = 2
    num_col = len(datasets)/num_row
    fig_width, fig_height = calc_width_and_height(num_col, num_row)
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust()         # define a window for a figure
    
    # method_labels = [method_labels_map[method] for method in methods]
    for di, (dataset, dataset_label) in enumerate(zip(datasets, dataset_labels)):
        # set up each sub-figure
        ax = plt.subplot(num_row, num_col, di+1)
        plt.title(dataset_label, fontsize=14) # title
        plt.xlim(0, 100)                      # limit (or range) of x-axis
        plt.xticks(np.arange(0, 101, step=20))
        plt.xlabel('Recall (%)')              # label of x-axis
        if di == 0 or di == num_col:          # add label of y-axis at 1st dataset
            plt.ylabel('Query Time (ms)')

        miny = 1e9
        maxy = -1e9
        for method_idx, method in zip(count(), methods):            
            
            # -------------------------------------------------------------------------
            method_idx = method_idx*2
            
            # get file name for this method on this dataset
            filename = get_filename(input_folder, dataset, method)
            if filename is None: continue
            print(filename)

            # get time-recall results
            time_recalls = []
            for _,res in parse_res(filename, chosen_top_k):
                time_recalls += [[gettime(res), getrecall(res)]]

            time_recalls = np.array(time_recalls)
            # print(time_recalls)

            # get the time-recall curve by convex hull and interpolation, where 
            # lower_recalls -> x, lower_times -> y
            lower_recalls, lower_times = lower_bound_curve(time_recalls) 
            miny = min(miny, np.min(lower_times))
            maxy = max(maxy, np.max(lower_times)) 
            
            method_label  = method_labels_map[method] + " (Center Preference)"
            method_color  = method_colors[method_idx]
            method_marker = method_markers[method_idx]
            ax.semilogy(lower_recalls, lower_times, '-', color=method_color, 
                marker=method_marker, label=method_label if di==0 else "", 
                markevery=10, markerfacecolor='none', markersize=7, 
                zorder=2*len(methods)-method_idx)
            
            # -------------------------------------------------------------------------
            method_idx = method_idx + 1
            # get file name for this method on this dataset
            filename = get_filename_branch(input_folder, dataset, method)
            if filename is None: continue
            print(filename)

            # get time-recall results
            time_recalls = []
            for _,res in parse_res(filename, chosen_top_k):
                time_recalls += [[gettime(res), getrecall(res)]]

            time_recalls = np.array(time_recalls)
            # print(time_recalls)

            # get the time-recall curve by convex hull and interpolation, where 
            # lower_recalls -> x, lower_times -> y
            lower_recalls, lower_times = lower_bound_curve(time_recalls) 
            miny = min(miny, np.min(lower_times))
            maxy = max(maxy, np.max(lower_times)) 
            
            method_label = method_labels_map[method] + " (Lower Bound Preference)"
            method_color  = method_colors[method_idx]
            method_marker = method_markers[method_idx]
            ax.semilogy(lower_recalls, lower_times, '-', color=method_color, 
                marker=method_marker, label=method_label if di==0 else "", 
                markevery=10, markerfacecolor='none', markersize=7, 
                zorder=2*len(methods)-method_idx)
            
        # set up the limit (or range) of y-axis
        ratio = maxy / miny
        if ratio <= 10:
            plt_helper.set_y_axis_close(ax, miny, maxy)
        else:
            plt_helper.set_y_axis_log10(ax, miny, maxy)
        plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
    
    # plot legend and save figure
    plt_helper.plot_fig_legend(ncol=len(methods)*2, legend_width=0.9)
    plt_helper.plot_and_save(output_folder, '%s_%d' % (fname, chosen_top_k))
    plt.show()
    

# ------------------------------------------------------------------------------
def plot_leaf_size(fname, chosen_top_k, method, datasets, input_folder, 
    output_folder, fig_width=6.5, fig_height=6.0):
    
    num_row = 2
    num_col = len(datasets)/num_row
    fig_width, fig_height = calc_width_and_height(num_col, num_row)
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust()
    
    for di, dataset in enumerate(datasets):
        ax = plt.subplot(num_row, num_col, di+1)
        plt.title(dataset_labels_map[dataset], fontsize=14) # title
        plt.xlim(0, 100)                      # limit (or range) of x-axis
        plt.xticks(np.arange(0, 101, step=20))
        ax.set_xlabel(r'Recall (%)')
        if di == 0 or di == num_col:
            ax.set_ylabel(r'Query Time (ms)')
        
        # get file name for this method on this dataset
        filename = get_filename(input_folder, dataset, method)
        print(filename, method, dataset)
        
        # fix_s=2
        data = []
        for record in parse_res(filename, chosen_top_k):
            # print(record)
            leaf   = get_leaf(record)
            cand   = get_cand(record)
            time   = get_time(record)
            recall = get_recall(record)
            
            print(leaf, cand, time, recall)
            data += [[leaf, cand, time, recall]]
        data = np.array(data)
        
        leafs = [100, 200, 500, 1000, 2000, 5000, 10000]
        leaf_labels = ["100", "200", "500", "1,000", "2,000", "5,000", "10,000"]
        maxy = -1e9
        miny = 1e9
        cnt = 0
        for color, marker, leaf, leaf_label in zip(method_colors, method_markers, leafs, leaf_labels):
            data_leafp = data[data[:, 0]==leaf]
            # print(m, data_mp)
            
            plt.semilogy(data_leafp[:, -1], data_leafp[:, -2], marker=marker, 
                label='$N_0 = %s$' % leaf_label if di==0 else "", c=color, 
                markerfacecolor='none', markersize=7)
            miny = min(miny, np.min(data_leafp[:,-2]) )
            maxy = max(maxy, np.max(data_leafp[:,-2]) ) 
        # plt.xlim(0, 100)
        # print(dataset, distance, miny, maxy)
        plt_helper.set_y_axis_log10(ax, miny, maxy)
        plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
    
    plt_helper.plot_fig_legend(ncol=len(leafs), legend_width=0.9)
    plt_helper.plot_and_save(output_folder, '%s_%s_%d' % (fname, method, 
        chosen_top_k))


# ------------------------------------------------------------------------------
def plot_lower_bound_time_k(fname, chosen_top_ks, recall_level, methods, 
    result_names, datasets, dataset_labels, input_folder, output_folder):
    '''
    draw the querytime-indexsize curves and querytime-indexingtime curves for 
    all methods on all datasets 

    :params chosen_top_ks: top_k value for drawing figure (list)
    :params recall_level: recall value for drawing figure (integer)
    :params methods: a list of method (list)
    :params input_folder: input folder (string)
    :params output_folder: output folder (string)
    :returns: None
    '''
    num_row = 2
    num_col = len(datasets)/num_row
    fig_width, fig_height = calc_width_and_height(num_col, num_row)
    
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust()       # define a window for a figure

    for di, (dataset, dataset_label) in enumerate(zip(datasets, dataset_labels)):
        # set up sub-figure
        ax_k = plt.subplot(num_row, num_col, di+1)
        plt.title(dataset_label, fontsize=14) # title
        plt.xlabel('$k$')                     # label of x-axis
        # plt.xlim(0, 100, 20)
        plt.xticks(np.arange(0, 41, step=10))
        if di == 0 or di == num_col:
            ax_k.set_ylabel('Query Time (ms)')

        miny = 1e9; maxy = -1e9
        for method_idx, method, result_name, method_color, method_marker in \
            zip(count(), methods, result_names, method_colors, method_markers):

            # get file name for this method on this dataset
            filename = '%s%s/%s' % (input_folder, dataset, result_name)
            if filename is None: continue
            print(filename)

            # get all results from disk
            chosen_ks_dict = defaultdict(list)
            for chosen_top_k in chosen_top_ks:
                for _,res in parse_res(filename, chosen_top_k):
                    query_time = gettime(res)
                    recall     = getrecall(res)
                    chosen_ks_dict[chosen_top_k] += [[recall, query_time]]

            # get querytime-indexsize and querytime-indexingtime results if its 
            # recall is higher than recall_level 
            chosen_ks, querytimes_at_recall = [], []
            for chosen_k, recall_querytimes_ in chosen_ks_dict.items():
                # add [[0, 0]] for interpolation
                recall_querytimes_ = np.array([[0, 0]] + recall_querytimes_)

                recalls, query_times = lower_bound_curve2(recall_querytimes_)
                if np.max(recalls) > recall_level:
                    # get the estimated time at recall level by interpolation 
                    f = interp1d(recalls, query_times)
                    querytime_at_recall = f(recall_level)

                    # update results
                    chosen_ks += [chosen_k]
                    querytimes_at_recall += [querytime_at_recall]

            chosen_ks = np.array(chosen_ks)
            querytimes_at_recall = np.array(querytimes_at_recall)

            miny = min(miny, np.min(querytimes_at_recall))
            maxy = max(maxy, np.max(querytimes_at_recall))
            method_label = method
            ax_k.plot(chosen_ks, querytimes_at_recall, color=method_color, 
                marker=method_marker, label=method_label if di==0 else "", 
                markerfacecolor='none', markersize=7, zorder=len(methods)-method_idx)
            # ax_k.semilogy(chosen_ks, querytimes_at_recall, '-', color=method_color, 
            #     marker=method_marker, label=method_label if di==0 else "", 
            #     markerfacecolor='none', markersize=7, zorder=len(methods)-method_idx)
                    
        # set up the limit (or range) of y-axis 
        plt_helper.set_y_axis_equal(ax_k, miny, maxy)
        # ratio = maxy / miny
        # if ratio <= 20:
        #     plt_helper.set_y_axis_close(ax_k, miny, maxy)
        # else:
        #     plt_helper.set_y_axis_log10(ax_k, miny, maxy)
        plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)

    # plot legend and save figure
    plt_helper.plot_fig_legend(ncol=len(methods), legend_width=0.6)
    plt_helper.plot_and_save(output_folder, '%s_%d' % (fname, recall_level))


# ------------------------------------------------------------------------------
def plot_scale_time_fraction_recall(fname, chosen_top_k, methods, datasets, 
    dataset_labels, input_folder, output_folder):
    '''
    draw the querytime-recall curves and fraction-recall curves for all methods 
    on all datasets 

    :params chosen_top_k: top_k value for drawing figure (integer)
    :params methods: a list of method (list)
    :params input_folder: input folder (string)
    :params output_folder: output folder (string)
    :returns: None
    '''
    n_datasets = len(datasets)
    fig_width, fig_height = calc_width_and_height2(n_datasets, 2, 3.2, 2.3)
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust()       # define a window for a figure
    
    method_labels = [method_labels_map[method] for method in methods]
    for di, (dataset, dataset_label) in enumerate(zip(datasets, dataset_labels)):
        # set up two sub-figures
        ax_recall = plt.subplot(2, n_datasets, di+1)
        plt.title(dataset_label)
        plt.xlabel('Recall (%)')
        plt.xlim(0, 100)
        plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
        
        ax_fraction = plt.subplot(2, n_datasets, n_datasets+di+1)
        plt.xlabel('Recall (%)')
        plt.xlim(0, 100)
        plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
        
        if di == 0:
            ax_recall.set_ylabel('Query Time (ms)')
            ax_fraction.set_ylabel('Fraction (%)')
        
        min_t_y = 1e9; max_t_y = -1e9
        min_f_y = 1e9; max_f_y = -1e9
        for method_idx, method, method_label, method_color, method_marker in \
            zip(count(), methods, method_labels, method_colors, method_markers):

            # get file name for this method on this dataset
            filename = get_filename(input_folder, dataset, method)
            if filename is None: continue
            print(filename)

            # get time-recall and fraction-recall results from disk
            time_recalls     = []
            fraction_recalls = []
            for _,res in parse_res(filename, chosen_top_k):
                time_recalls     += [[gettime(res),     getrecall(res)]]
                fraction_recalls += [[getfraction(res), getrecall(res)]]

            time_recalls     = np.array(time_recalls)
            fraction_recalls = np.array(fraction_recalls)
            # print(time_recalls, fraction_recalls)
            
            # get the time-recall curve by convex hull and interpolation
            lower_recalls, lower_times = lower_bound_curve(time_recalls)
            min_t_y = min(min_t_y, np.min(lower_times))
            max_t_y = max(max_t_y, np.max(lower_times))
            ax_recall.semilogy(lower_recalls, lower_times, '-', 
                color=method_color, marker=method_marker, 
                label=method_label if di==0 else "", markevery=10, 
                markerfacecolor='none', markersize=7)
            
            # get the fraction-recall curve by convex hull
            lower_recalls, lower_fractions = lower_bound_curve(fraction_recalls)
            min_f_y = min(min_f_y, np.min(lower_fractions))
            max_f_y = max(max_f_y, np.max(lower_fractions))
            ax_fraction.semilogy(lower_recalls, lower_fractions, '-', 
                color=method_color, marker=method_marker, label="", 
                markevery=10, markerfacecolor='none', markersize=7, 
                zorder=len(methods)-method_idx)
            
        # set up the limit (or range) of y-axis
        plt_helper.set_y_axis_log10(ax_recall,   min_t_y, max_t_y)
        plt_helper.set_y_axis_log10(ax_fraction, min_f_y, max_f_y)
        
    # plot legend and save figure
    plt_helper.plot_fig_legend(ncol=len(methods))
    plt_helper.plot_and_save(output_folder, '%s_%d' % (fname, chosen_top_k))


# ------------------------------------------------------------------------------
def plot_scale_time_recall(fname, chosen_top_k, methods, datasets, 
    dataset_labels, input_folder, output_folder):
    '''
    draw the querytime-recall curve for all methods on all datasets 

    :params chosen_top_k: top_k value for drawing figure (integer)
    :params methods: a list of method (list)
    :params input_folder: input folder (string)
    :params output_folder: output folder (string)
    :returns: None
    '''
    num_row = 1
    num_col = len(datasets)
    fig_width, fig_height = calc_width_and_height2(num_col, num_row, 3.2, 2.2)
    
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust(top_space=0.75)
    
    method_labels = [method_labels_map[method] for method in methods]
    for di, (dataset, dataset_label) in enumerate(zip(datasets, dataset_labels)):
        # set up each sub-figure
        ax = plt.subplot(num_row, num_col, di+1)
        plt.title(dataset_label, fontsize=13)
        plt.xlim(0, 100)
        plt.xticks(np.arange(0, 101, step=20))
        plt.xlabel('Recall (%)', fontsize=13)
        if di == 0 or di == num_col:
            plt.ylabel('Query Time (ms)', fontsize=13)

        miny = 1e9
        maxy = -1e9
        for method_idx, method, method_label, method_color, method_marker in \
            zip(count(), methods, method_labels, method_colors, method_markers):
            
            # get file name for this method on this dataset
            filename = get_filename(input_folder, dataset, method)
            if filename is None: continue
            print(filename)

            # get time-recall results
            time_recalls = []
            for _,res in parse_res(filename, chosen_top_k):
                time_recalls += [[gettime(res), getrecall(res)]]

            time_recalls = np.array(time_recalls)
            # print(time_recalls)

            # get the time-recall curve by convex hull and interpolation, where 
            # lower_recalls -> x, lower_times -> y
            lower_recalls, lower_times = lower_bound_curve(time_recalls) 
            miny = min(miny, np.min(lower_times))
            maxy = max(maxy, np.max(lower_times)) 
            ax.semilogy(lower_recalls, lower_times, '-', color=method_color, 
                marker=method_marker, label=method_label if di==0 else "", 
                markevery=10, markerfacecolor='none', markersize=7, 
                zorder=len(methods)-method_idx)

        # set up the limit (or range) of y-axis
        ratio = maxy / miny
        if ratio <= 10:
            plt_helper.set_y_axis_close(ax, miny, maxy)
        else:
            plt_helper.set_y_axis_log10(ax, miny, maxy)
        plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
    
    # plot legend and save figure
    plt_helper.plot_fig_legend(ncol=len(methods), font_size=13, legend_width=0.8)
    plt_helper.plot_and_save(output_folder, '%s_%d' % (fname, chosen_top_k))
    plt.show()


# ------------------------------------------------------------------------------
def plot_time_profile(fname, methods, datasets, dataset_labels, input_folder, 
    output_folder):
    
    num_row = 1
    num_col = len(datasets)
    fig_width, fig_height = calc_width_and_height2(num_col, num_row, 3.2, 2.2)
    
    plt_helper = PlotHelper(plt, fig_width, fig_height)
    plt_helper.plot_subplots_adjust(top_space=0.75)
    
    labels = ['BC', 'Ball', 'FH', 'NH']
    legends = ['Verification', 'Table Lookup', 'Lower Bounds', 'Others']
    width = 0.5
    colors = ['red', 'blue', 'forestgreen', 'blueviolet']
    
    # Cifar
    recalls = [87.5,87.4,88.05,83.91]
    total_times = [16.6636119,16.8091753,21.2639298,26.5400582]
    
    distance_times = [15.2988032,16.372755,19.303288,22.2796344] # distace
    lookup_times = [0,0,1.95718,4.2564253] # 
    lower_bound_times = [1.326908,0.3999694,0,0] 
    others_times = [0.0379004,0.0364508,0.0034616,0.0039987]
    
    lookup_distance_times = []
    for lookup,distance in zip(lookup_times,distance_times):
        lookup_distance_times.append(lookup+distance)
    lb_lookup_distance_times = []
    for lb,lookup_distance in zip(lower_bound_times, lookup_distance_times):
        lb_lookup_distance_times.append(lb+lookup_distance)
    
    ax = plt.subplot(1,2,1)
    ax.set_title('Cifar-10 ($d=512$)', fontsize=13)
    ax.set_ylabel('Query Time (ms)', fontsize=13)
    ax.set_yticks(np.arange(0, 28.7, step=9))
    
    ax.bar(labels, distance_times, width, label=legends[0], color=colors[0])
    ax.bar(labels, lookup_times, width, bottom=distance_times, label=legends[1], color=colors[1])
    ax.bar(labels, lower_bound_times, width, bottom=lookup_distance_times, label=legends[2], color=colors[2])
    ax.bar(labels, others_times, width, bottom=lb_lookup_distance_times, label=legends[3], color=colors[3])
    plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
    
    # Sun
    recalls = [89.9,89.3,87.49,85.96]
    total_times = [27.1990574,28.6885919,29.3981144,35.4865319]
    
    distance_times = [24.4788488,27.0822239,26.2729741,29.5984803]
    lookup_times = [0,0,3.1227891,5.884753]
    lower_bound_times = [2.6104424,1.4973298,0,0]
    others_times = [0.1097663,0.1090382,0.0023511,0.0032981]
    
    lookup_distance_times = []
    for lookup,distance in zip(lookup_times,distance_times):
        lookup_distance_times.append(lookup+distance)
    lb_lookup_distance_times = []
    for lb,lookup_distance in zip(lower_bound_times, lookup_distance_times):
        lb_lookup_distance_times.append(lb+lookup_distance)

    ax = plt.subplot(1,2,2)
    ax.set_title('Sun ($d=512$)', fontsize=13)
    # ax.set_ylabel('Query Time (ms)', fontsize=13)
    ax.set_yticks(np.arange(0, 37, step=12))
    
    ax.bar(labels, distance_times, width, color=colors[0])
    ax.bar(labels, lookup_times, width, bottom=distance_times, color=colors[1])
    ax.bar(labels, lower_bound_times, width, bottom=lookup_distance_times, color=colors[2])
    ax.bar(labels, others_times, width, bottom=lb_lookup_distance_times, color=colors[3])
    plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
    
    plt_helper.plot_fig_legend(ncol=len(legends), font_size=13, legend_width=0.95)
    plt_helper.plot_and_save(output_folder, '%s' % fname)


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    
    input_folder  = "../results/"
    output_folder = "../figures/"
    
    datasets = [
        'Music', 'GloVe100',
        'Sift', 'UKBench', 'Tiny1M', 'Msong', 'NUSW', 'Cifar', 
        'Sun', 'LabelMe', 'Gist', 'Enron', 'Trevi', 'P53']
    dataset_labels = [dataset_labels_map[dataset] for dataset in datasets]
    
    # 1. plot the curves of time vs. recall & fraction vs. recall
    chosen_top_k = 10
    methods = ['BC_Tree', 'Ball_Tree', 'FH', 'NH']
    plot_time_recall("time_recall", chosen_top_k, methods, datasets, 
        dataset_labels, input_folder, output_folder)
    plot_fraction_recall("fraction_recall", chosen_top_k, methods, datasets, 
        dataset_labels, input_folder, output_folder)
    
    # 2. plot the curves of time vs. k
    chosen_top_ks = [1, 10, 20, 40]
    methods = ['BC_Tree', 'Ball_Tree', 'FH', 'NH']
    recall_levels = [50, 60, 70, 80]
    for recall_level in recall_levels:
        plot_time_k("time_k", chosen_top_ks, recall_level, methods, datasets, 
            dataset_labels, input_folder, output_folder)
    
    # 3. plot the curves of time vs. recall for branch preference choice
    chosen_top_k = 10
    methods = ['BC_Tree', 'Ball_Tree']
    plot_preference_choice("preference", chosen_top_k, methods, datasets, 
        dataset_labels, input_folder, output_folder)
    
    # 4. plot leaf size for bc-treethe (curves of time vs. recall)
    chosen_top_k = 10
    methods = ['BC_Tree', 'Ball_Tree']
    for method in methods:
        plot_leaf_size("leaf_size", chosen_top_k, method, datasets, 
            input_folder, output_folder)
    
    # additional experiments
    # 5. plot the curves of time vs. k for lower bound validation
    chosen_top_ks = [1, 10, 20, 40]
    recall_level  = 80
    methods = ['BC-Tree', 'BC-Tree-wo-C', 'BC-Tree-wo-B', 'BC-Tree-wo-BC']
    result_names = ['BC_Tree.out', 'BC_Tree.out_cipc_ball', 
        'BC_Tree.out_cipc_cone', 'BC_Tree.out_cipc_none']
    plot_lower_bound_time_k("lower_bound_time_k", chosen_top_ks, recall_level, 
        methods, result_names, datasets, dataset_labels, input_folder, output_folder)
    
    # 6. plot the curves of time vs. recall & fraction vs. recall
    datasets = ['Deep100M', 'Sift100M']
    dataset_labels = [dataset_labels_map[dataset] for dataset in datasets]
    chosen_top_k = 10
    methods = ['BC_Tree', 'Ball_Tree', 'FH', 'NH']
    plot_scale_time_recall("scalability_time_recall", chosen_top_k, methods, 
        datasets, dataset_labels, input_folder, output_folder)
    
    # 7. plot time profile
    methods  = ['BC_Tree', 'Ball_Tree', 'FH', 'NH']
    datasets = ['Cifar', 'Sun']
    dataset_labels = [dataset_labels_map[dataset] for dataset in datasets]
    plot_time_profile("time_profile", methods, datasets, dataset_labels, 
        input_folder, output_folder)
    
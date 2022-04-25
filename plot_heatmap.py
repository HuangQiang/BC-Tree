import csv
import matplotlib
import matplotlib.pyplot as plt 
import numpy as np

from matplotlib import gridspec
from os         import makedirs
from os.path    import isdir, isfile, join

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42

dataset_labels_map = {
    'Sift':        'Sift',
    'UKBench':     'UKBench',
    'Tiny1M':      'Tiny',
    'Msong':       'Msong',
    'NUSW':        'NUSW',
    'Cifar':       'Cifar-10',
    'Sun':         'Sun',
    'LabelMe':     'LabelMe',
    'Gist':        'Gist',
    'Enron':       'Enron',
    'Trevi':       'Trevi',
    'P53':         'P53',
}

# ------------------------------------------------------------------------------
def read_csv(file_name):
    '''
    read 2-dim array (heatmap) from disk

    :params file_name: file name (string)
    :returns: 2-dim array
    '''
    data = []
    with open(file_name, newline='') as f:
        print(file_name)
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            one_data = []
            for r in row:
                one_data.append(float(r))
            data.append(one_data)
    # print(data)
    return data


# ------------------------------------------------------------------------------
def plot_heatmap(m, interval, datasets, input_folder, output_folder, 
    hist_width = 3.0, marginal_width = 0.8, subfigure_margin = 0.4, 
    hist_margin = 0.15, bar_width = 0.95):

    # read data
    dataset_labels = [dataset_labels_map[dataset] for dataset in datasets]
    base = 0; increment = m / interval; m_ticks = [base]
    for i in range(interval):
        base = base + increment
        m_ticks.append(base)

    all_data = []
    for dataset in datasets:
        file_name = input_folder + dataset + '/' + str(m) + '_Heatmap.out'
        data = read_csv(file_name)
        all_data.append(data)

    grids = np.array(all_data)

    # set up the width and height of the output figure
    nfigs = len(grids)
    plt.rcParams.update({'font.size': 14})

    fig_width = (hist_width+marginal_width)*nfigs + (subfigure_margin)*(nfigs-1)
    fig_height = (hist_width+marginal_width)

    width_ratios = [hist_width, marginal_width, 
        subfigure_margin-hist_margin] * (nfigs-1) + [hist_width, marginal_width]
    height_ratios = [marginal_width, hist_width]

    # specify grids
    fig = plt.figure(figsize=(fig_width, fig_height)) 
    gs  = gridspec.GridSpec(2, 3*nfigs-1, width_ratios=width_ratios, 
        height_ratios=height_ratios, wspace=hist_margin, hspace=hist_margin, 
        left=0.04, right=0.995, bottom=0.22, top=0.89) 
    
    # fig.title
    # fig.suptitle('The Heatmap', fontsize=16)

    for i, (grid, data_label) in enumerate(zip(grids, dataset_labels)):
        idx_top    = i*3
        idx_center = i*3 + (3*nfigs) - 1 
        idx_right  = i*3 + (3*nfigs) 

        #plot top marginal distribution
        ax0 = plt.subplot(gs[idx_top])
        gridx = np.sum(grid, axis=0)
        tmp_xs = range(0, len(gridx))
        # ax0.plot(gridx)
        ax0.bar(tmp_xs, gridx, width=bar_width)
        ax0.set_xlim(0-bar_width/2, len(gridx)-1+bar_width/2)
        ax0.set_xticks([])
        # plt.yticks(rotation=30)

        #x = hist_width/2 --> (hist_width+hist_margin+marginal_width)/2
        ax0.set_title(data_label, loc='center', y=1.1, 
            x=(hist_margin+marginal_width)/2+0.1, fontsize=18)

        #plot center 2d histogram
        ax1 = plt.subplot(gs[idx_center])
        plt.pcolormesh(grid)

        # ax1.set_xticks(m_ticks)
        # ax1.set_yticks(m_ticks)
        plt.xticks(m_ticks, rotation=60)
        plt.yticks(m_ticks)
        ax1.set_xlabel(r'$l_2$ norm $\Vert \mathbf{x} \Vert$ (%)', fontsize=16)
        if i == 0:
            ax1.set_ylabel(r'$|\cos~\theta|$ (%)', fontsize=16)

        #plot right marginal distribution
        ax2 = plt.subplot(gs[idx_right])
        gridy = np.sum(grid, axis=-1)
        # ax2.plot(gridy, tmp_xs)
        ax2.barh(tmp_xs, gridy, height=bar_width)
        ax2.set_ylim(0-bar_width/2, len(gridy)-1+bar_width/2)
        ax2.set_yticks([])
        plt.xticks(rotation=60)

    # plt.tight_layout()
    if not isdir(output_folder): makedirs(output_folder)
    plt.savefig('%sheatmap_%s_%d.pdf' % (output_folder, datasets[0], m))
    plt.savefig('%sheatmap_%s_%d.png' % (output_folder, datasets[0], m))
    # plt.savefig('%sheatmap_%s_%d.eps' % (output_folder, datasets[0], m))
    # plt.show()


# ------------------------------------------------------------------------------
if __name__ == '__main__':

    interval = 5
    m = 100
    
    datasets = ['Sift', 'UKBench', 'Tiny1M', 'Msong', 'NUSW', 'Cifar'] 
    input_folder  = '../results/'
    output_folder = '../figures/'
    plot_heatmap(m, interval, datasets, input_folder, output_folder)
    
    datasets = ['Sun', 'LabelMe', 'Gist', 'Enron', 'Trevi', 'P53']
    input_folder  = '../results/'
    output_folder = '../figures/'
    plot_heatmap(m, interval, datasets, input_folder, output_folder)

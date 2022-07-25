import numpy as np
import matplotlib.ticker as mticker
import matplotlib

from os import makedirs
from os.path import isdir, isfile, join

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42

# ------------------------------------------------------------------------------
class PlotHelper:
    def __init__(self, plt, fig_width, fig_height):
        self.plt = plt
        self.fig_width  = fig_width
        self.fig_height = fig_height
        self.minx = None
        self.maxx = None

    # --------------------------------------------------------------------------
    def plot_fig_legend(self, legend_width=0.9, legend_height=0.2, ncol=5, \
        font_size=15, columnspacing=None):
        '''
        draw the legend of figure
        '''
        bbox = (0.5-legend_width/2.0, 0.98-legend_height, legend_width, legend_height)
        
        self.plt.figlegend(fontsize=font_size, loc='upper center', 
            bbox_to_anchor=bbox, mode="expand", borderaxespad=0.05, ncol=ncol, 
            columnspacing=columnspacing)

    # --------------------------------------------------------------------------
    def plot_subplots_adjust(self, left_space=0.8, bottom_space=0.55, \
        top_space=0.9, right_space=0.25, wspace=0.24, hspace=0.45):
        '''
        adjust the size of sub-figure
        '''
        # define a window for a figure
        self.plt.figure(figsize=(self.fig_width, self.fig_height)) 
        self.plt.rcParams.update({'font.size': 13})
        
        bottom = bottom_space / self.fig_height
        top    = (self.fig_height - top_space) / self.fig_height
        left   = left_space / self.fig_width
        right  = (self.fig_width - right_space) / self.fig_width 
        
        self.plt.subplots_adjust(bottom=bottom, top=top, left=left, right=right, 
            wspace=wspace, hspace=hspace)

    # --------------------------------------------------------------------------
    def get_ticks(self, min_v, max_v, possible_ticks=[1, 10, 100], error=0):
        '''
        get ticks by min_v and max_v

        :params min_v: min value (float)
        :params max_v: max value (float)
        :returns: a list of ticks (list)
        '''
        for ipt, pt in enumerate(possible_ticks[::-1]):
            if pt <= min_v-error:
                pt_lower = len(possible_ticks)-ipt-1
                break
            
        for ipt, pt in enumerate(possible_ticks):
            if pt >= max_v+error:
                pt_upper = ipt
                break

        return possible_ticks[pt_lower:pt_upper+1]

    # --------------------------------------------------------------------------
    def get_max_tick(self, val, possible_ticks=[1, 10, 100], error=0):
        '''
        get max tick for a val in possible ticks

        :param val: input value (float)
        :return: max tick in possible_ticks (float)
        '''
        pt_upper = -1
        for ipt, pt in enumerate(possible_ticks):
            if pt >= val+error:
                pt_upper = ipt
                break
        
        if pt_upper == -1:
            return possible_ticks[-1]
        else:
            return possible_ticks[pt_upper]

    # --------------------------------------------------------------------------
    def set_y_axis_ratio(self, ax, miny, maxy):
        possible_ticks = [1., 1.02, 1.04, 1.06, 1.08]
        ticks = self.get_ticks(miny, maxy, possible_ticks)
        
        #disable minor ticks 
        ax.set_yticks([], minor=True)
        ax.set_yticks(ticks)
        ax.set_ylim(ticks[0], ticks[-1])

    # --------------------------------------------------------------------------
    def set_y_axis_log10(self, ax, miny, maxy):
        possible_ticks = [10**(i-10) for i in range(20)]
        ticks = self.get_ticks(miny, maxy, possible_ticks)
        
        #disable minor ticks 
        ax.set_yticks([], minor=True)
        ax.set_yticks(ticks)
        ax.set_ylim(ticks[0], ticks[-1])

    # --------------------------------------------------------------------------
    def set_y_axis_close(self, ax, miny, maxy, error=0.):
        possible_ticks = []
        for i in range(20):
            possible_ticks += [1*10**(i-10), 2*10**(i-10), 5*10**(i-10)]
        ticks = self.get_ticks(miny, maxy, possible_ticks, error)
        
        #disable minor ticks 
        if len(ticks) == 2:
            min_tick = ticks[0]; max_tick = ticks[-1]
            m = 4
            increament = (max_tick - min_tick) / 4
            ticks = [min_tick]
            for i in range(m):
                min_tick = min_tick + increament
                ticks.append(min_tick)

        ax.set_yticks([], minor=True)
        ticks_labels = ['%d' % t for t in ticks]
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticks_labels)
        ax.set_ylim(ticks[0], ticks[-1])
        
    # --------------------------------------------------------------------------
    def set_y_axis_equal(self, ax, miny, maxy, error=0.):
        possible_ticks = [16,30,50,70,120,180,240]
        for i in range(6):
            possible_ticks += [1*10**(i-3), 2*10**(i-3), 4*10**(i-3), 6*10**(i-3), 8*10**(i-3),]
        possible_ticks = sorted(possible_ticks)
        
        ticks = self.get_ticks(miny, maxy, possible_ticks, error)
        
        min_tick = ticks[0]
        max_tick = ticks[-1]
        m = 4;
        increament = (max_tick - min_tick) / 4
        
        ticks = [min_tick]
        for i in range(m):
            min_tick = min_tick + increament
            ticks.append(min_tick)
        
        ax.set_yticks([], minor=True)
        ticks_labels = ['%g' % t for t in ticks]
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticks_labels)
        ax.set_ylim(ticks[0], ticks[-1])
        
    # --------------------------------------------------------------------------
    def set_y_axis(self, ax, miny, maxy):
        # possible_ticks = [10, 20, 40, 100, 200, 400, 1000, 2000, 4000, 10000,
        #     20000, 40000, 100000]
        possible_ticks = []
        for i in range(20):
            possible_ticks += [1*10**(i-10), 2*10**(i-10), 5*10**(i-10)]
        max_tick = self.get_max_tick(maxy, possible_ticks)
        num = 4
        interval = max_tick / num
        ticks = [i*interval for i in range(num+1)]
        
        #disable minor ticks 
        ax.set_yticks([], minor=True)
        ax.set_yticks(ticks)
        ax.set_ylim(ticks[0], ticks[-1])

    # --------------------------------------------------------------------------
    def set_x_axis(self, ax, minx, maxx):
        possible_ticks = [10, 20, 40, 100, 200, 400, 1000, 2000, 4000, 10000,
            20000, 40000, 100000]
        max_tick = self.get_max_tick(maxx, possible_ticks)
        num = 4
        interval = max_tick / num
        ticks = [i*interval for i in range(num+1)]
        
        #disable minor ticks 
        ax.set_xticks([], minor=True)
        ax.set_xticks(ticks)
        ax.set_xlim(ticks[0], ticks[-1])

    # --------------------------------------------------------------------------
    def set_x_axis_log10(self, ax, minx, maxx):
        possible_ticks = [10**(i-10) for i in range(20)]
        ticks = self.get_ticks(minx, maxx, possible_ticks)
        
        #disable minor ticks 
        ax.set_xticks([], minor=True)
        ax.set_xticks(ticks)
        ax.set_xlim(ticks[0], ticks[-1])

    # --------------------------------------------------------------------------
    def plot_and_save(self, folder, name):
        if not isdir(folder): makedirs(folder)
        
        filename = join(folder, name)
        self.plt.savefig('%s.png' % filename, dpi=1000, bbox_inches="tight", 
            format='png')
        # self.plt.savefig('%s.eps' % filename, dpi=1000, bbox_inches="tight", 
        #     format='eps')
        self.plt.savefig('%s.pdf' % filename, dpi=1000, bbox_inches="tight", 
            format='pdf')
        # self.plt.show()


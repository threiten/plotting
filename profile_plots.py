import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
import matplotlib
import yaml
from plotting.plot_base import plotBase
matplotlib.use('Agg')

binsqpars = {'probeS4': (0, 2, 200001), 'probeR9': (0, 2, 200001), 'probeEtaWidth': (0, .1, 200001), 'probePhiWidth': (0, .5, 200001), 'probeSigmaIeIe': (0, .1, 400001), 'newPhoID': (-0.8, 1, 200001), 'probePt': (0, 200, 2000001), 'probeScEta': (-2.5, 2.5, 500001), 'probePhi': (-3.14, 3.14, 600001), 'rho': (0, 80, 800001), 'probeCovarianceIpIp': (0, 0.002, 200001), 'probeCovarianceIphiIphi': (
    0, 0.002, 200001), 'probeCovarianceIeIp': (-0.001, 0.001, 400001), 'probeCovarianceIetaIphi': (-0.001, 0.001, 400001), 'probeChIso03worst': (0, 20, 200001), 'probeChIso03': (0, 20, 20001), 'probeChIso03worst': (0, 20, 20001), 'probeEnergy': (0, 400, 400001), 'probePhoIso': (0, 20, 20001), 'probeScPreshowerEnergy': (0, 50, 250001), 'probeSigmaRR': (0, 1, 100001), 'newPhoIDtrIsoZ': (0, 1, 100001)}

dvar_bins = {'probePt': (25, 75, 51), 'probeScEta': {'EB': (-1.4442, 1.4442, 51), 'EE': (
    1.57, 2.5, 31)}, 'probePhi': (-3.14, 3.14, 61), 'rho': (0, 40, 41), 'run': (297050, 304797, 200)}


def wquantile(q, vals, bins, weights=None):
    centres = 0.5*(bins[1:]+bins[:-1])
    hist, _ = np.histogram(vals, bins=bins, weights=weights)
    cum_hist = np.cumsum(hist, dtype=float)
    cum_hist_n = cum_hist/cum_hist[-1]
    ind_high_bound = np.searchsorted(cum_hist_n, q)
    ind_low_bound = ind_high_bound-1
    inds = np.sort(np.ravel(np.array([ind_low_bound, ind_high_bound])))
    q_vals = np.interp(q, cum_hist_n[inds], centres[inds])
    return q_vals


def wquantile_unb(q, vals, weights):
    df = pd.DataFrame()
    df['weight'] = weights
    df['val'] = vals
    sort_df = df.sort_values('val')
    w_cum = np.cumsum(sort_df['weight'].values)
    cdf = np.vstack(((w_cum/w_cum[-1]), sort_df['val'].values))
    ind = np.searchsorted(cdf[0], q)
    return cdf[1][ind]


class profilePlot:

    def __init__(self, df_mc, df_data, nq, bintpe, var, diff_var, weightst_mc, EBEE, zoom=False, corrlabel=None, addlabel=None, label='', addlegd=None, corrname=None, weightst_data=None):

        self.var = var
        self.diff_var = diff_var
        self.nq = nq
        self.bintpe = bintpe
        self.weightst_mc = weightst_mc
        self.EBEE = EBEE
        self.corrlabel = corrlabel
        self.addlabel = addlabel
        self.qts = [0.5, .25, .75, .1, .9]
        self.label = label
        self.addlegd = addlegd
        self.zoom = zoom
        self.name = 'profile_' + diff_var + '_' + var + '_' + bintpe + '_' + label
        self.title = var + ' ' + label.replace('_', ' ')
        self.corrname = r'\textbf{\textit{sim corr}}'
        self.colors = list(matplotlib.cm.Dark2.colors)

        self.plotB = plotBase(df_mc, var, weightst_mc, label, 'profile')
        with open('/t3home/threiten/python/plotting/texReplacement.yaml') as f:
            self.tex_replace_dict = yaml.load(f, Loader=yaml.FullLoader)
            f.close()

        self.xloc = 'right'
        self.yloc = 'top'

        if corrname != None:
            self.corrname = 'sim ' + corrname
        if self.zoom:
            self.name = self.name + '_zoom'

        self.mc = df_mc.loc[:, [self.var, self.diff_var, weightst_mc]]

        self.corr = False
        if corrlabel != None:
            self.mc[self.var+self.corrlabel] = df_mc[self.var+self.corrlabel]
            self.corr = True

        self.add = False
        if addlabel != None:
            self.mc[self.var+self.addlabel] = df_mc[self.var+self.addlabel]
            self.add = True

        if weightst_data == None:
            self.data = df_data.loc[:, [self.var, self.diff_var]]
            self.data['weight_dumm'] = np.ones(self.data.index.size)
            self.weightst_data = 'weight_dumm'
        else:
            self.data = df_data.loc[:, [
                self.var, self.diff_var, weightst_data]]
            self.weightst_data = weightst_data

        if self.bintpe == 'equ':
            self.data[self.diff_var+'_bin'], self.diff_bins = pd.qcut(
                self.data[self.diff_var], self.nq, labels=np.arange(self.nq), retbins=True)
            self.centers = 0.5*(self.diff_bins[1:]+self.diff_bins[:-1])
            self.mc[self.diff_var+'_bin'] = pd.cut(
                self.mc[self.diff_var], bins=self.diff_bins, labels=np.arange(self.nq))

        elif self.bintpe == 'lin':
            if self.diff_var == 'probeScEta':
                if self.EBEE == 'EB':
                    self.diff_bins = np.linspace(
                        *dvar_bins[self.diff_var][self.EBEE])
                elif self.EBEE == 'EE':
                    self.diff_bins = np.hstack(
                        (-np.flip(np.linspace(*dvar_bins[self.diff_var][self.EBEE]), 0), np.linspace(*dvar_bins[self.diff_var][self.EBEE])))
            else:
                self.diff_bins = np.linspace(*dvar_bins[self.diff_var])

            self.centers = 0.5*(self.diff_bins[1:]+self.diff_bins[:-1])

            self.data[self.diff_var+'_bin'] = pd.cut(
                self.data[self.diff_var], bins=self.diff_bins, labels=self.centers)
            self.mc[self.diff_var+'_bin'] = pd.cut(
                self.mc[self.diff_var], bins=self.diff_bins, labels=self.centers)
            if self.diff_var == 'probeScEta':
                self.centers = np.delete(self.centers, np.where((abs(self.centers) < dvar_bins[self.diff_var][self.EBEE][0]) | (
                    abs(self.centers) > dvar_bins[diff_var][self.EBEE][1])))

#            self.nq = len(self.centers)

        else:
            print('Please choose bintype')
            sys.exit()

        self.datagb = self.data.groupby(self.diff_var+'_bin')
        self.mcgb = self.mc.groupby(self.diff_var+'_bin')

        # try:
        #     self.binsq = np.linspace(*binsqpars[self.var])
        # except KeyError:
        #     print('No predefined binparameters for ' + self.var + '. Please set yourself using set_binsq(min,max,#bins)')

    def get_quantiles(self):

        if self.bintpe == 'equ':
            self.data_quantiles = np.vstack([wquantile_unb(self.qts, self.datagb[self.var].get_group(
                i).values, weights=self.datagb[self.weightst_data].get_group(i).values) for i in np.arange(self.nq)])
            self.mc_quantiles = np.vstack([wquantile_unb(self.qts, self.mcgb[self.var].get_group(
                i).values, weights=self.mcgb[self.weightst_mc].get_group(i).values) for i in np.arange(self.nq)])
            if self.corr:
                self.mc_c_quantiles = np.vstack([wquantile_unb(self.qts, self.mcgb[self.var+self.corrlabel].get_group(
                    i).values, weights=self.mcgb[self.weightst_mc].get_group(i).values) for i in np.arange(self.nq)])
            if self.add:
                self.mc_add_quantiles = np.vstack([wquantile_unb(self.qts, self.mcgb[self.var+self.addlabel].get_group(
                    i).values, weights=self.mcgb[self.weightst_mc].get_group(i).values) for i in np.arange(self.nq)])
        elif self.bintpe == 'lin':
            self.data_quantiles = np.vstack([wquantile_unb(self.qts, self.datagb[self.var].get_group(
                i).values, self.datagb[self.weightst_data].get_group(i).values) for i in self.centers])
            self.mc_quantiles = np.vstack([wquantile_unb(self.qts, self.mcgb[self.var].get_group(
                i).values, self.mcgb[self.weightst_mc].get_group(i).values) for i in self.centers])
            if self.corr:
                self.mc_c_quantiles = np.vstack([wquantile_unb(self.qts, self.mcgb[self.var+self.corrlabel].get_group(
                    i).values, weights=self.mcgb[self.weightst_mc].get_group(i).values) for i in self.centers])
            if self.add:
                self.mc_add_quantiles = np.vstack([wquantile_unb(self.qts, self.mcgb[self.var+self.addlabel].get_group(
                    i).values, weights=self.mcgb[self.weightst_mc].get_group(i).values) for i in self.centers])

    def plot_profile(self, xunit='', yunit=''):

        rcP = {'text.usetex': True,
               'font.family': 'sans-serif',
               'font.sans-serif': ['Helvetica'],
               'pdf.fonttype': 42,
               'axes.labelsize': 20,
               'font.size': 16,
               'pgf.rcfonts': True,
               'text.latex.preamble': r"\usepackage{bm, xspace, amsmath}"}

        plt.rcParams.update(rcP)

        fig, axes = plt.subplots(1, figsize=(9, 7))

        hatchArr = [
            '////', r'//'.replace(' ', ''), r'///'.replace(' ', ''), '///']

        axes.plot(self.centers, self.data_quantiles[:, 0],
                  '-', markersize=0, color=self.colors[0], label=r'\textbf{\textit{data}}', linewidth=4)
        axes.fill_between(
            self.centers, self.data_quantiles[:, 1], self.data_quantiles[:, 2], hatch=hatchArr[0], edgecolor=self.colors[0], alpha=0.7, facecolor='None', linewidth=0)
        axes.plot(
            self.centers, self.data_quantiles[:, 3], '--', linewidth=3, color=self.colors[0])
        axes.plot(
            self.centers, self.data_quantiles[:, 4], '--', linewidth=3, color=self.colors[0])

        axes.plot(self.centers, self.mc_quantiles[:, 0], '-',
                  markersize=0, color=self.colors[1], label=r'\textbf{\textit{sim}}', linewidth=4)
        axes.fill_between(
            self.centers, self.mc_quantiles[:, 1], self.mc_quantiles[:, 2], hatch=hatchArr[1], edgecolor=self.colors[1], alpha=0.7, facecolor='None', linewidth=0)
        axes.plot(self.centers,
                  self.mc_quantiles[:, 3], '--', linewidth=3, color=self.colors[1])
        axes.plot(self.centers,
                  self.mc_quantiles[:, 4], '--', linewidth=3, color=self.colors[1])

        if self.add:
            axes.plot(self.centers, self.mc_add_quantiles[:, 0], '-',
                      markersize=0, color=self.colors[3], label=self.addlegd, linewidth=4)
            axes.fill_between(
                self.centers, self.mc_add_quantiles[:, 1], self.mc_add_quantiles[:, 2], hatch=hatchArr[3], edgecolor=self.colors[3], alpha=0.7, facecolor='None', linewidth=0)
            axes.plot(
                self.centers, self.mc_add_quantiles[:, 3], '--', linewidth=3, color=self.colors[3])
            axes.plot(
                self.centers, self.mc_add_quantiles[:, 4], '--', linewidth=3, color=self.colors[3])

        if self.corr:
            axes.plot(self.centers, self.mc_c_quantiles[:, 0], '-',
                      markersize=0, color=self.colors[2], label=self.corrname, linewidth=4)
            axes.fill_between(
                self.centers, self.mc_c_quantiles[:, 1], self.mc_c_quantiles[:, 2], hatch=hatchArr[2], edgecolor=self.colors[2], alpha=0.7, facecolor='None', linewidth=0)
            axes.plot(
                self.centers, self.mc_c_quantiles[:, 3], '--', linewidth=3, color=self.colors[2])
            axes.plot(
                self.centers, self.mc_c_quantiles[:, 4], '--', linewidth=3, color=self.colors[2])

        self.set_ylim()
        axes.set_ylim(0.85*self.ylim[0], self.ylim[1])
        if self.diff_var == 'probePt':
            axes.set_xlim(25, 60)
        # if xunit == None:
        #     xunit = ''
        # if yunit == None:
        #     yunit = ''
        # axes.xlabel(r'%s' % (self.diff_var.replace('_', '\textunderscore ')))
        # axes.ylabel(r'%s' % (self.var.replace('_', '\textunderscore ')))

        if self.var in self.tex_replace_dict:
            math, var, unit = self.plotB.parse_repl(
                self.tex_replace_dict[self.var])
            if unit == '':
                axes.set_ylabel(
                    r'$\boldsymbol{{{0}}}$'.format(var, unit), fontsize=20, loc=self.yloc)
            else:
                axes.set_ylabel(r'$\boldsymbol{{{0}}}\,\,\left[\textnormal{{{1}}}\right]$'.format(
                    var, unit), fontsize=20, loc=self.yloc)
        else:
            axes.set_ylabel(r'\textit{{{0}}}'.format(
                self.var.replace('_', '\_')), fontsize=20, loc=self.yloc)

        if self.diff_var in self.tex_replace_dict:
            math, var, unit = self.plotB.parse_repl(
                self.tex_replace_dict[self.diff_var])
            if unit == '':
                axes.set_xlabel(
                    r'$\boldsymbol{{{0}}}$'.format(var, unit), fontsize=20, loc=self.xloc)
            else:
                axes.set_xlabel(r'$\boldsymbol{{{0}}}\,\,\left[\textnormal{{{1}}}\right]$'.format(
                    var, unit), fontsize=20, loc=self.xloc)
        else:
            axes.set_xlabel(r'\textit{{{0}}}'.format(
                self.var.replace('_', '\_')), fontsize=20, loc=self.xloc)

        # plt.title(self.title,y=1.05)
        # fig.text(
        #     0.13, .91, r'\textbf{CMS} \textit{Work in Progress}', fontsize=11)
        axes.legend(loc='best', framealpha=0)
        LegHandles, LegLabels = axes.get_legend_handles_labels()
        for i, lab in enumerate(LegLabels):
            legPatch = matplotlib.patches.Patch(edgecolor=LegHandles[i].get_color(
            ), facecolor='None', hatch=hatchArr[i], linewidth=0, zorder=3)
            legLine = matplotlib.lines.Line2D([], [], color=LegHandles[i].get_color(
            ), marker='None', zorder=3, linewidth=2)
            # legLineT = matplotlib.lines.Line2D([], [], color=LegHandles[i].get_color(
            # ), marker='None', zorder=3, linewidth=1, linestyle='--')
            LegHandles[i] = (legPatch, legLine)
        axes.legend(labels=LegLabels, handles=LegHandles,
                    fontsize=16, framealpha=0, loc='best')

        self.fig = fig

    def set_ylim(self):
        if self.zoom:
            self.ylim = [self.data_quantiles[:, 0].min()-.05*self.data_quantiles[:, 0].min(
            ), self.data_quantiles[:, 0].max()+.05*self.data_quantiles[:, 0].max()]
        else:
            self.ylim = [self.data_quantiles[:, 3].min()-.05*self.data_quantiles[:, 3].min(
            ), self.data_quantiles[:, 4].max()+.05*self.data_quantiles[:, 4].max()]

    def set_diffbins(self, mn, mx, nbins):
        self.diff_bins = np.linspace(mn, mx, nbins+1)

    def save(self, outDir):
        self.fig.savefig(outDir + '/' + self.name +
                         '.png', bbox_inches='tight')
        self.fig.savefig(outDir + '/' + self.name +
                         '.pdf', bbox_inches='tight')

    def set_binsq(self, low, high, nob):
        self.binsq = np.linspace(low, high, nob+1)

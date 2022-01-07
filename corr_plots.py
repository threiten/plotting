import numpy as np
import pandas as pd
import matplotlib
import yaml
from plotting.plot_base import plotBase
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


def wcorr(arr1, arr2, weights):
    m1 = np.average(arr1, weights=weights)*np.ones_like(arr1)
    m2 = np.average(arr2, weights=weights)*np.ones_like(arr2)
    cov_11 = float((weights*(arr1-m1)**2).sum()/weights.sum())
    cov_22 = float((weights*(arr2-m2)**2).sum()/weights.sum())
    cov_12 = float((weights*(arr1-m1)*(arr2-m2)).sum()/weights.sum())
    return cov_12/np.sqrt(cov_11*cov_22)


class corrMat:

    def __init__(self, df_mc, df_data, varrs, varrs_corr, weightst, label=''):

        self.label = label
        self.varrs = varrs
        self.varrs_corr = varrs_corr

        mc_crl = np.zeros((len(varrs), len(varrs)))
        mc_c_crl = np.zeros((len(varrs_corr), len(varrs_corr)))
        data_crl = np.zeros((len(varrs), len(varrs)))

        for i, var1 in enumerate(self.varrs):
            mc_crl[i, :] = np.hstack((np.array([100*wcorr(df_mc[var1].values, df_mc[var2].values, df_mc[weightst])
                                                for var2 in varrs[:varrs.index(var1)+1]]), np.zeros(len(varrs)-(varrs.index(var1)+1))))
            data_crl[i, :] = np.hstack((np.array([100*wcorr(df_data[var1].values, df_data[var2].values,
                                                            df_data['weight'].values) for var2 in varrs[:varrs.index(var1)+1]]), np.zeros(len(varrs)-(varrs.index(var1)+1))))

        for i, var1 in enumerate(self.varrs_corr):
            mc_c_crl[i, :] = np.hstack((np.array([100*wcorr(df_mc[var1].values, df_mc[var2].values, df_mc[weightst])
                                                  for var2 in varrs_corr[:varrs_corr.index(var1)+1]]), np.zeros(len(varrs_corr)-(varrs_corr.index(var1)+1))))

        self.mc_crl = mc_crl
        self.mc_c_crl = mc_c_crl
        self.data_crl = data_crl

        # self.mc_crl = np.array([[100*wcorr(df_mc[var1].values, df_mc[var2].values,
        #                                    df_mc[weightst]) for var2 in varrs[:varrs.index(var1)+1]] for var1 in varrs])
        # self.mc_c_crl = np.array([[100*wcorr(df_mc[var1].values, df_mc[var2].values,
        #                          df_mc[weightst]) for var2 in varrs_corr[:varrs_corr.index(var1)+1]] for var1 in varrs_corr])

        # self.data_crl = np.array([[100*wcorr(df_data[var1].values, df_data[var2].values,
        #                          df_data['weight'].values) for var2 in varrs[:varrs.index(var1)+1]] for var1 in varrs])

        self.fig_name = []

        self.mc_crl_meanabs = np.mean(np.abs(self.mc_crl))
        self.mc_c_crl_meanabs = np.mean(np.abs(self.mc_c_crl))
        self.data_crl_meanabs = np.mean(np.abs(self.data_crl))

        self.diff_crl_meanabs = np.mean(
            np.abs(np.array(self.mc_crl)-np.array(self.data_crl)))
        self.diff_c_crl_meanabs = np.mean(
            np.abs(np.array(self.mc_c_crl)-np.array(self.data_crl)))

        self.plotB = plotBase(df_mc, varrs[0], weightst, label, 'profile')
        with open('/t3home/threiten/python/plotting/texReplacement.yaml') as f:
            self.tex_replace_dict = yaml.load(f, Loader=yaml.FullLoader)
            f.close()

        self.texlabel = r''
        c_list = self.label.split()
        for st in c_list:
            repl = self.plotB.get_tex_repl(st)
            if repl[0] != '$' and repl[-3:] != '$\\':
                self.texlabel += '$' + repl[:-2] + r'$\\'
            else:
                self.texlabel += repl

        rcP = {'text.usetex': True,
               'font.family': 'sans-serif',
               'font.sans-serif': ['Helvetica'],
               'pdf.fonttype': 42,
               'axes.labelsize': 20,
               'font.size': 16,
               'pgf.rcfonts': True,
               'text.latex.preamble': r"\usepackage{bm, xspace, amsmath}"}

        plt.rcParams.update(rcP)

    def plot_corr_mat(self, key):

        self.key = key
        fig1 = plt.figure(figsize=(10, 10))
        ax1 = fig1.add_subplot(111)

        plt.set_cmap('bwr')

        if key == 'data':
            cax1 = ax1.matshow(self.data_crl, vmin=-100, vmax=100)
            self.plot_numbers(ax1, self.data_crl)
            # plt.title(r'Correlation data ' + self.label.replace('_', ' ') +
            #          ' Mean abs: {:.3f}'.format(self.data_crl_meanabs), y = 1.4)
            ax1.text(0.95, 0.95, r'\begin{{flushright}}$\boldsymbol{{Data}}$\\{{{0}}}\end{{flushright}}'.format(
                self.texlabel).replace(' ', '\,\,'), transform=ax1.transAxes, va='top', ha='right', fontsize=20)
            name = 'data_' + self.label
        elif key == 'mc':
            cax1 = ax1.matshow(self.mc_crl, vmin=-100, vmax=100)
            self.plot_numbers(ax1, self.mc_crl)
            # plt.title(r'Correlation mc ' + self.label.replace('_', ' ') +
            #          ' Mean abs: {:.3f}'.format(self.mc_crl_meanabs), y=1.4)
            ax1.text(0.95, 0.95, r'\begin{{flushright}}$\boldsymbol{{Simulation}}$\\{0}\end{{flushright}}'.format(
                self.texlabel).replace(' ', '\,\,'), transform=ax1.transAxes, va='top', ha='right', fontsize=20)
            name = 'mc_' + self.label
        elif key == 'mcc':
            cax1 = ax1.matshow(self.mc_c_crl, vmin=-100, vmax=100)
            self.plot_numbers(ax1, self.mc_c_crl)
            # plt.title(r'Correlation mc corrected ' + self.label.replace('_',
            #          ' ') + ' Mean abs: {:.3f}'.format(self.mc_c_crl_meanabs), y=1.4)
            ax1.text(0.95, 0.95, r'\begin{{flushright}}$\boldsymbol{{Simulation corrected}}$\\{0}\end{{flushright}}'.format(
                self.texlabel).replace(' ', '\,\,'), transform=ax1.transAxes, va='top', ha='right', fontsize=20)
            name = 'mc_corr_' + self.label
        elif key == 'diff':
            cax1 = ax1.matshow(np.array(self.mc_crl) -
                               np.array(self.data_crl), vmin=-15, vmax=15)
            self.plot_numbers(ax1, np.array(self.mc_crl) -
                              np.array(self.data_crl))
            # plt.title(r'Correlation difference ' + self.label.replace('_',
            #          ' ') + ' Mean abs: {:.3f}'.format(self.diff_crl_meanabs), y=1.4)
            ax1.text(0.95, 0.95, r'\begin{{flushright}}$\boldsymbol{{Difference}}$\\{0}\end{{flushright}}'.format(
                self.texlabel).replace(' ', '\,\,'), transform=ax1.transAxes, va='top', ha='right', fontsize=20)
            name = 'diff_' + self.label
        elif key == 'diffc':
            cax1 = ax1.matshow(np.array(self.mc_c_crl) -
                               np.array(self.data_crl), vmin=-15, vmax=15)
            self.plot_numbers(ax1, np.array(self.mc_c_crl) -
                              np.array(self.data_crl))
            # plt.title(r'Correlation difference corrected ' + self.label.replace('_',
            #          ' ') + ' Mean abs: {:.3f}'.format(self.diff_c_crl_meanabs), y=1.4)
            ax1.text(0.95, 0.95, r'\begin{{flushright}}$\boldsymbol{{Difference corrected}}$\\{0}\end{{flushright}}'.format(
                self.texlabel).replace(' ', '\,\,'), transform=ax1.transAxes, va='top', ha='right', fontsize=20)
            name = 'diff_corr_' + self.label

        cbar = fig1.colorbar(cax1)
        cbar.set_label(r'\textbf{\textit{Correlation (\%)}}')

        # for i in range(len(self.varrs)):
        #     self.varrs[i] = self.varrs[i].replace('probe', '')
        ax1.set_yticks(np.arange(len(self.varrs)))
        ax1.set_xticks(np.arange(len(self.varrs)))

        ticklabels = []
        for varr in self.varrs:
            math, var, unit = self.plotB.parse_repl(
                self.tex_replace_dict[varr])
            ticklabels.append(r'$\boldsymbol{{{0}}}$'.format(var))

        ax1.set_xticklabels(ticklabels, rotation='vertical')
        ax1.set_yticklabels(ticklabels)

        self.fig_name.append((fig1, name))

    def plot_numbers(self, ax, mat):
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                c = mat[j, i]
                if np.abs(c) >= 1:
                    ax.text(i, j, r'${:.0f}$'.format(c), fontdict={
                            'size': 14}, va='center', ha='center')

    def save(self, outDir):
        for fig, name in self.fig_name:
            fig.savefig(outDir + '/crl_' + name.replace(' ',
                        '_') + '.png', bbox_inches='tight')
            fig.savefig(outDir + '/crl_' + name.replace(' ',
                        '_') + '.pdf', bbox_inches='tight')

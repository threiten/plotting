import pickle
import gzip
import copy
import yaml

import matplotlib.pyplot as plt
import matplotlib.cm as cm

cmsTextDic = {
    'wip': r'\textbf{CMS} \textit{Work in Progress}',
    'prelim': r'\textbf{CMS} \textit{Preliminary}',
    'final': r'\textbf{CMS}',
    'sim': r'\textbf{CMS} \textit{Simulation}'
}


class plotBase(object):

    def __init__(self, df_mc, var, weightstr_mc, label, type, df_data=None, **kwargs):

        self.var = var
        self.weightstr_mc = weightstr_mc
        if label is not None:
            self.title = '{} {}'.format(var, label).replace('_', ' ')
        else:
            self.title = '{}'.format(var).replace('_', ' ')
        self.pl_tpe = type

        if 'cut' in kwargs:
            df_mc_read = df_mc.query(kwargs['cut'], engine='python')
            if df_data is not None:
                df_data_read = df_data.query(kwargs['cut'], engine='python')
            self.cut = kwargs['cut']
        else:
            df_mc_read = df_mc
            if df_data is not None:
                df_data_read = df_data

        if 'lumi' in kwargs:
            self.lumi = kwargs['lumi']

        self.lumiStr = None
        if 'lumiStr' in kwargs:
            self.lumiStr = kwargs['lumiStr']
            if 'lumi' in kwargs:
                if kwargs['lumi'] != kwargs['lumiStr']:
                    print(
                        'Warning: lumi and lumi to be written on the plot not the same!')

        if 'exts' in kwargs:
            self.mc_vars = [var] + \
                ['{}{}'.format(var, ext) for ext in kwargs['exts']]
        else:
            self.mc_vars = [var]

        if 'num' in kwargs:
            self.title += ' {}'.format(kwargs['num'])

        self.cmsText = None
        if 'cmsText' in kwargs:
            self.cmsText = kwargs['cmsText']
        else:
            self.cmsText = None

        self.mc = df_mc_read.loc[:, self.mc_vars]
        if df_data is not None:
            self.data = df_data_read.loc[:, [var]].values
        else:
            self.data = None

        self.mc_weights = df_mc_read.loc[:, [weightstr_mc]].values
        self.mc_weights_cache = df_mc_read.loc[:, [weightstr_mc]].values

        with open('/t3home/threiten/python/plotting/texReplacement.yaml') as f:
            self.tex_replace_dict = yaml.load(f, Loader=yaml.FullLoader)
            f.close()

        if 'weightstr_data' in kwargs and df_data is not None:
            self.weightstr_data = kwargs['weightstr_data']
            self.data_weights = df_data_read.loc[:, [
                kwargs['weightstr_data']]].values

        if 'xgrid' in kwargs:
            self.xgrid = kwargs['xgrid']

        if 'ygrid' in kwargs:
            self.ygrid = kwargs['ygrid']

        if 'cut_str' in kwargs:
            self.cut_str = kwargs['cut_str']

        if 'leg_loc' in kwargs:
            self.leg_loc = kwargs['leg_loc'].replace('_', ' ')

        if 'colors' in kwargs:
            self.colors = kwargs['colors']
        else:
            self.colors = list(cm.Dark2.colors)

        self.xloc = None
        if 'xloc' in kwargs:
            self.xloc = kwargs['xloc']

    def save(self, out_dir, save_dill=False):

        if not hasattr(self, 'fig'):
            raise AttributeError("Draw figure before saving it!")

        self.fig.savefig('{}/{}_{}.png'.format(out_dir, self.pl_tpe,
                         self.title).replace(' ', "_"), bbox_inches='tight')
        self.fig.savefig('{}/{}_{}.pdf'.format(out_dir, self.pl_tpe,
                         self.title).replace(' ', "_"), bbox_inches='tight')
        if save_dill:
            self.dill_save('{}/{}_{}.pkl'.format(out_dir,
                           self.pl_tpe, self.title).replace(' ', "_"))

    def dill_save(self, fname):

        pickle.dump(self.fig, gzip.open('{}.gz'.format(fname), mode='wb'))

    def set_style(self):

        rcP = {'text.usetex': True,
               'font.family': 'sans-serif',
               'font.sans-serif': ['Helvetica'],
               'pdf.fonttype': 42,
               'axes.labelsize': 20,
               'font.size': 16,
               'pgf.rcfonts': True,
               'text.latex.preamble': r"\usepackage{bm, xspace, amsmath}"}
        # 'text.latex.preview': True}
        plt.rcParams.update(rcP)

    @staticmethod
    def parse_repl(repl):

        ret = repl.split(':')

        if len(ret) == 1:
            ret = ['', ret[0], '']
        elif len(ret) == 2:
            if ret[0] == 'math':
                ret.append('')
            else:
                ret = [''] + ret

        if ret[0] == 'math':
            ret[0] = True
        else:
            ret[0] = False

        return tuple(ret)

    def get_tex_repl(self, st):

        lest = len(st)
        stLis = [st[i:j+1] for i in range(lest) for j in range(i, lest)]
        stLis.sort(key=lambda s: len(s))
        unit = ''
        mat = False
        ret = st
        for sti in stLis:
            if sti in self.tex_replace_dict:
                mat, repl, unit = self.parse_repl(self.tex_replace_dict[sti])
                ret = st.replace(sti, repl)

        if unit != '':
            ret += r'~\textnormal{{{0}}}'.format(unit)
        if mat:
            return r'${0}$\\'.format(ret)
        elif ret == '':
            return r''
        else:
            return r'{0}\\'.format(ret)

    def get_tex_cut(self):

        c_list = self.cut.replace('and', '').split()

        self.cut_str_tex = r''
        for st in c_list:
            self.cut_str_tex += self.get_tex_repl(st)

    def normalize_mc(self):

        self.mc_weights = copy.deepcopy(self.mc_weights_cache)

        if hasattr(self, 'lumi'):
            self.mc_weights *= self.lumi
        else:
            if hasattr(self, 'data_weights'):
                self.mc_weights *= self.data_weights.sum() / self.mc_weights.sum()
            else:
                self.mc_weights *= self.data.shape[0] / self.mc_weights.sum()

    @staticmethod
    def drawCMSLogo(ax, opt, fs=22):

        ax.text(0, 1, cmsTextDic[opt], fontsize=fs,
                transform=ax.transAxes, va='bottom')

    @staticmethod
    def drawIntLumi(ax, intL=138, fs=22):

        if intL is None:
            ax.text(1, 1, r'13\ensuremath{\,\text{Te\hspace{-.08em}V}}\xspace',
                    fontsize=fs, transform=ax.transAxes, ha='right', va='bottom')
        else:
            ax.text(1, 1, r'{}\mbox{{\ensuremath{{\,\text{{fb}}^{{-1}}}}}}\xspace (13\ensuremath{{\,\text{{Te\hspace{{-.08em}}V}}}}\xspace)'.format(
                intL), fontsize=fs, transform=ax.transAxes, ha='right', va='bottom')

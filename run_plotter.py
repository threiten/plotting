from joblib import delayed, Parallel
from qRC.tmva.IdMVAComputer import helpComputeIdMva
import qRC.syst.qRC_systematics as syst
import plotting.plot_dmc_hist as pldmc
import plotting.parse_yaml as pyml
import root_pandas
import argparse
import itertools
import warnings
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore')


def remove_duplicates(vars):
    mask = len(vars) * [True]
    for i in range(len(vars)):
        for j in range(i+1, len(vars)):
            if vars[i] == vars[j]:
                mask[j] = False

    return list(itertools.compress(vars, mask))


def chunked_loader(fpath, columns, **kwargs):
    fitt = pd.read_hdf(fpath, columns=columns,
                       chunksize=10000, iterator=True, **kwargs)
    df = pd.concat(fitt, copy=False)

    return df


def make_unique_names(plt_list):

    mtl_list = len(plt_list) * [0]
    for i in range(len(plt_list)):
        mult = 0
        for j in range(i):
            if plt_list[i]['type'] == plt_list[j]['type'] and plt_list[i]['var'] == plt_list[j]['var']:
                mult += 1

        mtl_list[i] = mult

    for i in range(len(plt_list)):
        plt_list[i]['num'] = mtl_list[i]

    return plt_list


def make_vars(plot_dict, extra=[], extens=True):
    ret = []
    for dic in plot_dict:
        ret.append(dic['var'])
        if 'exts' in dic.keys() and extens:
            for ext in dic['exts']:
                ret.append(dic['var'] + ext)

    ret.extend(extra)

    return remove_duplicates(ret)


def check_vars(df, varrs):

    varmiss = len(varrs) * [False]
    for i, var in enumerate(varrs):
        if not var in df.columns:
            varmiss[i] = True

    return varmiss


def main(options):

    plot_dict = make_unique_names(pyml.yaml_parser(options.config)())
    varrs = make_vars(plot_dict, ['weight', 'weight_clf', 'probePassEleVeto', 'tagPt', 'tagScEta',
                      'probeScEnergy', 'probeSigmaRR'])  # 'probeEtaWidth_Sc', 'probePhiWidth_Sc',
    varrs_data = make_vars(plot_dict, ['weight', 'probePassEleVeto', 'tagPt', 'tagScEta',
                           'probeScEnergy', 'probeSigmaRR'], extens=False)  # , 'probeEtaWidth_Sc', 'probePhiWidth_Sc'

    # if 'probePhoIso03_uncorr' in varrs:
    #     varrs.pop(varrs.index('probePhoIso03_uncorr'))

    if options.mc.split('.')[-1] == 'root':
        if options.mc_tree is None:
            raise NameError(
                'mc_tree has to be in options if a *.root file is used as input')
        df_mc = root_pandas.read_root(
            options.mc, options.mc_tree, columns=varrs)
    else:
        df_mc = pd.read_hdf(options.mc, columns=varrs)

    if options.data.split('.')[-1] == 'root':
        if options.data_tree is None:
            raise NameError(
                'data_tree has to be in options if a *.root file is used as input')
        df_data = root_pandas.read_root(
            options.data, options.data_tree, columns=varrs_data)
    else:
        df_data = pd.read_hdf(options.data, columns=varrs_data)

    if 'weight_clf' not in df_mc.columns and not options.no_reweight:
        if options.reweight_cut is not None:
            df_mc['weight_clf'] = syst.utils.clf_reweight(
                df_mc, df_data, n_jobs=10, cut=options.reweight_cut)
        else:
            warnings.warn(
                'Cut for reweighting is taken from 0th and 1st plot. Make sure this is the right one')
            if 'abs(probeScEta)<1.4442' in plot_dict[0]['cut'] and 'abs(probeScEta)>1.56' in plot_dict[1]['cut']:
                df_mc.loc[np.abs(df_mc['probeScEta']) < 1.4442, 'weight_clf'] = syst.utils.clf_reweight(
                    df_mc.query('abs(probeScEta)<1.4442'), df_data, n_jobs=10, cut=plot_dict[0]['cut'])
                df_mc.loc[np.abs(df_mc['probeScEta']) > 1.56, 'weight_clf'] = syst.utils.clf_reweight(
                    df_mc.query('abs(probeScEta)>1.56'), df_data, n_jobs=10, cut=plot_dict[1]['cut'])
            else:
                warnings.warn(
                    'Cut from 0th plot used to reweight whole dataset. Make sure this makes sense')
                df_mc['weight_clf'] = syst.utils.clf_reweight(
                    df_mc, df_data, n_jobs=10, cut=plot_dict[0]['cut'])

    # if 'probePhiWidth' in varrs:
    #     df_data['probePhiWidth'] = df_data['probePhiWidth_Sc']

    # if 'probeEtaWidth' in varrs:
    #     df_data['probeEtaWidth'] = df_data['probeEtaWidth_Sc']

    # if 'probePhoIso03' in varrs:
    #     df_mc['probePhoIso03_uncorr'] = df_mc['probePhoIso_uncorr']

    for var in ['probeChIso03', 'probeSigmaIeIe']:
        df_data[var+'_corr'] = df_data[var]

    if options.recomp_mva:
        stride = int(df_mc.index.size/10)
        print(stride)
        correctedVariables = ['probeR9', 'probeS4', 'probeCovarianceIeIp', 'probeEtaWidth',
                              'probePhiWidth', 'probeSigmaIeIe', 'probePhoIso', 'probeChIso03', 'probeChIso03worst']
        weightsEB = "/work/threiten/QReg/ReReco17_data/camp_3_1_0/PhoIdMVAweights/HggPhoId_94X_barrel_BDT_v2.weights.xml"
        weightsEE = "/work/threiten/QReg/ReReco17_data/camp_3_1_0/PhoIdMVAweights/HggPhoId_94X_endcap_BDT_v2.weights.xml"
        df_mc['probeScPreshowerEnergy'] = np.zeros(df_mc.index.size)
        df_mc['probePhoIdMVA_uncorr'] = np.concatenate(Parallel(n_jobs=10, verbose=20)(delayed(helpComputeIdMva)(
            weightsEB, weightsEE, correctedVariables, df_mc[ch:ch+stride], 'uncorr', False) for ch in range(0, df_mc.index.size, stride)))
        # df_mc['probePhoIdMVA_uncorr'] = helpComputeIdMva(weightsEB,weightsEE,correctedVariables,df_mc,'uncorr', False)

    varrs_miss = check_vars(df_mc, varrs)
    varrs_data_miss = check_vars(df_data, varrs_data)
    if any(varrs_miss + varrs_data_miss):
        print('Missing variables from mc df: ', list(
            itertools.compress(varrs, varrs_miss)))
        print('Missing variables from data df: ', list(
            itertools.compress(varrs_data, varrs_data_miss)))
        raise KeyError('Variables missing !')

    plots = []
    for i, dic in enumerate(plot_dict):
        print('Initializing plot {}/{}'.format(i+1, len(plot_dict)), end='\r')
        plots.append(pldmc.plot_dmc_hist(
            df_mc, df_data=df_data, label=options.label, **dic))

    sys.stdout.flush()
    for i, plot in enumerate(plots):
        print('Drawing plot {}/{}'.format(i+1, len(plot_dict)), end='\r')
        plot.draw()
        plot.save(options.outdir, save_dill=options.save_dill)
        matplotlib.pyplot.close(plot.fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group()
    requiredArgs.add_argument(
        '-m', '--mc', action='store', type=str, required=True)
    requiredArgs.add_argument(
        '-d', '--data', action='store', type=str, required=True)
    requiredArgs.add_argument(
        '-c', '--config', action='store', type=str, required=True)
    requiredArgs.add_argument(
        '-o', '--outdir', action='store', type=str, required=True)
    optionalArgs = parser.add_argument_group()
    # optionalArgs.add_argument(
    #     '-r', '--ratio', action='store_true', default=False)
    # optionalArgs.add_argument(
    #     '-n', '--norm', action='store_true', default=False)
    optionalArgs.add_argument(
        '-p', '--save_dill', action='store_true', default=False)
    optionalArgs.add_argument('-w', '--no_reweight',
                              action='store_true', default=False)
    # optionalArgs.add_argument(
    #     '-k', '--cutstr', action='store_true', default=False)
    optionalArgs.add_argument('-M', '--recomp_mva',
                              action='store_true', default=False)
    optionalArgs.add_argument('-l', '--label', action='store', type=str)
    optionalArgs.add_argument('-N', '--n_evts', action='store', type=int)
    optionalArgs.add_argument('-t', '--mc_tree', action='store', type=str)
    optionalArgs.add_argument('-s', '--data_tree', action='store', type=str)
    optionalArgs.add_argument('-u', '--reweight_cut', action='store', type=str)
    options = parser.parse_args()
    main(options)

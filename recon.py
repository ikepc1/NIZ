import pandas as pd
import numpy as np

from utils import run_multiprocessing
from tyro_fit import tyro, TyroFit
from niche_plane import get_nfits_from_df, NichePlane, NicheFit
from fit import FitProcedure, AllTunka, E, BasicParams
from mcmc import main
from config import CounterConfig

def compile_nfit_inputs(df: pd.DataFrame) -> list[list[NicheFit]]:
    '''This function gets a list of lists of niche fits to pass to multiprocessing
    '''
    nfit_list = []
    indices = []
    for i in range(len(df)):
        if has_multiple_triggers(df,i):
            nfit_list.append(get_nfits_from_df(df, i))
            indices.append(i)
    return nfit_list, indices

def has_multiple_triggers(df: pd.DataFrame, index: int) -> bool:
    '''This function checks if more than one counter triggered
    for the event at index.
    '''
    return len(df.loc[index][12:][df.loc[index][12:].notna()]) > 1

def run_dataframe_recon(df: pd.DataFrame, cfg: CounterConfig) -> None:
    '''This function reconstructs each event in the dataframe.
    '''
    inputs, indices = compile_nfit_inputs(df)
    print('Adding Tyros...')
    tyros = np.array(run_multiprocessing(tyro, inputs),dtype='O')

    has_core = np.array([ty.has_contained_core for ty in tyros])

    print('Adding plane fits...')
    pfs = np.array(run_multiprocessing(NichePlane, inputs),dtype='O')

    has_enough_triggers = np.array([len(f.counters) > 4 for f in pfs])

    for index, tyr, pf in zip(indices, tyros, pfs):
        df.at[index,'Fit'] = tyr
        df.at[index,'Plane Fit'] = pf

    if not has_core.any():
        return None
    
    if not has_enough_triggers.any():
        return None

    cut = has_core & has_enough_triggers
    print('Adding minuit fits...')
    fp = FitProcedure(cfg)
    guesses = run_multiprocessing(fp.fit_procedure, list(zip(tyros[cut],pfs[cut])),chunksize=1)
    fit_dict = {str(name):[] for name in df.columns[:7]}

    for guess in guesses:
        for par in guess:
            if par.name in fit_dict:
                fit_dict[par.name].append(par.value)
        fit_dict['E'].append(E(guess[1].value))

    # print('Running mcmc...')
    # core_inputs = np.array(inputs, dtype='O')[has_core]
    # fit_dict = {str(name):[] for name in df.columns[:7]}
    # for i, (input,guess) in enumerate(zip(core_inputs, guesses)):
    #     print(f'event no: {i}')
    #     s = AllTunka(input,BasicParams,cfg)
    #     names = [p.name for p in guess if not p.fixed]
    #     ndim = len(names)
    #     sampler, pos, prob, state = main(s,guess,niter=250,nwalkers=2*ndim)
    #     best_sample = sampler.flatchain[np.argmax(sampler.flatlnprobability)]
    #     for name, val in zip(names,best_sample):
    #         if name in fit_dict:
    #             fit_dict[name].append(val)
    #     fit_dict['E'].append(E(best_sample[1]))
    
    fitdf = pd.DataFrame.from_dict(fit_dict)
    evdf = df.iloc[indices,0:7].iloc[cut]
    return fitdf, evdf, guesses




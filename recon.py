import pandas as pd
from multiprocessing import Pool, cpu_count
from alive_progress import alive_bar
from typing import Callable

from tyro_fit import tyro
from niche_plane import get_event_from_df, NichePlane, NicheFit

# def run_multiprocessing(func: Callable[[object],object], inputs: list[object]) -> list[object]:
#     '''This function maps a function to imap with the use of context managers
#     and a progress bar.
#     '''
#     results = []
#     with alive_bar(len(inputs)) as bar:
#         with Pool(cpu_count()) as p:
#             for result in p.imap(func, inputs, chunksize=250):
#                 results.append(result)
#                 bar()
#     return results

def run_multiprocessing(func: Callable[[object],object], inputs: list[object], chunksize = 250) -> list[object]:
    '''This function maps a function to imap with the use of context managers
    and a progress bar.
    '''
    results = []
    with alive_bar(len(inputs)) as bar:
        with Pool(cpu_count()) as p:
            for result in p.imap(func, inputs, chunksize=chunksize):
                results.append(result)
                bar()
    return results

def compile_nfit_inputs(df: pd.DataFrame) -> list[list[NicheFit]]:
    '''This function gets a list of lists of niche fits to pass to multiprocessing
    '''
    nfit_list = []
    indices = []
    for i in range(len(df)):
        if has_multiple_triggers(df,i):
            nfit_list.append(get_event_from_df(df, i))
            indices.append(i)
    return nfit_list, indices

def has_multiple_triggers(df: pd.DataFrame, index: int) -> bool:
    '''This function checks if more than one counter triggered
    for the event at index.
    '''
    return len(df.loc[index][12:][df.loc[index][12:].notna()]) > 1

def run_dataframe_recon(df: pd.DataFrame) -> None:
    '''This function reconstructs each event in the dataframe.
    '''
    inputs, indices = compile_nfit_inputs(df)
    print(len(indices))
    print(len(inputs))
    print('Adding Tyros...')
    tyros = run_multiprocessing(tyro, inputs)
    print('Adding plane fits...')
    pfs = run_multiprocessing(NichePlane, inputs)
    for index, tyr, pf in zip(indices, tyros, pfs):
        df.at[index,'Fit'] = tyr
        df.at[index,'Plane Fit'] = pf

    

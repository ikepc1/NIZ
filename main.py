import pandas as pd
import numpy as np
from dataclasses import dataclass
import sys

from utils import save_df
from throw_shower_params import gen_event_params
from gen_ckv_signals import get_showlib_ckv, LibraryEvent
from process_showers import ProcessCkv
from niche_plane import NichePlane
from showlib import draw_shower_from_library, LibraryShower
from fit import *
from config import *
from counter_config import init_config, CounterConfig

def library_shower_generator(mc_dict: dict):
    '''This generator yields a LibraryShower after updating the mc dict with 
    that shower's xmax and nmax.
    '''
    showids = np.random.randint(1,SHOWLIB_DRAWER_SIZE+1,len(mc_dict['E']))
    for i, iput in enumerate(zip(mc_dict['E'],showids)):
        s = draw_shower_from_library(iput)
        mc_dict['nmax'][i] = s.nmax
        mc_dict['xmax'][i] = s.xmax
        yield s

def main(data_date_and_time: str, n_thrown: int = N_THROWN, bad_dets: list[str] = []) -> pd.DataFrame:
    '''mizzain.
    '''
    #get data files and corresponding noise files for given night

    #set config object with counters active in data part
    cfg = init_config(data_date_and_time)

    #throw shower energies, cores, and, angles
    mc_dict = gen_event_params(cfg,MIN_LE,MAX_LE,SPECTRUM,n_thrown)

    #draw shower profiles from library
    print('Drawing showers from library...')
    # showids = np.random.randint(1,SHOWLIB_DRAWER_SIZE+1,n_thrown)
    # library_showers = [draw_shower_from_library(i) for i in list(zip(mc_dict['E'],showids))]
    # for i,s in enumerate(library_showers):
    #     mc_dict['nmax'][i] = s.nmax
    #     mc_dict['xmax'][i] = s.xmax
    library_showers = library_shower_generator(mc_dict)

    library_events = (LibraryEvent(s,t,p,x,y,z) for s,t,p,x,y,z in zip(library_showers,
                                                                       mc_dict['zenith'],
                                                                       mc_dict['azimuth'],
                                                                       mc_dict['corex'],
                                                                       mc_dict['corey'],
                                                                       mc_dict['corez']))
    
    #simulate Cherenkov light
    ckvs = (get_showlib_ckv((e,cfg)) for e in library_events)

    #simulate triggers
    print('Simulating Cherenkov light and detector response...')
    
    pc = ProcessCkv(cfg,frozen_noise=False)

    for i,ckv in enumerate(ckvs):
        #simulate triggers, very occasionally the signal on one is at the edge 
        #of the massive time domain for a trigger, if this happens, it wouldn't be usable anyways...
        try:
            nfits = pc.gen_nfits_from_ckv(ckv)
        except:
            continue

        #remove detectors on bad list from nightsky dataframe
        nfits = [f for f in nfits if f.name not in bad_dets]

        #need at least 3 counters in an event
        if len(nfits) < 3:
            continue

        #add nfits
        for nfit in nfits:
            mc_dict[nfit.name][i] = nfit

        #need plane fit to succeed
        try:
            p = NichePlane(nfits)
        except:
            continue
        mc_dict['Plane_Fit'][i] = p

        #add tyro fit
        t = tyro(nfits, cfg.gains)
        mc_dict['Fit'][i] = t

        #add guess
        mc_dict['guess'][i] = make_guess(t,p,cfg)

    #finally, create dataframe
    return pd.DataFrame.from_dict(mc_dict)

def get_average_weather(nightsky_df_path: Path) -> float:
    '''This function gets the average weather sum for a data part.
    '''
    nightsky_df = pd.read_pickle(nightsky_df_path)
    weat_codes = nightsky_df['weather']
    wsums = np.array([np.sum([int(i) for i in w]) for w in weat_codes])
    return wsums.mean()

def get_df_size(nightsky_df_path: Path) -> int:
    nightsky_df = pd.read_pickle(nightsky_df_path)
    return len(nightsky_df)

def get_bad_detnames(nightsky_df_path: Path) -> list[str]:
    '''This function finds the detectors whose triggers don't ever match with others.
    '''
    ns_df = pd.read_pickle(nightsky_df_path)
    bad_dets = []
    for c in ns_df.columns[15:]:
        if len(ns_df[c][ns_df[c].notna()]) == 0:
            bad_dets.append(c)
    return bad_dets

if __name__  == '__main__':
    ns_df_pkl = Path(sys.argv[1])
    n_events = get_df_size(ns_df_pkl)
    weat = get_average_weather(ns_df_pkl)
    bad_dets = get_bad_detnames(ns_df_pkl)

    target_filename = 'mc_' + ns_df_pkl.name
    target_path = MC_DF_PATH / target_filename
    if target_path.exists():
        pass
    elif weat > 0.:
        pass
    else:
        print(f'Processing {ns_df_pkl.name[:-4]}...')
        n = n_events * 80
        print(f'Throwing {n} events...')
        mc_df = main(ns_df_pkl.name[:-4], n, bad_dets)
        save_df(mc_df, target_filename, MC_DF_PATH)

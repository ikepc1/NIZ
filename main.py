import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
from dataclasses import dataclass

from datafiles import get_data_files, preceding_noise_file
from throw_shower_params import gen_params_in_bins, MCParams
from process_showers import add_triggers_to_dataframe
from tyro_fit import plot_event
from import_nightsky_data import init_niche_nightsky_df
from recon import run_dataframe_recon
from niche_plane import get_event_from_df
from config import CounterConfig, COUNTER_POSITIONS

def plot_generator(sim_datafram: pd.DataFrame) -> None:
    '''This is a generator which produces plots one at a time of each 
    trigger in a sim.
    '''
    for i, row in sim_datafram[sim_datafram['Fit'].notna()].iterrows():
        plot_event(row['Fit'])
        plt.suptitle(f"E = {row['E']:.1e} eV, zenith =  {np.rad2deg(row['zenith']):.2f}")
        plt.scatter(row['corex'], row['corey'], c='r', label='core')
        axis_x = -np.cos(row['zenith']) * np.cos(row['azimuth'])
        axis_y = -np.cos(row['zenith']) * np.sin(row['azimuth'])
        plt.quiver(row['corex'], row['corey'], axis_x, axis_y,color = 'k' ,label='thrown axis',scale_units='xy', scale=.01)
        plt.quiver(row['corex'], row['corey'], -row['Plane Fit'].nx, -row['Plane Fit'].ny, color='g', label = 'plane fit',scale_units='xy', scale=.01)
        plt.legend()
        plt.xlim(COUNTER_POSITIONS[:,0].min() - 100., COUNTER_POSITIONS[:,0].max() + 100.)
        plt.ylim(COUNTER_POSITIONS[:,1].min() - 100., COUNTER_POSITIONS[:,1].max() + 100.)
        yield

def process_energy_bin(params: MCParams, cfg: CounterConfig) -> pd.DataFrame:
    '''This function generates MC for one MCParams set of parameters.
    '''
    print(f'Processing mc for 10^{params.lEmin} eV < E < 10^{params.lEmax} eV...')
    return add_triggers_to_dataframe(params.gen_event_params(), cfg)

@dataclass
class MCRun:
    '''This is the container for the results of an mc run.
    '''
    cfg: CounterConfig
    mc: list[pd.DataFrame]
    ns: pd.DataFrame

def main(data_date_and_time: str) -> MCRun:
    '''mizzain.
    '''
    #get data files and corresponding noise files for given night
    data_files = get_data_files(data_date_and_time)
    noise_files = [preceding_noise_file(f) for f in data_files]

    #set config object with counters active in data part
    cfg = CounterConfig(data_files, noise_files)

    #get real data events and perform reconstruction
    print('Processing nightsky data...')
    ns_df = init_niche_nightsky_df(cfg)

    #recon real data
    run_dataframe_recon(ns_df)

    #Throw shower parameters
    shower_params = gen_params_in_bins(cfg)

    #Simulate showers for each energy bin
    shower_dataframes = []
    for params in shower_params:
        df = process_energy_bin(params, cfg)
        run_dataframe_recon(df)
        shower_dataframes.append(df)

    return MCRun(cfg, shower_dataframes, ns_df)

def event_ntriggers(df: pd.DataFrame) -> np.ndarray:
    lens = []
    for i in range(len(df)):
        lens.append(len(get_event_from_df(df,i)))
    return np.array(lens)

if __name__  == '__main__':
    date_time = sys.argv[1]
    np.seterr(all="ignore")

    mc = main(str(date_time))

    mc_ev_lens = event_ntriggers(mc.mc[0])
    ns_ev_lens = event_ntriggers(mc.ns)
    bins = .5 + np.arange(13)

    plt.ion()
    plt.figure()
    plt.hist(mc_ev_lens,label = 'mc events',density=True, histtype='step',bins=bins)
    plt.hist(ns_ev_lens,label = 'ns events',density=True, histtype='step',bins=bins)
    plt.legend()


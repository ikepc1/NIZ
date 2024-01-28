from typing import Callable
from multiprocessing import Pool, cpu_count
from alive_progress import alive_bar
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

from config import COUNTER_POSITIONS, RAW_DATA_PATH
from pathlib import Path
from niche_bin import bin_to_raw, NicheRaw
from tyro_fit import TyroFit

def isolate_digits(string: str) -> str:
    '''This function removes non integer characters from a string.
    '''
    return ''.join(c for c in string if c.isdigit())

def datetime_from_filename(file: Path) -> datetime:
    '''This function creates a datetime object from the date and time in the 
    filename of a niche datafile.
    '''
    return datetime.strptime(isolate_digits(file.name), '%Y%m%d%H%M%S')

def preceding_noise_file(datafile: Path) -> Path:
    '''This function finds the noise file immediately preceding the input data
    file.
    '''
    data_time = datetime_from_filename(datafile)
    noise_files = [file for file in datafile.parent.glob('*.bg.bin')]
    noise_times = np.array([datetime_from_filename(file) for file in noise_files]).astype(datetime)
    deltas = data_time - noise_times
    return noise_files[np.abs(deltas).argmin()]

def get_file_ts(file: Path) -> np.datetime64:
    '''This function gets the timestamp from the filename.
    '''
    y = file.name[:4]
    m = file.name[4:6]
    d = file.name[6:8]
    H = file.name[8:10]
    M = file.name[10:12]
    S = file.name[12:14]
    return np.datetime64(y+'-'+m+'-'+d+' '+H+':'+M+':'+S)

def date2bytes() -> bytearray:
    '''This function returns a bytestring of the current date.
    '''
    now = datetime.now()
    return bytearray(now.strftime("%Y%m%d%H%M%S"), 'utf-8')

def get_data_files(timestr: str) -> list[str]:
    '''This function finds the corresponding data parts for each
    counter at a given time. The second wildcard in the glob makes
    this work for either noise or nightsky files.
    '''
    data_directory_path = Path(RAW_DATA_PATH) / timestr[:8]
    return sorted(data_directory_path.glob(f'*/{timestr}*.bin'))

def read_niche_file(filepath: Path) -> list[NicheRaw]:
    '''This function reads a noise file and returns a numoy array of the traces.
    '''
    with filepath.open('rb') as open_file:
        nraw_list = list(set(bin_to_raw(open_file.read(), filepath.parent.name, retfit=False)))
    return nraw_list

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

def plot_detectors() -> None:
    '''This function adds the NICHE counters to a plot.
    '''
    plt.scatter(COUNTER_POSITIONS[:,0], COUNTER_POSITIONS[:,1], c = 'b',label = 'detectors')
    ax = plt.gca()
    ax.set_aspect('equal')

def init_niche_plot() -> None:
    plt.ion()
    plt.figure()
    plot_detectors()
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Niche Event')
    plt.grid()

def plot_triggers(fit: TyroFit) -> None:
    '''This function plots the detectors that triggered where the size of 
    each scatter point is the pulse area and the color is the time.
    '''
    plt.scatter(fit.counter_pos[:,0], fit.counter_pos[:,1], s = fit.pa, c=fit.t)#, cmap = 'inferno')
    plt.colorbar(label='nanoseconds')

def plot_event(fit: TyroFit, title: str='') -> None:
    '''This function plots an individual Niche event.
    '''
    init_niche_plot()
    plot_triggers(fit)
    plt.suptitle(title)

def plot_generator(event_dataframe: pd.DataFrame) -> None:
    '''This is a generator which produces plots one at a time of each 
    trigger in an event dataframe.
    '''
    for i, row in event_dataframe[event_dataframe['Fit'].notna()].iterrows():
        init_niche_plot()
        plot_triggers(row['Fit'])
        plt.suptitle(f"Event index = {i}, E = {row['E']:.1e} eV, zenith =  {np.rad2deg(row['Plane Fit'].theta):.2f}")
        # plt.suptitle(f"Event index = {i}, E = {row['E']:.1e} eV, zenith =  {np.rad2deg(row['zenith']):.2f}")
        plt.scatter(row['corex'], row['corey'], c='r', label='core')
        axis_x = -np.cos(row['zenith']) * np.cos(row['azimuth'])
        axis_y = -np.cos(row['zenith']) * np.sin(row['azimuth'])
        plt.quiver(row['corex'], row['corey'], axis_x, axis_y,color = 'k' ,label='thrown axis',scale_units='xy', scale=.01)
        plt.quiver(row['corex'], row['corey'], -row['Plane Fit'].nx, -row['Plane Fit'].ny, color='g', label = 'plane fit',scale_units='xy', scale=.01)
        plt.legend()
        plt.xlim(COUNTER_POSITIONS[:,0].min() - 100., COUNTER_POSITIONS[:,0].max() + 100.)
        plt.ylim(COUNTER_POSITIONS[:,1].min() - 100., COUNTER_POSITIONS[:,1].max() + 100.)
        yield
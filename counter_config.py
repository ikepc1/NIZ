import numpy as np
from pathlib import Path
from dataclasses import dataclass

from config import COUNTER_POSITIONS_DICT, COUNTER_QE, COUNTER_PMT_DELAY, COUNTER_FADC_PER_PHOTON, NSBG
from utils import read_niche_file, read_noise_file, get_data_files, preceding_noise_files
from noise import noise_fft, freq_mhz
from niche_raw import NicheRaw

@dataclass
class CounterConfig:
    '''This is the container for all the constants related to particular counters for
    a particular mc run.
    '''
    data_files: list[Path]
    noise_closed_files: list[Path]
    noise_open_files: list[Path]
    max_photon_zenith: float = np.deg2rad(45)

    # def __post_init__(self) -> None:

    #     # self.active_counters = [f.parent.name for f in self.data_files]
    #     # self.positions = self.dict_comp(COUNTER_POSITIONS_DICT)
    #     # self.positions_array = np.array(list(self.positions.values()))
    #     self.quantum_efficiency = self.dict_comp(COUNTER_QE)
    #     self.pmt_gain = self.dict_comp(COUNTER_PMT_DELAY)
    #     self.pmt_delay = self.dict_comp(COUNTER_PMT_DELAY)
    #     self.fadc_per_pe = self.dict_comp(COUNTER_FADC_PER_PE)
    #     self.radii = np.full((self.positions_array.shape[0]), .0508)

    def dict_comp(self, og_dict: dict) -> dict:
        '''This method wraps a dictionary comprehension on the keys.
        '''
        return {name:og_dict[name] for name in self.active_counters}
    
    @property
    def active_counters(self) -> list[str]:
        return [f.parent.name for f in self.data_files]
    
    @property
    def positions(self) -> dict:
        return self.dict_comp(COUNTER_POSITIONS_DICT)
    
    @property
    def positions_array(self) -> np.ndarray:
        return np.array(list(self.positions.values()))
    
    @property
    def quantum_efficiency(self) -> dict:
        return self.dict_comp(COUNTER_QE)
    
    @property
    def pmt_gain(self) -> dict:
        return self.dict_comp()

    @property
    def counter_center(self) -> np.ndarray:
        '''This is the centroid of the active counter's positions.
        '''
        avgx = self.positions_array[:,0].mean()
        avgy = self.positions_array[:,1].mean()
        avgz = self.positions_array[:,2].mean()
        return np.array([avgx, avgy, avgz])
    
    @property
    def counter_bottom(self) -> np.ndarray:
        '''This is the centroid of the active counter's positions.
        '''
        avgx = self.positions_array[:,0].mean()
        avgy = self.positions_array[:,1].mean()
        avgz = self.positions_array[:,2].min()
        return np.array([avgx, avgy, avgz])
    
    def avg_temp(self) -> dict[str,float]:
        '''This method calculates the average temperature for each counter
        during the data part.
        '''
        tempdict = {}
        for file in self.data_files:
            tempdict[file.parent.name] = np.mean([nraw.temp for nraw in read_niche_file(file)])
        return tempdict
    
    # def noise_level(self) -> dict[str,float]:
    #     '''This function returns the average value of the noise open minus 
    #     noise closed power spectra for each counter.
    #     '''
    #     noisedict = {}
    #     for open, closed in zip(self.noise_open_files,self.noise_closed_files):
    #         open_noise = noise_fft(read_noise_file(open))
    #         closed_noise = noise_fft(read_noise_file(closed))
    #         noisedict[open.parent.name] = np.mean(open_noise - closed_noise)
    #     return noisedict

def noise_level(cfg: CounterConfig) -> dict[str,float]:
    '''This function returns the average value of the noise open minus 
    noise closed power spectra for each counter.
    '''
    freq = freq_mhz()
    noisedict = {}
    for open, closed in zip(cfg.noise_open_files,cfg.noise_closed_files):
        try:
            open_traces = read_noise_file(open)
            closed_traces = read_noise_file(closed)
        except:
            # print('empty noise:')
            # print(open.parent.name)
            # print(open.name)
            # print(closed.name)
            noisedict[open.parent.name] = np.nan
            continue
        open_noise = noise_fft(open_traces)
        closed_noise = noise_fft(closed_traces)
        max = np.max((open_noise**2 - closed_noise**2)[freq>0.])
        mean = np.mean(open_noise[freq>0.] - closed_noise[freq>0.])
        #if the difference is small or negative, either the shutter didn't open, or the files were messed up.
        if mean < 1.:
            noisedict[open.parent.name] = np.nan
            # print('negative noise level:')
            # print(open.parent.name)
            # print(open.name)
            # print(closed.name)
        else:
            noisedict[open.parent.name] = max
    return noisedict

def estimate_nsbg(cfg: CounterConfig) -> dict[str,float]:
    ''''''
    noisedict = noise_level(cfg)
    return {c:noisedict[c]/COUNTER_FADC_PER_PHOTON[c]**2 for c in noisedict if c in COUNTER_FADC_PER_PHOTON}

    
def estimate_gain(cfg: CounterConfig) -> dict[str,float]:
    '''This function estimates the number of FADC counts per photon for 
    each counter.
    '''
    noisedict = noise_level(cfg)
    gaindict = {c:np.sqrt(noisedict[c]/NSBG) for c in noisedict}
    return gaindict

def avg_temp(cfg: CounterConfig) -> dict[str,float]:
    '''This method calculates the average temperature for each counter
    during the data part.
    '''
    tempdict = {}
    for file in cfg.data_files:
        tempdict[file.parent.name] = np.mean([nraw.temp for nraw in read_niche_file(file)])
    return tempdict

def raw_data(cfg: CounterConfig) -> dict[str,list[NicheRaw]]:
    '''This method calculates the average temperature for each counter
    during the data part.
    '''
    tempdict = {}
    for file in cfg.data_files:
        tempdict[file.parent.name] = [nraw for nraw in read_niche_file(file)]
    return tempdict

def pt(cfg: CounterConfig) -> dict[str,list[NicheRaw]]:
    '''This method calculates the average temperature for each counter
    during the data part.
    '''
    tempdict = {}
    for file in cfg.data_files:
        nraws = read_niche_file(file)
        avg_temp = np.mean([nraw.temp for nraw in nraws])
        avg_press = np.mean([nraw.press for nraw in nraws])
        tempdict[file.parent.name] = (avg_press,avg_temp)
    return tempdict

def init_config(ts: str) -> CounterConfig:
    '''This function generates a config object for a data part with a given timestamp.
    '''
    alldata = get_data_files(ts)
    allnsdata = [file for file in alldata if (file.name.endswith('.bin') and file.name[-5].isnumeric())]
    ns_data_part = [file for file in allnsdata if file.name[:-5] == ts[:-1]]
    noise_files = [preceding_noise_files(file) for file in ns_data_part]
    closed_noise_files = [tup[0] for tup in noise_files]
    open_noise_files = [tup[1] for tup in noise_files]
    cfg = CounterConfig(ns_data_part, closed_noise_files, open_noise_files)
    return cfg

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from pathlib import Path
    from datetime import datetime

    from noise import *
    from utils import datetime_from_filename, get_data_files, run_multiprocessing
    counters = ['bardeen','bell','curie','feynman','newton','noether','rossi','rubin','rutherford','wu','einstein','yukawa','dirac','meitner']
    niche_data_dir = Path('/home/isaac/niche_data/')
    cfgs = []
    times = []
    for datepath in niche_data_dir.iterdir():
        alldata = get_data_files(datepath.name)
        datatimes = list(set([f.name[:-4] for f in alldata if f.name.endswith('.bin') and f.name[-5].isnumeric()]))
        cfgs.extend([init_config(t) for t in datatimes])
        times.extend([datetime.strptime(t, '%Y%m%d%H%M%S') for t in datatimes])

    noise_levels = run_multiprocessing(noise_level,cfgs,chunksize=1)
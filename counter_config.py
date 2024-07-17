import numpy as np
from pathlib import Path
from dataclasses import dataclass

from config import COUNTER_POSITIONS_DICT, COUNTER_QE, COUNTER_PMT_DELAY, COUNTER_FADC_PER_PE
from utils import read_niche_file, read_noise_file, get_data_files, preceding_noise_files
from noise import noise_fft

@dataclass
class CounterConfig:
    '''This is the container for all the constants related to particular counters for
    a particular mc run.
    '''
    data_files: list[Path]
    noise_closed_files: list[Path]
    noise_open_files: list[Path]
    max_photon_zenith: float = np.deg2rad(45)

    def __post_init__(self) -> None:
        self.active_counters = [f.parent.name for f in self.data_files]
        self.positions = self.dict_comp(COUNTER_POSITIONS_DICT)
        self.positions_array = np.array(list(self.positions.values()))
        self.quantum_efficiency = self.dict_comp(COUNTER_QE)
        self.pmt_gain = self.dict_comp(COUNTER_PMT_DELAY)
        self.pmt_delay = self.dict_comp(COUNTER_PMT_DELAY)
        self.fadc_per_pe = self.dict_comp(COUNTER_FADC_PER_PE)
        self.radii = np.full((self.positions_array.shape[0]), .0508)

    def dict_comp(self, og_dict: dict) -> dict:
        '''This method wraps a dictionary comprehension on the keys.
        '''
        return {name:og_dict[name] for name in self.active_counters}
    
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
    
    def noise_level(self) -> dict[str,float]:
        '''This function returns the average value of the noise open minus 
        noise closed power spectra for each counter.
        '''
        noisedict = {}
        for open, closed in zip(self.noise_open_files,self.noise_closed_files):
            open_noise = noise_fft(read_noise_file(open))
            closed_noise = noise_fft(read_noise_file(closed))
            noisedict[open.parent.name] = np.mean(open_noise - closed_noise)
        return noisedict

def init_config(ts: str) -> CounterConfig:
    '''This function generates a config object for a data part with a given timestamp.
    '''
    alldata = get_data_files(ts)
    allnsdata = [file for file in alldata if (file.name.endswith('.bin') and file.name[-5].isnumeric())]
    ns_data_part = [file for file in allnsdata if file.name[:-4] == ts]
    noise_files = [preceding_noise_files(file) for file in ns_data_part]
    closed_noise_files = [tup[0] for tup in noise_files]
    open_noise_files = [tup[1] for tup in noise_files]
    cfg = CounterConfig(ns_data_part, closed_noise_files, open_noise_files)
    return cfg

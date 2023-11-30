from dataclasses import dataclass, field, fields
import numpy as np
from pathlib import Path

'''constants'''
#paths
RAW_DATA_PATH = '/home/isaac/niche_data/'

#niche array params
CXF_ALTITUDE = 1500.
WAVEFORM_SIZE = 1024 # number
NICHE_TIMEBIN_SIZE = 5. # ns
N_SIM_TRIGGER_WINDOWS = 10
PHOTONS_WINDOW_SIZE = round(N_SIM_TRIGGER_WINDOWS * WAVEFORM_SIZE * NICHE_TIMEBIN_SIZE) #number of photon signal bins
PHOTONS_TIMEBIN_SIZE = 1. #in ns
TRIGGER_VARIANCE = 49.
TRIGGER_WIDTH = 8

#MC Parameters
N_ENERGY_BINS = 1
MIN_LE = 14.
MAX_LE = 18.
E_BIN_EDGES = np.linspace(MIN_LE,MAX_LE,N_ENERGY_BINS+1)
E_BINS = E_BIN_EDGES[:-1] + np.diff(E_BIN_EDGES)/2
N_THROWN = 8
THROW_RADIUS = 300. #meters per 10^12

#CHASM inputs
MIN_WAVELENGTH = 300.
MAX_WAVELENGTH = 450.
N_WAVELENGTH_BINS = 1
CHASM_MESH = True


'''counter properties'''
ACTIVE_COUNTERS = {'curie':     True,
                   'dirac':     True,
                   'einstein':  True,
                   'feynman':   True,
                   'meitner':   True,
                   'newton':    True,
                   'noether':   True,
                   'rutherford':True,
                   'wu':        True,
                   'yukawa':    True,
                   'bardeen':   True,
                   'bell':      True,
                   'rossi':     True,
                   'rubin':     True}

NAMES = {0:'curie',
         1:'dirac',
         2:'einstein',
         3:'feynman',
         4:'meitner',
         5:'newton',
         6:'noether',
         7:'rutherford',
         8:'wu',
         9:'yukawa',
         10: 'bardeen',
         11: 'bell',
         12: 'rossi',
         13: 'rubin'}

COUNTER_NO = {'curie':     0,
              'dirac':     1,
              'einstein':  2,
              'feynman':   3,
              'meitner':   4,
              'newton':    5,
              'noether':   6,
              'rutherford':7,
              'wu':        8,
              'yukawa':    9,
              'bardeen':  10,
              'bell':     11,
              'rossi':    12,
              'rubin':    13}

COUNTER_POSITIONS_DICT = { 'curie':(392.8,-711.4,-24),
                      'dirac':(574.2,-607.3,-26),
                      'einstein':(489.2,-514.0,-23),
                      'feynman':(577.4,-720.0,-27),
                      'meitner':(489.6,-821.0,-27),
                      'newton':(379.5,-619.1,-26),
                      'noether':(389.1,-508.5,-25),
                      'rutherford':(489.2,-615.1,-29),
                      'wu':(290.4,-508.3,-26),
                      'yukawa':(483.0,-709.7,-25),
                      'bardeen':(283.3,-708.1,-24),
                      'bell':(592.0,-823.6,-23),
                      'rossi':(286.0,-610.4,-21),
                      'rubin':(397.3,-808.0,-26)}
COUNTER_POSITIONS = np.array(list(COUNTER_POSITIONS_DICT.values()))

COUNTER_QE = {'curie':     .75, #photocathode quantum efficiency
              'dirac':     .75,
              'einstein':  .75,
              'feynman':   .75,
              'meitner':   .75,
              'newton':    .75,
              'noether':   .75,
              'rutherford':.75,
              'wu':        .75,
              'yukawa':    .75,
              'bardeen':   .75,
              'bell':      .75,
              'rossi':     .75,
              'rubin':     .75}

COUNTER_PMT_GAIN = {'curie':     1.e6,
                    'dirac':     1.e6,
                    'einstein':  1.e6,
                    'feynman':   1.e6,
                    'meitner':   1.e6,
                    'newton':    1.e6,
                    'noether':   1.e6,
                    'rutherford':1.e6,
                    'wu':        1.e6,
                    'yukawa':    1.e6,
                    'bardeen':   1.e6,
                    'bell':      1.e6,
                    'rossi':     1.e6,
                    'rubin':     1.e6}

COUNTER_PMT_DELAY = {'curie':    40., # in ns
                    'dirac':     40.,
                    'einstein':  40.,
                    'feynman':   40.,
                    'meitner':   40.,
                    'newton':    40.,
                    'noether':   40.,
                    'rutherford':40.,
                    'wu':        40.,
                    'yukawa':    40.,
                    'bardeen':   40.,
                    'bell':      40.,
                    'rossi':     40.,
                    'rubin':     40.}

# COUNTER_FADC_PER_PE =  {'curie':     1., # number of fadc counts per PE
#                         'dirac':     1.,
#                         'einstein':  1.,
#                         'feynman':   1.,
#                         'meitner':   1.,
#                         'newton':    1.,
#                         'noether':   1.,
#                         'rutherford':1.,
#                         'wu':        1.,
#                         'yukawa':    1.,
#                         'bardeen':   1.,
#                         'bell':      1.,
#                         'rossi':     1.,
#                         'rubin':     1.}

COUNTER_FADC_PER_PE =  {'curie':     10., # number of fadc counts per PE
                        'dirac':     10.,
                        'einstein':  10.,
                        'feynman':   10.,
                        'meitner':   10.,
                        'newton':    10.,
                        'noether':   10.,
                        'rutherford':10.,
                        'wu':        10.,
                        'yukawa':    10.,
                        'bardeen':   10.,
                        'bell':      10.,
                        'rossi':     10.,
                        'rubin':     10.}
@dataclass
class CounterConfig:
    '''This is the container for all the constants related to particular counters for
    a particular mc run.
    '''
    data_files: list[Path]
    noise_files: list[Path]

    def __post_init__(self) -> None:
        self.active_counters = [f.parent.name for f in self.data_files]
        self.positions = self.dict_comp(COUNTER_POSITIONS_DICT)
        self.positions_array = np.array(list(self.positions.values()))
        self.quantum_efficiency = self.dict_comp(COUNTER_QE)
        self.pmt_gain = self.dict_comp(COUNTER_PMT_DELAY)
        self.pmt_delay = self.dict_comp(COUNTER_PMT_DELAY)
        self.fadc_per_pe = self.dict_comp(COUNTER_FADC_PER_PE)
        self.radii = np.full((self.positions_array.shape[0]), .1)

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




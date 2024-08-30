from dataclasses import dataclass, field, fields
import numpy as np
from pathlib import Path

from atmosphere import CorsikaAtmosphere, Atmosphere

'''constants'''
#paths
RAW_DATA_PATH = Path('/home/isaac/niche_data/')
NIZ_DIRECTORY = Path('/home/isaac/NIZ/')
NIGHTSKY_DF_PATH = NIZ_DIRECTORY / 'nightsky_dfs/'
MC_DF_PATH = NIZ_DIRECTORY / 'mc_dfs/'

#niche array params
CXF_ALTITUDE = 1500.
WAVEFORM_SIZE = 1024 # number
NICHE_TIMEBIN_SIZE = 5. # ns
N_SIM_TRIGGER_WINDOWS = 10
PHOTONS_WINDOW_SIZE = round(N_SIM_TRIGGER_WINDOWS * WAVEFORM_SIZE * NICHE_TIMEBIN_SIZE) #number of photon signal bins
PHOTONS_TIMEBIN_SIZE = 1. #in ns
TRIGGER_VARIANCE = 49.
TRIGGER_WIDTH = 8
TRIGGER_POSITION = 524
TEL_RADII = .0508

#MC Parameters
N_ENERGY_BINS = 1
MIN_LE = 14.
MAX_LE = 16.
E_BIN_EDGES = np.linspace(MIN_LE,MAX_LE,N_ENERGY_BINS+1)
E_BINS = E_BIN_EDGES[:-1] + np.diff(E_BIN_EDGES)/2
N_THROWN = 10
THROW_RADIUS = 500. #meters per 10^12
SHOWLIB_DRAWER_SIZE = 512

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

COUNTER_FADC_PER_PE =  {'curie':     1., # number of fadc counts per PE
                        'dirac':     1.,
                        'einstein':  1.,
                        'feynman':   1.,
                        'meitner':   1.,
                        'newton':    1.,
                        'noether':   1.,
                        'rutherford':1.,
                        'wu':        1.,
                        'yukawa':    1.,
                        'bardeen':   1.,
                        'bell':      1.,
                        'rossi':     1.,
                        'rubin':     1.}

'''Both of the following are from the calibration night.
'''
# COUNTER_FADC_PER_PHOTON = {'bardeen': 2.188203688465158,
#                            'bell': 7.406880637204029,
#                            'curie': 9.561962861718685,
#                            'feynman': 7.734872492963434,
#                            'newton': 11.281268225032628,
#                            'noether': 9.834954480425155,
#                            'rossi': 5.1908236939723436,
#                            'rubin': 5.783684658654795}

COUNTER_FADC_PER_PHOTON = {'bardeen': 0.4301294450523357,
 'bell': 1.4639139998316604,
 'curie': 1.8795107279931322,
 'feynman': 1.5014120362528374,
 'newton': 2.182147680064487,
 'noether': 1.8765021501429047,
 'rossi': 1.0044919895844284,
 'rubin': 1.131329751694293}

COUNTER_NOISE_LEVEL = {'bardeen': 4.44597979813936,
                       'bell': 36.87722110836859,
                       'curie': 45.91072757590564,
                       'feynman': 41.874449050198805,
                       'newton': 45.9294395593107,
                       'noether': 39.88911558052739,
                       'rossi': 22.573392919805574,
                       'rubin': 14.682870253902644,
                       }
# NSBG = 242.9651726980374
NSBG = 7965.114197731613

# COUNTER_FADC_PER_PE =  {'curie':     2., # number of fadc counts per PE
#                         'dirac':     2.,
#                         'einstein':  2.,
#                         'feynman':   2.,
#                         'meitner':   2.,
#                         'newton':    2.,
#                         'noether':   2.,
#                         'rutherford':2.,
#                         'wu':        2.,
#                         'yukawa':    2.,
#                         'bardeen':   2.,
#                         'bell':      2.,
#                         'rossi':     2.,
#                         'rubin':     2.}

def photon_time_bins() -> np.ndarray:
    '''This method returns the full array of photon time bins of size
    PHOTONS_TIMEBIN_SIZE used for the signal in each counter
    '''
    width = (PHOTONS_TIMEBIN_SIZE * PHOTONS_WINDOW_SIZE) / 2
    return np.arange(-width, width + PHOTONS_TIMEBIN_SIZE , PHOTONS_TIMEBIN_SIZE)

@dataclass
class AxisConfig:
    '''This is the container for axis config parameters.
    '''
    N_POINTS: int = 500
    N_IN_RING: int = 11
    MIN_CHARGED_PARTICLES: float = 7.e4 #number of charged particles for a step to be considered in cherenkov calcs
    ATM: Atmosphere = CorsikaAtmosphere()
    # ATM: Atmosphere = USStandardAtmosphere()
    MAX_RING_SIZE: float = 300


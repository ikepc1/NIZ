from dataclasses import dataclass
import polars as pl
import pandas as pd
from scipy.stats import norm
import numpy as np
from write_niche import CounterTrigger
from config import CounterConfig, WAVEFORM_SIZE, MIN_LE, THROW_RADIUS, E_BIN_EDGES, N_THROWN

def gen_zeniths(N_events: int = 100) -> np.ndarray:
    '''This function draws random zenith angles from the pdf 
    2sin(theta)cos(theta).
    '''
    return np.arcsin(np.sqrt(np.random.uniform(size = N_events)))
    # return np.full(N_events,np.deg2rad(60))

def gen_azimuths(N_events: int = 100) -> np.ndarray:
    '''This function generates uniformly distributed shower azimuthal
    angles.
    '''
    return np.random.uniform(size = N_events) * 2 * np.pi

def gen_core_pos(r: float, cfg: CounterConfig, N_events: int = 100) -> tuple[np.ndarray]:
    '''This function generates the core positions of events in a 
    circle of radius r.
    '''
    rs = r * np.sqrt(np.random.uniform(size = N_events))
    phis = gen_azimuths(N_events) #random phis
    zs = np.zeros_like(phis)
    return np.vstack((rs * np.cos(phis), rs * np.sin(phis), zs)).T + cfg.counter_center

def Nmax(E: float | np.ndarray) -> float:
    '''This function calculates a shower's Nmax based on its energy.
    '''
    return E * 1.3e-9

def draw_Xmax(lE: float) -> float:
    '''This function draws a random Xmax based on the energy.
    '''
    meanXmax = 600. + (lE - 15.7) * 33.
    return norm.rvs(loc = meanXmax, scale = 80.37, size = 1)

def X_cdf(lE: float, Xs: np.ndarray) -> float | np.ndarray:
    '''This function is the cdf values of the xmax cdf corresponding to lE at 
    grammage values Xs.
    '''
    meanXmax = 600. + (lE - 15.7) * 33.
    return norm.cdf(Xs, loc = meanXmax, scale = 80.37)

class DrawXmax:
    '''This class draws Xmax values from my pre-compiled table
    of Xmax cdfs.
    '''
    table_file = 'xcdf_lE.npz'

    def __init__(self) -> None:
        table = np.load(self.table_file)
        self.lEs = table['lEs']
        self.Xs = table['Xs']
        self.cdfs = table['xcdfs']

    def gen_Xmax(self, lE: float) -> float:
        '''This method draws a random xmax from the cdf most closely corresponding to log10(E/eV)
        specified by lE.
        '''
        rvs = np.random.uniform(size = 1)
        cdf = self.cdfs[np.abs(lE-self.lEs).argmin()]
        return np.interp(rvs,cdf,self.Xs)

def Xmax_of_lE(lEs: np.ndarray) -> np.ndarray:
    '''This function returns a list of drawn Xmax values corresponding to the input
    energies.    # return np.arcsin(np.random.uniform(size = N_events))

    '''
    xgen = DrawXmax()
    xmaxs = np.empty_like(lEs)
    for i, lE in enumerate(lEs):
        xmaxs[i] = xgen.gen_Xmax(lE)
    return xmaxs

def placeholder_triggers(n_events: int, cfg: CounterConfig) -> np.ndarray[CounterTrigger]:
    '''This function generates an array of null CounterTrigger objects so
    full size df can be generated.
    '''
    arr = np.empty((len(cfg.active_counters), n_events), dtype='o')
    for i, counter_name in enumerate(cfg.active_counters):
        trig = CounterTrigger(counter_name, np.zeros(WAVEFORM_SIZE), np.zeros(WAVEFORM_SIZE))
        arr[i] = np.full(n_events, trig, dtype='o')
    return arr

@dataclass
class MCParams:
    '''This is the container for the parameters used in an MC run.
    '''

    cfg: CounterConfig
    lEmin: float = 14.
    lEmax: float = 17.
    gamma: float = -2.
    N_events: int = 1000
    # radius: float = 5000.

    def __post_init__(self) -> None:
        # self.radius = (self.lEmin / MIN_LE)**3 * THROW_RADIUS
        self.radius = THROW_RADIUS
        self.Emin = 10**self.lEmin
        self.Emax = 10**self.lEmax

    def draw_Es(self) -> np.ndarray:
        '''This method draws random energies within the interval 
        specified from a spectrum specified by gamma.
        '''
        cdf = np.random.uniform(size = self.N_events)
        p = 1 + self.gamma
        return (cdf*(self.Emax**p - self.Emin**p) + self.Emin**p)**(1/p)

    def gen_event_params(self) -> pd.DataFrame:
        '''This method returns a dataframe with the parameters for each 
        event, with empty columns for the NicheRaw objects for each counter 
        and the fits.
        '''
        Es = self.draw_Es()
        cores = gen_core_pos(self.radius, self.cfg, N_events=self.N_events)
        params_dict = {
                'E': Es,
                'xmax': Xmax_of_lE(np.log10(Es)),     
                'nmax': Nmax(Es),
                'zenith': gen_zeniths(self.N_events),
                'azimuth': gen_azimuths(self.N_events),
                # 'corex': cores[:,0],
                # 'corey': cores[:,1],
                'corex': np.full_like(cores[:,0],437.),
                'corey': np.full_like(cores[:,0],-660.),
                'corez': cores[:,2],
                'X0': np.zeros_like(cores[:,0]),
                'Lambda': np.full_like(cores[:,0], 70.)
            }
        params_dict['Fit'] = np.full(self.N_events,np.NAN)
        params_dict['Plane Fit'] = np.full(self.N_events,np.NAN)
        params_dict['guess'] = np.empty(self.N_events, dtype='O')
        for counter_name in self.cfg.active_counters:
            params_dict[counter_name] = np.full(self.N_events,np.NAN)
        return pd.DataFrame.from_dict(params_dict)


def gen_params_in_bins(cfg: CounterConfig) -> list[MCParams]:
    '''This function creates a list of shower parameter Dataframes for each energy bin.
    '''
    param_list = []
    for low, high in zip(E_BIN_EDGES[:-1], E_BIN_EDGES[1:]):
        param_list.append(MCParams(cfg, N_events=N_THROWN, lEmin=low, lEmax=high))
    return param_list

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.ion()
    mc = MCParams(N_events=10000).gen_event_params()

    plt.figure()
    plt.hist(mc['Xmax'],histtype='step', bins = 100)
    plt.xlabel('Xmax')

    plt.figure()
    plt.scatter(np.log10(mc['E']),mc['Xmax'],s=.01)
    plt.semilogy()
    plt.xlabel('log10(E/eV)')
    plt.ylabel('Xmax')

    plt.figure()
    plt.scatter(mc['corex'],mc['corey'],s=.01)
    plt.xlabel('core x (m)')
    plt.ylabel('core y (m)')
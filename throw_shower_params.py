from dataclasses import dataclass
import polars as pl
import pandas as pd
from scipy.stats import norm
import numpy as np
from datetime import datetime

from write_niche import CounterTrigger
from config import WAVEFORM_SIZE, MIN_LE, THROW_RADIUS, E_BIN_EDGES, N_THROWN
from counter_config import CounterConfig
from gen_ckv_signals import Event

def gen_zeniths(N_events: int = 100) -> np.ndarray:
    '''This function draws random zenith angles from the pdf 
    2sin(theta)cos(theta).
    '''
    return np.arcsin(np.sqrt(.5*np.random.uniform(size = N_events)))
    # return np.full(N_events,np.deg2rad(30))

def gen_azimuths(N_events: int = 100) -> np.ndarray:
    '''This function generates uniformly distributed shower azimuthal
    angles.
    '''
    return np.random.uniform(size = N_events) * 2 * np.pi

def gen_core_pos(r: float, cfg: CounterConfig, N_events: int = 100) -> np.ndarray:
    '''This function generates the core positions of events in a 
    circle of radius r.
    '''
    rs = r * np.sqrt(np.random.uniform(size = N_events))
    phis = gen_azimuths(N_events) #random phis
    zs = np.zeros_like(phis)
    return np.vstack((rs * np.cos(phis), rs * np.sin(phis), zs)).T + cfg.counter_bottom

def Nmax(E: float | np.ndarray) -> float | np.ndarray:
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

def placeholder_triggers(n_events: int, cfg: CounterConfig) -> np.ndarray:
    '''This function generates an array of null CounterTrigger objects so
    full size df can be generated.
    '''
    arr = np.empty((len(cfg.active_counters), n_events), dtype='o')
    for i, counter_name in enumerate(cfg.active_counters):
        trig = CounterTrigger(counter_name, np.zeros(WAVEFORM_SIZE), np.zeros(WAVEFORM_SIZE), datetime.now())
        arr[i] = np.full(n_events, trig, dtype='o')
    return arr

@dataclass
class MCParams:
    '''This is the container for the parameters used in an MC run.
    '''

    # cfg: CounterConfig
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

    def gen_event_params(self, cfg: CounterConfig) -> pd.DataFrame:
        '''This method returns a dataframe with the parameters for each 
        event, with empty columns for the NicheRaw objects for each counter 
        and the fits.
        '''
        Es = self.draw_Es()
        cores = gen_core_pos(self.radius, cfg, N_events=self.N_events)
        zeniths = gen_zeniths(self.N_events)
        # zeniths[zeniths>np.deg2rad(45.)] = np.deg2rad(45.)
        params_dict = {
                'E': Es,
                'xmax': Xmax_of_lE(np.log10(Es)),     
                'nmax': Nmax(Es),
                'zenith': zeniths,
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
        for counter_name in cfg.active_counters:
            params_dict[counter_name] = np.full(self.N_events,np.NAN)
        return pd.DataFrame.from_dict(params_dict)

def draw_Es(N_events: int, lEmin: float = 14., lEmax: float = 17., gamma: float = -3.) -> np.ndarray:
    '''This method draws random energies within the interval 
    specified from a spectrum specified by gamma.
    '''
    Emin = 10**lEmin
    Emax = 10**lEmax
    cdf = np.random.uniform(size = N_events)
    p = 1 + gamma
    return (cdf*(Emax**p - Emin**p) + Emin**p)**(1/p)

def gen_events(N_events: int, Emin: float = 14., Emax: float = 17., gamma: float = -3.) -> list[Event]:
    Es = draw_Es(N_events, Emin, Emax, gamma)
    xmaxs = Xmax_of_lE(np.log10(Es))
    nmaxs = Nmax(Es)
    zeniths = gen_zeniths(N_events)
    azimuths = gen_azimuths(N_events)
    cores = np.zeros((N_events,3))
    X0s = np.zeros_like(cores[:,0])
    Lambdas = np.full_like(cores[:,0], 70.)
    retlist = [Event(E,x,n,z,a,*c,x0,l) for E,x,n,z,a,c,x0,l in zip(Es,xmaxs,nmaxs,zeniths,azimuths,cores,X0s,Lambdas)] # type: ignore
    return retlist

def gen_params_in_bins() -> list[MCParams]:
    '''This function creates a list of shower parameter Dataframes for each energy bin.
    '''
    param_list = []
    for low, high in zip(E_BIN_EDGES[:-1], E_BIN_EDGES[1:]):
        param_list.append(MCParams(N_events=N_THROWN, lEmin=low, lEmax=high))
    return param_list

def showlib_infile(E: float, thinrat: str, maxweight: str, no: int) -> str:
    '''This function generates a corsika infile for a given event.
    '''
    if no <10:
        n = f'0{no}'
    else:
        n = f'{no}'
    egev = E * 1.e-9
    # tel_def = TelescopeDefinition(cfg.positions_array, cfg.radii)
    # counter_pos = np.round(tel_def.shift_counters(event.core_location) * 100.) #cm
    string =  (f'RUNNR   0000{n}                        number of run\n'
    f'EVTNR   1                             no of first shower event\n'
    f'SEED    90000{n}  0  0                  seed for hadronic part\n'
    f'SEED    90000{n}  0  0                  seed for EGS4 part\n'
    f'NSHOW   1                             no of showers to simulate\n'
    f'PRMPAR  14                            primary particle code (proton)\n'
    f'ERANGE  {egev:.2e} {egev:.2e}           energy range of primary particle   \n'
    f'ESLOPE  -1.0                          slope of energy spectrum\n'
    f'THETAP  0. 35.                       range of zenith angle (deg)\n'
    f'PHIP    0. 360                       range of azimuth angle (deg)\n'
    f'THIN    {thinrat} {maxweight} 0.0           thinning parameters\n'
    f'ECUTS   0.3 0.3 0.001 0.001           energy cuts for particles (15 MeV for e, below Cherenkov thresh)\n'
    f'ATMOSPHERE 11 T                       use atmosphere 11 (Utah average) with refractions\n'
    f'MAGNET  21.95 46.40                   magnetic field (TA .. Middle Drum)\n'
    f'ARRANG  0.0			      rotation of array to north (X along magnetic north) Declination assumed 11.9767 deg\n'
    f'OBSLEV  1.534e5                       observation level (in cm) (ground at lowest NICHE counter)\n'
    f'PAROUT  F F                           no DATnnnnnn, no DATnnnnnn.tab\n'
    f'DATBAS  F                             write database file\n'
    f'LONGI   T 1 T T                       create logitudinal info & fit\n')
    return string

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sys
    from utils import *
    plt.ion()
    # evs = gen_events(64,Emin=15.5,gamma=-2)

    # plt.figure()
    # plt.hist([e.E for e in evs])
    # from gen_ckv_signals import *


    lE = float(sys.argv[1])
    thinrat = str(sys.argv[2])
    maxweight = str(sys.argv[3])
    E = 10**lE

    showlib_dir = Path(f'showlib/log10E_{lE}')
    showlib_dir.mkdir(exist_ok=True)

    names = []
    for i in range(1,513):
        names.append(f'{str(showlib_dir)}/DAT000{str(i):>03}.in')

    for i, name in enumerate(names):
        write_file(name,showlib_infile(E,thinrat,maxweight,i+1))
    
    # lEs = [15.6,15.7,15.8]

    # i = 1
    # for lE in lEs:
    #     showlib_dir = Path(f'showlib/log10E_{lE}')
    #     showlib_dir.mkdir(exist_ok=True)
    #     E = 10**lE
    #     for j in range(512):
    #         name = f'{str(showlib_dir)}/DAT{str(i):>06}.in'
    #         write_file(name,showlib_infile(E,'1.0e-06','5.0e+00',i))
    #         i += 1
        


    # mc = MCParams(N_events=10000)

    # plt.figure()
    # plt.hist(mc['Xmax'],histtype='step', bins = 100)
    # plt.xlabel('Xmax')

    # plt.figure()
    # plt.scatter(np.log10(mc['E']),mc['Xmax'],s=.01)
    # plt.semilogy()
    # plt.xlabel('log10(E/eV)')
    # plt.ylabel('Xmax')

    # plt.figure()
    # plt.scatter(mc['corex'],mc['corey'],s=.01)
    # plt.xlabel('core x (m)')
    # plt.ylabel('core y (m)')
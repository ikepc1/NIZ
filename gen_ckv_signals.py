import CHASM as ch
import numpy as np
from dataclasses import dataclass
from pathlib import Path

from config import photon_time_bins, CXF_ALTITUDE, MIN_WAVELENGTH, MAX_WAVELENGTH, N_WAVELENGTH_BINS, CHASM_MESH, COUNTER_POSITIONS, COUNTER_POSITIONS_DICT
from counter_config import CounterConfig
from angular_dependence import filter

@dataclass
class Event:
    '''This is the container for parameters of a single event.
    '''
    E: float
    Xmax: float   
    Nmax: float
    zenith: float
    azimuth: float
    corex: float
    corey: float
    corez: float
    X0: float
    Lambda: float

    def __post_init__(self) -> None:
        if self.zenith < 0.:
            self.zenith += 2*np.pi
        if self.zenith > 2*np.pi:
            self.zenith -= 2*np.pi
        if self.azimuth < 0.:
            self.azimuth += 2*np.pi
        if self.azimuth > 2*np.pi:
            self.azimuth -= 2*np.pi

    @property
    def variable_parameters(self) -> list:
        return [self.Xmax, self.Nmax, self.X0]

    @property
    def core_altitude(self) -> float:
        '''This is the altitude of the core.
        '''
        return CXF_ALTITUDE + self.corez
    
    @property
    def core_location(self) -> np.ndarray:
        '''This is a vector to the core location in cxf coords.
        '''
        return np.array([self.corex, self.corey, self.corez])
    
    @property
    def needs_curved_atm(self) -> bool:
        '''This is for whether the event needs a curved atmosphere correction.
        '''
        if self.zenith > np.radians(60.):
            return True
        else:
            return False

@dataclass
class TelescopeDefinition:
    '''This is the container for information about the telescopes location.
    '''
    cxf_positions: np.ndarray
    radii: float

    def shift_counters(self, core_loc: np.ndarray) -> np.ndarray:
        '''This method transforms the coordinates of counters with respect to the 
        cxf to coordinates with respect to the shower core.
        '''
        return self.cxf_positions - core_loc

def cut_photon_zeniths(sig: ch.ShowerSignal, max_zenith = np.pi/2) -> np.ndarray:
    '''This function sets the value of each photon bunch arriving with a zenith
    angle greater than max_zenith to zero.
    '''
    # survival_fractions = filter(np.arccos(sig.cos_theta))
    sig.photons[sig.cos_theta[:,np.newaxis,:] < np.cos(max_zenith)] = 0.
    return sig.photons #* survival_fractions

def extract_signal(sim: ch.ShowerSimulation,att: bool) -> ch.ShowerSignal:
    '''This function is a wrapper for the run method of the CHASM sim.
    '''
    return sim.run(att=att,mesh=CHASM_MESH)

@dataclass
class CherenkovOutput:
    '''This is the container for the cherenkov light of a shower.
    '''
    photons: np.ndarray
    times: np.ndarray
    cfg: CounterConfig

class GetCkv:
    def __init__(self, cfg: CounterConfig) -> None:
        self.cfg = cfg
        self.sim = ch.ShowerSimulation()
        self.tel_def = TelescopeDefinition(cfg.positions_array, cfg.radii)

    def run(self, event: Event) -> CherenkovOutput:
        '''This function adds elements to the CHASM sim, runs it, and returns 
        the photon counts and times.
        '''
        self.sim.add(ch.GHShower(event.Xmax, event.Nmax, event.X0, event.Lambda))
        self.sim.add(ch.DownwardAxis(event.zenith, 
                                event.azimuth, 
                                event.core_altitude,
                                event.needs_curved_atm))
        self.sim.add(ch.SphericalCounters(self.tel_def.shift_counters(event.core_location),
                                        self.tel_def.radii))
        self.sim.add(ch.Yield(MIN_WAVELENGTH, MAX_WAVELENGTH, N_WAVELENGTH_BINS))
        sig = extract_signal(self.sim, att = True)
        photons = cut_photon_zeniths(sig, self.cfg.max_photon_zenith)
        # photons = sig.photons
        # return photons, sig.times
        return CherenkovOutput(photons, sig.times, self.cfg)

def get_ckv(event: Event, cfg: CounterConfig, att: bool = True) -> CherenkovOutput:
    '''This function adds elements to the CHASM sim, runs it, and returns 
    the photon counts and times.
    '''
    tel_def = TelescopeDefinition(cfg.positions_array, cfg.radii)
    sim = ch.ShowerSimulation()
    sim.add(ch.GHShower(event.Xmax, event.Nmax, event.X0, event.Lambda))
    sim.add(ch.DownwardAxis(event.zenith, 
                            event.azimuth, 
                            event.core_altitude,
                            event.needs_curved_atm))
    sim.add(ch.SphericalCounters(tel_def.shift_counters(event.core_location),
                                    tel_def.radii))
    sim.add(ch.Yield(MIN_WAVELENGTH, MAX_WAVELENGTH, N_WAVELENGTH_BINS))
    sig = extract_signal(sim,att)
    photons = cut_photon_zeniths(sig, cfg.max_photon_zenith)
    # photons = sig.photons
    # return photons, sig.times
    return CherenkovOutput(photons, sig.times, cfg)

LIST_POSITIONS = {}

def read_in_corsika(file: str, cfg: CounterConfig) -> CherenkovOutput:
    ''''''
    timebins = photon_time_bins()
    ei = ch.EventioWrapper(file)
    photons_array = np.empty((len(cfg.active_counters), 1, timebins.size - 1))
    photons_dict = {}
    for i,c in enumerate(COUNTER_POSITIONS_DICT):
        times = np.array(ei.get_photon_times(i))
        photons = np.array(ei.get_photons(i))
        thetas = np.array(ei.get_zeniths(i))
        photons[thetas>cfg.max_photon_zenith] = 0.
        photons_dict[c] = np.histogram(times, bins=timebins, weights = photons, density = False)[0]
    photon_times = timebins[:-1] + np.diff(timebins)/2
    photon_times = np.array([photon_times for i in range(len(cfg.active_counters))])
    for i,c in enumerate(cfg.active_counters):
        photons_array[i,0,:] = photons_dict[c]
    return CherenkovOutput(photons_array, photon_times, cfg)

def tilemask(tilearray: np.ndarray, cfg: CounterConfig, shift: np.ndarray = np.array([0.,0.,0.])) -> np.ndarray:
    '''This function selects the tiles which most closely match the niche array.
    '''
    counter_pos = cfg.positions_array - shift
    diffs = np.sqrt(((tilearray[:,np.newaxis,:] - counter_pos)**2).sum(axis = 2))
    return diffs.argmin(axis=0)

def read_tilefile(file: Path, cfg: CounterConfig) -> CherenkovOutput:
    timebins = photon_time_bins()
    ei = ch.EventioWrapper(file)
    photons_array = np.empty((ei.n_telescopes, 1, timebins.size - 1))
    for i in range(ei.n_telescopes):
        times = np.array(ei.get_photon_times(i))
        photons = np.array(ei.get_photons(i))
        thetas = np.array(ei.get_zeniths(i))
        photons[thetas>cfg.max_photon_zenith] = 0.
        photons_array[i,0,:] = np.histogram(times, bins=timebins, weights = photons, density = False)[0]
    photon_times = timebins[:-1] + np.diff(timebins)/2
    photon_times = np.array([photon_times for i in range(ei.n_telescopes)])
    return photons_array, photon_times, ei.counter_vectors

def ckv_from_tilefile(ei: ch.EventioWrapper, cfg: CounterConfig, shift: np.ndarray = np.array([0.,0.,0.])) -> CherenkovOutput:
    ''''''
    timebins = photon_time_bins()
    mask = tilemask(ei.counter_vectors, cfg, shift)
    photons_array = np.empty((len(cfg.active_counters), 1, timebins.size - 1))
    for j,i in enumerate(mask):
        times = np.array(ei.get_photon_times(i))
        photons = np.array(ei.get_photons(i))
        thetas = np.array(ei.get_zeniths(i))
        photons[thetas>cfg.max_photon_zenith] = 0.
        photons_array[j,0,:] = np.histogram(times, bins=timebins, weights = photons, density = False)[0]
    photon_times = timebins[:-1] + np.diff(timebins)/2
    photon_times = np.array([photon_times for i in range(ei.n_telescopes)])
    # return photons_array, photon_times, ei.counter_vectors[mask]
    return CherenkovOutput(photons_array, photon_times, cfg)

def counter_bottom() -> np.ndarray:
    '''This is the centroid of the active counter's positions.
    '''
    avgx = COUNTER_POSITIONS[:,0].mean()
    avgy = COUNTER_POSITIONS[:,1].mean()
    avgz = COUNTER_POSITIONS[:,2].min()
    return np.array([avgx, avgy, avgz])

def counter_center() -> np.ndarray:
    '''This is the centroid of the active counter's positions.
    '''
    avgx = COUNTER_POSITIONS[:,0].mean()
    avgy = COUNTER_POSITIONS[:,1].mean()
    avgz = COUNTER_POSITIONS[:,2].mean()
    return np.array([avgx, avgy, avgz])
    
def corsika_infile(event: Event, no: int, shift: np.ndarray = np.array([0.,0.,0.])) -> str:
    '''This function generates a corsika infile for a given event.
    '''
    if no <10:
        n = f'0{no}'
    else:
        n = f'{no}'
    egev = event.E * 1.e-9
    zenithdeg = np.rad2deg(event.zenith)
    azmuthdeg = np.rad2deg(event.azimuth) - 180.
    # tel_def = TelescopeDefinition(cfg.positions_array, cfg.radii)
    # counter_pos = np.round(tel_def.shift_counters(event.core_location) * 100.) #cm
    counter_pos = (COUNTER_POSITIONS - counter_bottom() + shift) * 100.
    obslev = event.core_altitude * 100. #cm
    string =  (f'RUNNR   0000{n}                        number of run\n'
    f'EVTNR   1                             no of first shower event\n'
    f'SEED    900001  0  0                  seed for hadronic part\n'
    f'SEED    900002  0  0                  seed for EGS4 part\n'
    f'SEED    900003  0  0                  seed for Cherenkov part\n'
    f'SEED    900004  0  0                  seed for CSCAT part\n'
    f'NSHOW   1                             no of showers to simulate\n'
    f'PRMPAR  14                            primary particle code (proton)\n'
    f'ERANGE  {egev:.2e} {egev:.2e}           energy range of primary particle   \n'
    f'ESLOPE  -1.0                          slope of energy spectrum\n'
    f'THETAP  {zenithdeg:.3e} {zenithdeg:.3e}                       range of zenith angle (deg)\n'
    f'PHIP    {azmuthdeg:.3e} {azmuthdeg:.3e}                       range of azimuth angle (deg)\n'
    f'THIN    1.0e-06 1.0E+2 0.0           thinning parameters\n'
    f'ECUTS   0.3 0.3 0.001 0.001           energy cuts for particles (15 MeV for e, below Cherenkov thresh)\n'
    f'ATMOSPHERE 11 T                       use atmosphere 11 (Utah average) with refractions\n'
    f'CERSIZ  1.0			      bunch size for Cherenkov photons\n'
    f'CERFIL  0			      Cherenkov output to particle file\n'
    f'CWAVLG  300. 450.		      Cherenkov wavelength band (for NICHE)\n'
    f'CSCAT   1 0. 0.	         	      resample Cherenkov events scattered over circle\n'
    f'MAGNET  21.95 46.40                   magnetic field (TA .. Middle Drum)\n'
    f'ARRANG  0.0			      rotation of array to north (X along magnetic north) Declination assumed 11.9767 deg\n'
    f'OBSLEV  {obslev:.3e}                       observation level (in cm) (ground at lowest NICHE counter)\n'
    f'CERQEF F T F\n')
    for pos in counter_pos:
        string += f'TELESCOPE {pos[0]} {pos[1]} {pos[2]} 5.08\n'
    string += (f'PAROUT  F F                           no DATnnnnnn, no DATnnnnnn.tab\n'
    f'DATBAS  T                             write database file\n'
    f'LONGI   T 1 T T                       create logitudinal info & fit\n')
    return string

def biggrid(spacing: float) -> np.ndarray:
    '''This function generates a grid 100 meters bigger than NICHE on each side with 
    a given spacing.
    '''
    xrange = COUNTER_POSITIONS[:,0].max() - COUNTER_POSITIONS[:,0].min() + 200.
    yrange = COUNTER_POSITIONS[:,1].max() - COUNTER_POSITIONS[:,1].min() + 200.
    x = np.arange(-xrange/2, xrange/2, spacing)
    y = np.arange(-yrange/2, yrange/2, spacing)
    xx,yy = np.meshgrid(x,y)
    zz = np.full_like(xx.flatten(), COUNTER_POSITIONS[:,2].mean())
    retarray = np.round(np.vstack((xx.flatten(),yy.flatten(), zz)).T * 100.) #convert to cm
    return retarray

def tile_infile(event: Event, grid_spacing: float, no: int) -> str:
    grid = biggrid(grid_spacing)
    if no <10:
        n = f'0{no}'
    else:
        n = f'{no}'
    egev = event.E * 1.e-9
    zenithdeg = np.rad2deg(event.zenith)
    azmuthdeg = np.rad2deg(event.azimuth) - 180.
    obslev = event.core_altitude * 100.
    string =  (f'RUNNR   0000{n}                        number of run\n'
    f'EVTNR   1                             no of first shower event\n'
    f'SEED    900001  0  0                  seed for hadronic part\n'
    f'SEED    900002  0  0                  seed for EGS4 part\n'
    f'SEED    900003  0  0                  seed for Cherenkov part\n'
    f'SEED    900004  0  0                  seed for CSCAT part\n'
    f'NSHOW   1                             no of showers to simulate\n'
    f'PRMPAR  14                            primary particle code (proton)\n'
    f'ERANGE  {egev:.2e} {egev:.2e}           energy range of primary particle   \n'
    f'ESLOPE  -1.0                          slope of energy spectrum\n'
    f'THETAP  {zenithdeg:.3e} {zenithdeg:.3e}                       range of zenith angle (deg)\n'
    f'PHIP    {azmuthdeg:.3e} {azmuthdeg:.3e}                       range of azimuth angle (deg)\n'
    f'THIN    1.0e-06 1.0E+2 0.0           thinning parameters\n'
    f'ECUTS   0.3 0.3 0.001 0.001           energy cuts for particles (15 MeV for e, below Cherenkov thresh)\n'
    f'ATMOSPHERE 11 T                       use atmosphere 11 (Utah average) with refractions\n'
    f'CERSIZ  1.0			      bunch size for Cherenkov photons\n'
    f'CERFIL  0			      Cherenkov output to particle file\n'
    f'CWAVLG  300. 450.		      Cherenkov wavelength band (for NICHE)\n'
    f'CSCAT   1 0. 0.	         	      resample Cherenkov events scattered over circle\n'
    f'MAGNET  21.95 46.40                   magnetic field (TA .. Middle Drum)\n'
    f'ARRANG  0.0			      rotation of array to north (X along magnetic north) Declination assumed 11.9767 deg\n'
    f'OBSLEV  {obslev:.3e}                       observation level (in cm) (ground at lowest NICHE counter)\n'
    f'CERQEF F T F\n')
    for pos in grid:
        string += f'TELESCOPE {pos[0]} {pos[1]} {pos[2]} 5.08\n'
    string += (f'PAROUT  F F                           no DATnnnnnn, no DATnnnnnn.tab\n'
    f'DATBAS  T                             write database file\n'
    f'LONGI   T 1 T T                       create logitudinal info & fit\n')
    return string

if __name__ == '__main__':
    from fit import *
    import matplotlib.pyplot as plt
    from counter_config import *
    plt.ion()
    from utils import plot_event, plot_generator, get_data_files, preceding_noise_files
    data_date_and_time = '20200616050451'
    cfg = init_config(data_date_and_time)
    tf = Path('/home/isaac/NIZ/tilefiles/batch/iact_DAT000001.dat')
    ei = ch.EventioWrapper(tf)

    shift1 = cfg.counter_bottom
    shift2 = cfg.counter_bottom + np.array([150.,150.,150.])
    shift3 = cfg.counter_bottom + np.array([-150.,-150.,-150.])

    allcounters = cfg.positions_array - cfg.counter_bottom

    p,t,v = ckv_from_tilefile(tf,cfg,shift1)
    plt.figure()
    plt.scatter(ei.counter_vectors[:,0], ei.counter_vectors[:,1], s = .5, label = 'tiles')
    plt.scatter(allcounters[:,0], allcounters[:,1], label = 'detectors')
    plt.scatter(v[:,0],v[:,1], label='center core selected')
    plt.legend()
    ax = plt.gca()
    ax.set_aspect('equal')
    # plt.scatter(v[mask2,0],v[mask2,1], label='shift1 selected')
    # plt.scatter(v[mask3,0],v[mask3,1], label='shift2 selected')
    # pars = [500.,2.e6,np.deg2rad(40.),np.deg2rad(315.), 450., -660.,-29.,0,70]
    # ev = BasicParams.get_event(pars)
    # string = tile_infile(ev,10.,1)
    # write_file('test.in',string)

import CHASM as ch
import numpy as np
from dataclasses import dataclass

from config import photon_time_bins, CXF_ALTITUDE, MIN_WAVELENGTH, MAX_WAVELENGTH, N_WAVELENGTH_BINS, CHASM_MESH, CounterConfig

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
    radii: np.ndarray

    def shift_counters(self, core_loc: np.ndarray) -> np.ndarray:
        '''This method transforms the coordinates of counters with respect to the 
        cxf to coordinates with respect to the shower core.
        '''
        return self.cxf_positions - core_loc

def cut_photon_zeniths(sig: ch.ShowerSignal, max_zenith = np.pi/2) -> np.ndarray:
    '''This function sets the value of each photon bunch arriving with a zenith
    angle greater than max_zenith to zero.
    '''
    sig.photons[sig.cos_theta[:,np.newaxis,:] < np.cos(max_zenith)] = 0.
    return sig.photons

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

    def run(self, event: Event) -> tuple[np.ndarray]:
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
        sig = extract_signal(self.sim)
        photons = cut_photon_zeniths(sig, self.cfg.max_photon_zenith)
        # photons = sig.photons
        # return photons, sig.times
        return CherenkovOutput(photons, sig.times, self.cfg)

def get_ckv(event: Event, cfg: CounterConfig, att: bool = False) -> CherenkovOutput:
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

def read_in_corsika(file: str, cfg: CounterConfig) -> CherenkovOutput:
    ''''''
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
    return CherenkovOutput(photons_array, photon_times, cfg)

def corsika_infile(event: Event, cfg: CounterConfig, no: int) -> str:
    '''This function generates a corsika infile for a given event.
    '''
    if no <10:
        n = f'0{no}'
    else:
        n = f'{no}'
    egev = event.E * 1.e-9
    zenithdeg = np.rad2deg(event.zenith)
    azmuthdeg = np.rad2deg(event.azimuth) - 180.
    tel_def = TelescopeDefinition(cfg.positions_array, cfg.radii)
    counter_pos = np.round(tel_def.shift_counters(event.core_location) * 100.) #cm
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
    f'CERQEF F F F\n')
    for pos in counter_pos:
        string += f'TELESCOPE {pos[0]} {pos[1]} {pos[2]} 5.08\n'
    string += (f'PAROUT  F F                           no DATnnnnnn, no DATnnnnnn.tab\n'
    f'DATBAS  T                             write database file\n'
    f'LONGI   T 1 T T                       create logitudinal info & fit\n')
    return string
    


if __name__ == '__main__':
    from fit import *
    import matplotlib.pyplot as plt
    from utils import write_file
    plt.ion()
    from utils import plot_event, plot_generator, get_data_files, preceding_noise_file
    data_date_and_time = '20190504034237'
    data_files = get_data_files(data_date_and_time)
    noise_files = [preceding_noise_file(f) for f in data_files]
    cfg = CounterConfig(data_files, noise_files)
    
    pars = [500.,2.e6,np.deg2rad(40.),np.deg2rad(315.), 450., -660.,-29.,0,70]
    ev = BasicParams.get_event(pars)
    string = corsika_infile(ev,cfg)
    write_file('test.in',string)

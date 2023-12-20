import CHASM as ch
import numpy as np
from dataclasses import dataclass
from config import CXF_ALTITUDE, MIN_WAVELENGTH, MAX_WAVELENGTH, N_WAVELENGTH_BINS, CHASM_MESH, CounterConfig

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

def extract_signal(sim: ch.ShowerSimulation) -> ch.ShowerSignal:
    '''This function is a wrapper for the run method of the CHASM sim.
    '''
    return sim.run(att=True,mesh=CHASM_MESH)

@dataclass
class CherenkovOutput:
    '''This is the container for the cherenkov light of a shower.
    '''
    photons: np.ndarray
    times: np.ndarray
    cfg: CounterConfig

def get_ckv(event: Event, cfg: CounterConfig) -> tuple[np.ndarray]:
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
    sig = extract_signal(sim)
    photons = cut_photon_zeniths(sig, cfg.max_photon_zenith)
    # photons = sig.photons
    # return photons, sig.times
    return CherenkovOutput(photons, sig.times, cfg)
    

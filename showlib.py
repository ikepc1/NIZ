from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass, field

def particle_maxi(file: Path) -> int:
    '''This function returns the final line of the particle data from
    a corsika longitudinal file.
    '''
    with file.open('r') as f:
        lines = f.readlines()
    for i,line in enumerate(lines[2:]):
        if not line[5].isnumeric():
            return int(i + 2)
    raise Exception('Probably not a longfile...')

def read_longfile(file: Path) -> pd.DataFrame:
    '''This function copies the particle information from a corsika 
    longitudinal file into a pandas dataframe.
    '''
    with file.open('r') as f:
        lines = f.readlines()
    colnames = lines[1].split()
    dictlist = []
    for line in lines[2:particle_maxi(file)]:
        dictlist.append({n:float(val) for n,val in zip(colnames,line.split())})
    return pd.DataFrame(dictlist)

def read_cors_fit(file: Path) -> dict[str,float]:
    '''This function copies the particle information from a corsika 
    longitudinal file into a pandas dataframe.
    '''
    with file.open('r') as f:
        lines = f.readlines()
    pnames = [f'P{n}' for n in range(1,7)]
    pars = lines[-4].split()[2:]
    pardict = {name:float(par) for name, par in zip(pnames,pars)}
    pardict['CHI2'] = float(lines[-3].split()[2])
    return pardict

def get_shower_angles(file: Path) -> tuple[float,float]:
    '''This function reads the output file for the angles.
    '''
    with file.open('r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith(' PRIMARY ANGLES ARE:'):
            words = line.split()
            return float(words[5]), float(words[9])
    raise Exception('Probably not an outfile...')

def nmax_of_E(E: float | np.ndarray) -> float | np.ndarray:
    return E*6.06842266e-10 - 1.29163718e+04

def xmax_of_log10E(lE: float | np.ndarray) -> float | np.ndarray:
    return lE*56.14314754 - 274.14784308

def adjust_profile(E: float, nch: np.ndarray) -> np.ndarray:
    '''This function scales a shower profile to reflect the thrown energy.
    '''
    lEs = lE_drawers()
    lE = np.log10(E)
    closest_lE = lEs[np.abs(lEs-lE).argmin()]
    nmax_at_E = nmax_of_E(E)
    nmax_at_drawer = nmax_of_E(10.**closest_lE)
    return nch * (nmax_at_E/nmax_at_drawer)

def lE_drawers() -> np.ndarray:
    drawers = sorted(Path('showlib').iterdir())
    lEs = np.array([float(drawer.name[-4:]) for drawer in drawers])
    return lEs

def adjust_xmax(E: float, nch: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''This function shifts the shower profile so its xmax reflects the thrown energy.
    '''
    lEs = lE_drawers()
    lE = np.log10(E)
    closest_lE = lEs[np.abs(lEs-lE).argmin()]
    xmax_at_lE = xmax_of_log10E(lE)
    xmax_at_drawer = xmax_of_log10E(closest_lE)
    diff = xmax_at_lE / xmax_at_drawer
    scaled_depths = diff * X
    new_depths = np.arange(X.min(),np.round(scaled_depths.max()))
    new_nch = np.interp(new_depths, scaled_depths, nch)
    return new_nch, new_depths
    
def adjust_fitxmax(E: float, xmax: float) -> float:
    '''This function shifts the shower profile so its xmax reflects the thrown energy.
    '''
    lEs = lE_drawers()
    lE = np.log10(E)
    closest_lE = lEs[np.abs(lEs-lE).argmin()]
    xmax_at_lE = xmax_of_log10E(lE)
    xmax_at_drawer = xmax_of_log10E(closest_lE)
    diff = xmax_at_lE / xmax_at_drawer
    scaled_xmax = diff * xmax
    return scaled_xmax

def adjust_fitnmax(E: float, nmax: float) -> float:
    '''This function shifts the shower profile so its xmax reflects the thrown energy.
    '''
    lEs = lE_drawers()
    lE = np.log10(E)
    closest_lE = lEs[np.abs(lEs-lE).argmin()]
    nmax_at_E = nmax_of_E(E)
    nmax_at_drawer = nmax_of_E(10.**closest_lE)
    return nmax * (nmax_at_E/nmax_at_drawer)


@dataclass
class LibraryShower:
    '''This is a container for a shower drawn from the shower library.
    '''
    E: float
    depths: np.ndarray = field(repr=False)
    nch: np.ndarray = field(repr=False)
    corfit: dict[str,float] = field(repr=False)

    @property
    def xmax(self) -> float:
        return self.depths[self.nch.argmax()]
    
    @property
    def nmax(self) -> float:
        return self.nch.max()

def get_drawer(E: float) -> Path:
    '''This function returns the drawer closest to the energy specified by E.
    '''
    lE = np.log10(E)
    drawers = sorted(Path('showlib').iterdir())
    lEs = np.array([float(drawer.name[-4:]) for drawer in drawers])
    return drawers[np.abs(lEs-lE).argmin()]

def draw_shower_from_library(eid: tuple[float,int]) -> LibraryShower:
    '''This function draws a random shower of energy E from the shower library.
    '''
    E = eid[0]
    showid = eid[1]
    drawer = get_drawer(E)
    longfile = drawer / f'DAT000{str(showid):>03}.long'
    outfile = drawer / f'DAT000{str(showid):>03}.out'
    showerdf = read_longfile(longfile)
    depths = showerdf.DEPTH.to_numpy()
    nch = showerdf.CHARGED.to_numpy()
    nch -= nch[0]
    adjusted_nch = adjust_profile(E,nch)
    final_nch, final_depths = adjust_xmax(E,adjusted_nch,depths)
    # angles = get_shower_angles(outfile)
    corfit = read_cors_fit(longfile)
    corfit['P1'] = adjust_fitnmax(E,corfit['P1'])
    corfit['P3'] = adjust_fitxmax(E,corfit['P3'])
    return LibraryShower(E, final_depths, final_nch, corfit)
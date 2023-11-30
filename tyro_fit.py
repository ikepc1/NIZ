import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# from datetime import datetime
from dataclasses import dataclass

from config import COUNTER_POSITIONS, CounterConfig
from niche_fit import NicheFit, NicheRaw

def hex_to_s(hex_ts: str) -> float:
    '''This function converts the hexadecimal string values to a value in 
    nanoseconds.
    '''
    return int(hex_ts, 16) * 5.# * 1.e-9

def linear_function(t: float | np.ndarray, C1: float, C0: float) -> float | np.ndarray:
    '''This is the fit function which returns a cartesian coordinate as a 
    function of time.
    '''
    return C1 + C0*t

def plot_detectors() -> None:
    '''This function adds the NICHE counters to a plot.
    '''
    plt.scatter(COUNTER_POSITIONS[:,0], COUNTER_POSITIONS[:,1], c = 'b',label = 'detectors')
    ax = plt.gca()
    ax.set_aspect('equal')
    # plt.xlim(COUNTER_POSITIONS[:,0].min() - 100, COUNTER_POSITIONS[:,0].max() + 100)
    # plt.ylim(COUNTER_POSITIONS[:,1].min() - 100, COUNTER_POSITIONS[:,1].max() + 100)

@dataclass
class TyroFit:
    '''This is the container for the results of a Tyro fit.
    '''
    t: np.ndarray
    pa: np.ndarray
    counter_pos: np.ndarray

def tyro(event: list[NicheFit]) -> TyroFit:
    '''This function returns the Tyro estimate for the axis position.
    '''
    PAs = np.array([fit.intsignal for fit in event])
    times = np.array([fit.trigtime() for fit in event])
    positions = np.array([fit.position for fit in event])
    sig = 1 / (PAs/PAs.sum())
    # ts = np.array(times - times.min())
    return TyroFit(times, PAs, positions)

def plot_triggers(fit: TyroFit) -> None:
    '''This function plots the detectors that triggered where the size of 
    each scatter point is the pulse area and the color is the time.
    '''
    plt.scatter(fit.counter_pos[:,0], fit.counter_pos[:,1], s = fit.pa, c=fit.t)#, cmap = 'inferno')
    plt.colorbar(label='nanoseconds')

def plot_event(fit: TyroFit) -> None:
    '''This function plots an individual Niche event.
    '''
    plt.ion()
    plt.figure()
    plot_detectors()
    plot_triggers(fit)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Niche Event')
    plt.grid()
    plt.legend()





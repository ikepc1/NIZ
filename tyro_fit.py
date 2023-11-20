import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# from datetime import datetime
from dataclasses import dataclass

from config import COUNTER_POSITIONS, CounterConfig
# from niche_fit import NicheFit, NicheRaw

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
    xfit: np.ndarray
    yfit: np.ndarray
    zfit: np.ndarray
    t: np.ndarray
    pa: np.ndarray
    counter_pos: np.ndarray

def tyro(event: pd.Series, cfg: CounterConfig) -> TyroFit:
    '''This function returns the Tyro estimate for the axis position.
    '''
    PAs = np.array([fit.intsignal for fit in event[event.notna()]])
    times = np.array([hex_to_s(fit.__str__()[-8:]) for fit in event[event.notna()]])
    positions = cfg.positions_array[event.notna()]
    sig = 1 / (PAs/PAs.sum())
    ts = np.array(times - times.min())
    X,_ = curve_fit(linear_function, ts, positions[:,0], sigma = sig)
    Y,_ = curve_fit(linear_function, ts, positions[:,1], sigma = sig)
    Z,_ = curve_fit(linear_function, ts, positions[:,2], sigma = sig)
    return TyroFit(X, Y, Z, ts, PAs, positions)

def plot_triggers(fit: TyroFit) -> None:
    '''This function plots the detectors that triggered where the size of 
    each scatter point is the pulse area and the color is the time.
    '''
    plt.scatter(fit.counter_pos[:,0], fit.counter_pos[:,1], s = fit.pa, c=fit.t)#, cmap = 'inferno')
    plt.colorbar(label='nanoseconds')

def plot_axis_estimate(fit: TyroFit) -> None:
    '''This function plots the estimate for the 2d projection of the shower
    axis.
    '''
    x = linear_function(fit.t, *fit.xfit)
    y = linear_function(fit.t, *fit.yfit)
    plt.plot(x,y,c='k',linewidth=.5)

def plot_event(fit: TyroFit) -> None:
    '''This function plots an individual Niche event.
    '''
    plt.ion()
    plt.figure()
    plot_detectors()
    plot_triggers(fit)
    plot_axis_estimate(fit)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Niche Event')
    plt.grid()
    plt.legend()





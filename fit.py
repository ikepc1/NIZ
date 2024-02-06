from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.optimize import curve_fit
from scipy.stats import norm
import numpy as np
from dataclasses import dataclass, field
import pandas as pd
from functools import cached_property
from typing import Protocol, Callable
from abc import ABC, abstractmethod

from utils import run_multiprocessing
from tyro_fit import tyro, TyroFit
from niche_plane import NichePlane, NicheFit
from gen_ckv_signals import Event, get_ckv, CherenkovOutput
from process_showers import ProcessEvents
from config import CounterConfig, COUNTER_NO, NAMES, COUNTER_POSITIONS, WAVEFORM_SIZE, TRIGGER_POSITION, NICHE_TIMEBIN_SIZE, COUNTER_QE, COUNTER_FADC_PER_PE
from trigger import TriggerSim, rawphotons2fadc
from noise import read_noise_file

def E(Nmax: float) -> float:
    '''This function estimates the energy of an event based on Nmax.
    '''
    return Nmax / 1.3e-9
    
# def get_event(parameters: np.ndarray) -> Event:
#     '''This function maps the list of shower parameters to the Shower
#     Event data container.
#     '''
#     nmax = np.exp(parameters[1])
#     return Event(
#     E=E(nmax),
#     Xmax=np.exp(parameters[0]),
#     Nmax=nmax,
#     zenith= parameters[2],
#     azimuth= parameters[3],
#     corex=np.exp(parameters[4]),
#     corey=-np.exp(parameters[5]),
#     corez=parameters[6],
#     X0=parameters[7],
#     Lambda=parameters[8]
#     )

# def get_event(parameters: np.ndarray) -> Event:
#     '''This function maps the list of shower parameters to the Shower
#     Event data container.
#     '''
#     s = np.exp(parameters[1])
#     xmax = np.exp(parameters[0])
#     nmax = s - xmax
#     return Event(
#     E=E(nmax),
#     Xmax=xmax,
#     Nmax=nmax,
#     zenith= parameters[2],
#     azimuth= parameters[3],
#     corex=np.exp(parameters[4]),
#     corey=-np.exp(parameters[5]),
#     corez=parameters[6],
#     X0=parameters[7],
#     Lambda=parameters[8]
#     )

def remove_noise(f: NicheFit) -> np.ndarray:
    '''This function calculates the number of incident photons from a 
    niche fadc trace.
    '''
    wf = f.tunka_fit(f.t,f.peaktime,f.peak,f.risetime,f.falltime,f.baseline)
    wf -= f.baseline
    return wf

def fadc_dict(nfits: list[NicheFit]) -> dict[str,np.ndarray]:
    '''This function takes the niche data traces and returns a dictionary
    mapping the counter names to arrays of incident Cherenkov photons.
    '''
    return {f.name:remove_noise(f) for f in nfits}

def full_wf_times(trigtime: float) -> np.ndarray:
    '''This '''
    times_ns = np.arange(NICHE_TIMEBIN_SIZE*WAVEFORM_SIZE, step = NICHE_TIMEBIN_SIZE)
    times_ns -= times_ns[TRIGGER_POSITION]
    times_ns += trigtime
    return times_ns

def get_times_array(nfits: list[NicheFit]) -> np.ndarray:
    ''''''
    trigtimes = np.array([nfit.trigtime() for nfit in nfits])
    trigtimes = np.float64(trigtimes - trigtimes.min())
    return trigtimes

def ckv_signal_dict(ckv: CherenkovOutput) -> tuple[dict[str,tuple[np.ndarray]],np.ndarray]:
    '''This function compiles the waveforms from CHASM into a dictionary where
    the keys are the counter names.
    '''
    photons, times = rawphotons2fadc(ckv)
    return {name:p for name, p in zip(ckv.cfg.active_counters, photons)}, times

def do_wf_fit(wf: np.ndarray) -> np.ndarray:
    '''This function fits a waveform to the Tunka PMT pulse function. Dont return
    the final value which is the baseline.
    '''
    p0 = (np.argmax(wf),wf.max(),2.,4.,0)
    try:
        pb, _ = curve_fit(NicheFit.tunka_fit,np.arange(len(wf)),wf,p0,2.*np.ones_like(wf))
    except:
        pb = np.zeros(5,dtype=float)
    return pb

def do_pulse_integration(wf: np.ndarray) -> float:
    '''This function finds the pulse area of a waveform.
    '''
    pbs = do_wf_fit(wf)
    intstart = int(np.floor(pbs[0] - 5.*pbs[2]))
    intend   = int(np.ceil(pbs[0] + 5.*pbs[3]))
    intsignal = wf[intstart:intend+1].sum()
    return intsignal

# @dataclass
# class EventFit:
#     '''This class is responible for compiling the features to be fit from
#     real data.
#     '''
#     nfits: list[NicheFit]
#     param_mapper: ParamMapper
#     cfg: CounterConfig = field(repr=False)
#     n_features: int = 4

#     @property
#     def pas(self) -> np.ndarray:
#         return np.array([f.intsignal for f in self.nfits])
    
#     @cached_property
#     def normalized_pas(self) -> np.ndarray:
#         return self.pas / self.pas.sum()
    
#     @property
#     def pa_error(self) -> np.ndarray:
#         return np.array([f.eintsignal for f in self.nfits])
    
#     @cached_property
#     def normalized_pa_error(self) -> np.ndarray:
#         return self.pa_error / self.pas.sum()

#     @property
#     def nfit_dict(self) -> dict[str, NicheFit]:
#         '''This is a dictionary of the Nichefit objects in the real event.
#         '''
#         return {f.name:f for f in self.nfits}

#     @cached_property
#     def biggest_counter(self) -> str:
#         '''This is the name of the counter in the data with the biggest
#         pulse.
#         '''
#         return self.nfits[self.pas.argmax()].name
    
#     @cached_property
#     def biggest_trigtime(self) -> np.datetime64:
#         '''This is the time when the biggest trigger happened.
#         '''
#         return self.nfit_dict[self.biggest_counter].trigtime()
    
#     @cached_property
#     def biggest_peaktime_difference(self) -> np.datetime64:
#         '''This is the time when the biggest trigger happened.
#         '''
#         return self.nfit_dict[self.biggest_counter].ns_diff

#     def adjust_data_peaktime(self, nfit: NicheFit) -> float:
#         '''This method calculates the peaktime of the 'nfit' counter relative to the peaktime of 
#         the largest trigger.
#         '''
#         trigtime_delta = nfit.trigtime() - self.biggest_trigtime
#         peaktime_difference = trigtime_delta.astype('float64') + nfit.ns_diff - self.biggest_peaktime_difference
#         return peaktime_difference
    
#     @cached_property
#     def real_peaktimes(self) -> np.ndarray:
#         return np.array([self.adjust_data_peaktime(f) for f in self.nfits])
    
#     @property
#     def real_peaktime_errors(self) -> np.ndarray:
#         return np.array([f.epeaktime for f in self.nfits])
    
#     def ckv_from_params(self, parameters: np.ndarray) -> np.ndarray:
#         ev = self.param_mapper.get_event(parameters)
#         print(ev)
#         ckv = get_ckv(ev, self.cfg)
#         return ckv

#     @cached_property
#     def real_output(self) -> np.ndarray:
#         '''This is the fitted parameters for each counter triggered in the event. 
#         They are the output data points to be compared to the model.
#         '''
#         pars = [np.array([self.adjust_data_peaktime(f), f.peak, f.risetime, f.falltime]) for f in self.nfits]
#         return np.hstack(pars)
    
#     @cached_property
#     def real_error(self) -> np.ndarray:
#         '''This is the fitted parameters for each counter triggered in the event. 
#         They are the output data points to be compared to the model.
#         '''
#         errs = [np.array([f.epeaktime * NICHE_TIMEBIN_SIZE, f.epeak, f.erisetime, f.efalltime]) for f in self.nfits]
#         return np.hstack(errs)
    
#     @property
#     def input_indices(self) -> np.ndarray:
#         '''This is the enumeration of the terms of the chi-squared statistic.
#         '''
#         return np.arange(self.n_features * len(self.nfits))

#     def get_output(self, parameters: np.ndarray) -> np.ndarray:
#         '''This method gives the data output for a set of shower parameters.
#         '''
#         # ev = self.param_mapper.get_event(parameters)
#         # print(ev)
#         # ckv = get_ckv(ev, self.cfg)
#         ckv = self.ckv_from_params(parameters)
#         sigdict, times = ckv_signal_dict(ckv)

#         #do tunka fits
#         pb_array = np.array([do_wf_fit(sigdict[name])[:-1] for name in self.nfit_dict])

#         #tunka fit returns approximate index of peak, find times those corresponds to
#         peaktimes = np.interp(pb_array[:,0], np.arange(len(times)), times)
        
#         #adjust times so biggest counter peaktime is the start
#         peaktimes -= peaktimes[self.pas.argmax()]
#         pb_array[:,0] = peaktimes
#         return pb_array.flatten()
    
#     def get_pa_output(self, parameters: np.ndarray) -> np.ndarray:
#         '''This method gets the pulse area from a sim.
#         '''
#         ckv = self.ckv_from_params(parameters)
#         sigdict, _ = ckv_signal_dict(ckv)
#         pa_array = np.array([do_pulse_integration(sigdict[name]) for name in self.nfit_dict])
#         return pa_array
    
#     def get_normalized_pa_output(self, parameters: np.ndarray) -> np.ndarray:
#         '''This method gets the pulse area from a sim.
#         '''
#         pa_array = self.get_pa_output(parameters)
#         return pa_array / pa_array.sum()
    
#     def get_peaktimes_output(self, parameters: np.ndarray) -> np.ndarray:
#         ckv = self.ckv_from_params(parameters)
#         sigdict, times = ckv_signal_dict(ckv)

#         #do tunka fits
#         pb_array = np.array([do_wf_fit(sigdict[name])[:-1] for name in self.nfit_dict])

#         #tunka fit returns approximate index of peak, find times those corresponds to
#         peaktimes = np.interp(pb_array[:,0], np.arange(len(times)), times)
        
#         #adjust times so biggest counter peaktime is the start
#         peaktimes -= peaktimes[self.pas.argmax()]
#         return peaktimes

#     def chi_square(self, parameters: np.ndarray) -> float:
#         '''This is a direct calculation of the chi square statistic for a set of shower 
#         parameters.
#         '''
#         output = self.get_output(parameters)
#         return ((self.real_output - output)**2/self.real_error**2).sum()

#     def model(self, chi_square_indices: np.ndarray, parameters: np.ndarray) -> np.ndarray:
#         '''This is the model to be supplied to minuit. The indices are ignored.
#         '''
#         return self.get_output(parameters)
    
#     def tmodel(self, chi_square_indices: np.ndarray, parameters: np.ndarray) -> np.ndarray:
#         '''This is the model to be supplied to minuit. The indices are ignored.
#         '''
#         return self.get_peaktimes_output(parameters)
    
#     def pamodel(self, chi_square_indices: np.ndarray, parameters: np.ndarray) -> np.ndarray:
#         '''This is the model to be supplied to minuit. The indices are ignored.
#         '''
#         return self.get_pa_output(parameters)
    
#     def get_pulse_width_output(self, parameters):
#         ckv = self.ckv_from_params(parameters)
#         sigdict, _ = ckv_signal_dict(ckv)
#         pa_array = np.array([do_wf_fit(sigdict[name])[:-1] for name in self.nfit_dict])
#         pulse_widths = pa_array[:,2] + pa_array[:,3]
#         return pulse_widths
    
#     @property
#     def real_pulse_widths(self):
#         return np.array([f.risetime+f.falltime for f in self.nfits])
    
#     @property
#     def real_pulse_width_error(self):
#         return np.array([np.sqrt(f.erisetime**2+f.efalltime**2) for f in self.nfits])
    
#     def pwmodel(self, chi_square_indices: np.ndarray, parameters: np.ndarray) -> np.ndarray:
#         '''This is the model to be supplied to minuit. The indices are ignored.
#         '''
#         return self.get_pulse_width_output(parameters)

class ParamMapper(Protocol):
    '''This protocol maps a given set of parameters to an actual shower
    event.
    '''

    def get_event(self, pars: np.ndarray) -> Event:
        ...

    def get_params(self, evt: Event) -> np.ndarray:
        ...

    def adjust_guess():
        ...

class FitFeature(ABC):
    '''This is the protocol for a fit feature.
    '''
    def __init__(self, nfits: list[NicheFit], param_mapper: ParamMapper, cfg: CounterConfig) -> None:
        self.nfits = nfits
        self.param_mapper = param_mapper
        self.cfg = cfg
        self.target_parameters = []

    def ckv_from_params(self, parameters: np.ndarray) -> CherenkovOutput:
        '''This method returns the Cherenkov light in each counter at each time bin.
        '''
        ev = self.param_mapper.get_event(parameters)
        print(ev)
        ckv = get_ckv(ev, self.cfg)
        return ckv

    @property
    def pas(self) -> np.ndarray:
        '''This is the area of each pulse in each counter.
        '''
        return np.array([f.intsignal for f in self.nfits])
    
    @property
    def biggest_counter(self) -> str:
        '''This is the name of the counter in the data with the biggest
        pulse.
        '''
        return self.nfits[self.pas.argmax()].name
    
    @property
    def biggest_trigtime(self) -> np.datetime64:
        '''This is the time when the biggest trigger happened.
        '''
        return self.nfit_dict[self.biggest_counter].trigtime()
    
    @property
    def biggest_peaktime_difference(self) -> np.datetime64:
        '''This is the time when the biggest trigger happened.
        '''
        return self.nfit_dict[self.biggest_counter].ns_diff
    
    def adjust_data_peaktime(self, nfit: NicheFit) -> float:
        '''This method calculates the peaktime of the 'nfit' counter relative to the peaktime of 
        the largest trigger.
        '''
        trigtime_delta = nfit.trigtime() - self.biggest_trigtime
        peaktime_difference = trigtime_delta.astype('float64') + nfit.ns_diff - self.biggest_peaktime_difference
        return peaktime_difference
    
    @property
    def nfit_dict(self) -> dict[str, NicheFit]:
        '''This is a dictionary of the Nichefit objects in the real event.
        '''
        return {f.name:f for f in self.nfits}

    def model(self, input_indices: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        '''This is the model to be supplied to minuit. The indices are ignored.
        '''
        return self.get_output(parameters)

    def chi2(self, parameters: np.ndarray) -> float:
        '''This method calculates the chi squared value for a run with the 
        parameters specified.
        '''
        output = self.get_output(parameters)
        return ((self.real_values - output)**2/self.error**2).sum()

    @property
    @abstractmethod
    def real_values() -> np.ndarray:
        '''This property should return the values of the feature in the real data.
        '''

    @property
    @abstractmethod
    def error() -> np.ndarray:
        '''This property should return the errors in the real data.
        '''

    # @property
    # @abstractmethod
    # def target_parameters(self) -> list[str]:
    #     '''This property should return the shower parameters which are best given 
    #     by fitting this feature.
    #     '''

    @property
    @abstractmethod
    def output_size(self) -> int:
        '''This should return the length of the output array.
        '''

    @abstractmethod
    def get_output(self, parameters: np.ndarray) -> np.ndarray:
        '''This method should return the simulated feature when run with the given
        parameters.
        '''

class PeakTimes(FitFeature):
    '''This is the container for the peaktimes fit feature.
    '''

    @property
    def real_values(self) -> np.ndarray:
        return np.array([self.adjust_data_peaktime(f) for f in self.nfits])
    
    @property
    def error(self) -> np.ndarray:
        return np.array([NICHE_TIMEBIN_SIZE * f.epeaktime for f in self.nfits])
    
    @property
    def output_size(self) -> int:
        return len(self.nfits)
    
    def get_output(self, parameters: np.ndarray) -> np.ndarray:
        ckv = self.ckv_from_params(parameters)
        sigdict, times = ckv_signal_dict(ckv)

        #do tunka fits
        pb_array = np.array([do_wf_fit(sigdict[name])[:-1] for name in self.nfit_dict])

        #tunka fit returns approximate index of peak, find times those corresponds to
        peaktimes = np.interp(pb_array[:,0], np.arange(len(times)), times)
        
        #adjust times so biggest counter peaktime is the start
        peaktimes -= peaktimes[self.pas.argmax()]
        return peaktimes

class PulseWidth(FitFeature):
    '''This is the implementation of the pulse width fit feature.
    '''
    def get_output(self, parameters):
        ckv = self.ckv_from_params(parameters)
        sigdict, _ = ckv_signal_dict(ckv)
        pa_array = np.array([do_wf_fit(sigdict[name])[:-1] for name in self.nfit_dict])
        pulse_widths = pa_array[:,2] + pa_array[:,3]
        return pulse_widths
    
    @cached_property
    def real_values(self):
        return np.array([f.risetime+f.falltime for f in self.nfits])
    
    @cached_property
    def error(self):
        return np.array([np.sqrt(f.erisetime**2+f.efalltime**2) for f in self.nfits])
    
    @property
    def output_size(self) -> int:
        return len(self.nfits)

class PulseArea(FitFeature):
    '''This is the implementation of the pulse area fit feature.
    '''
    
    @property
    def real_values(self) -> np.ndarray:
        return self.pas
    
    @property
    def error(self) -> np.ndarray:
        return np.array([f.eintsignal for f in self.nfits])
    
    def get_output(self, parameters: np.ndarray) -> np.ndarray:
        '''This method gets the pulse area from a sim.
        '''
        ckv = self.ckv_from_params(parameters)
        sigdict, _ = ckv_signal_dict(ckv)
        pa_array = np.array([do_pulse_integration(sigdict[name]) for name in self.nfit_dict])
        return pa_array
    
    @property
    def output_size(self) -> int:
        return len(self.nfits)

class AllTunka(FitFeature):
    '''This is the container for all tunka parameters as features simultaneously.
    '''

    @property
    def real_values(self) -> np.ndarray:
        '''This is the fitted parameters for each counter triggered in the event. 
        They are the output data points to be compared to the model.
        '''
        pars = [np.array([self.adjust_data_peaktime(f), f.peak, f.risetime, f.falltime]) for f in self.nfits]
        return np.hstack(pars)
    
    @property
    def error(self) -> np.ndarray:
        '''This is the fitted parameters for each counter triggered in the event. 
        They are the output data points to be compared to the model.
        '''
        errs = [np.array([f.epeaktime * NICHE_TIMEBIN_SIZE, f.epeak, f.erisetime, f.efalltime]) for f in self.nfits]
        return np.hstack(errs)

    def get_output(self, parameters: np.ndarray) -> np.ndarray:
        '''This method gives the data output for a set of shower parameters.
        '''
        ckv = self.ckv_from_params(parameters)
        sigdict, times = ckv_signal_dict(ckv)

        #do tunka fits
        pb_array = np.array([do_wf_fit(sigdict[name])[:-1] for name in self.nfit_dict])

        #tunka fit returns approximate index of peak, find times those corresponds to
        peaktimes = np.interp(pb_array[:,0], np.arange(len(times)), times)
        
        #adjust times so biggest counter peaktime is the start
        peaktimes -= peaktimes[self.pas.argmax()]
        pb_array[:,0] = peaktimes
        return pb_array.flatten()
    
    @property
    def output_size(self) -> int:
        return 4 * len(self.nfits)
    
class Peak(FitFeature):
    '''This is the implementation of the pulse peak fit feature.
    '''

    @property
    def real_values(self) -> np.ndarray:
        return np.array([f.peak for f in self.nfits])
    
    @property
    def error(self) -> np.ndarray:
        return np.array([f.epeak for f in self.nfits])
    
    def get_output(self, parameters: np.ndarray) -> np.ndarray:
        ckv = self.ckv_from_params(parameters)
        sigdict, _ = ckv_signal_dict(ckv)
        pa_array = np.array([do_wf_fit(sigdict[name])[:-1] for name in self.nfit_dict])
        peak_array = pa_array[:,1]
        return peak_array
    
    @property
    def output_size(self) -> int:
        return len(self.nfits)

class NormalizedPulseArea(PulseArea):
    '''This is the implementation of normalized pulse areas fit feature.
    '''
    @property
    def output_size(self) -> int:
        return len(self.nfits)

    @cached_property
    def pa_sum(self) -> float:
        return self.pas.sum()

    @cached_property
    def real_values(self) -> np.ndarray:
        return self.pas / self.pa_sum
    
    @cached_property
    def error(self) -> np.ndarray:
        return super().error / self.pa_sum
    
    def get_output(self, parameters: np.ndarray) -> np.ndarray:
        return super().get_output(parameters) / self.pa_sum

@dataclass
class FitParam:
    '''This is a wrapper for a minuit parameter for easy management.
    '''
    name: str
    value: float
    limits: tuple[float]
    error: float

def make_guess(ty: TyroFit, pf: NichePlane) -> list[FitParam]:
    '''This function makes a guess for the fit parameters.
    '''
    parlist = [
        FitParam('xmax', 300., (300., 600.), 50.),
        FitParam('nmax', 1.e5, (1.e5, 1.e7), 1.e5),
        FitParam('zenith', pf.theta, (pf.theta -.1, pf.theta +.1), np.deg2rad(1.)),
        FitParam('azimuth', pf.phi, (pf.phi -.1, pf.phi +.1), np.deg2rad(1.)),
        FitParam('corex',ty.core_estimate[0],ty.xlimits, 5.),
        FitParam('corey',ty.core_estimate[1],ty.ylimits, 5.),
        FitParam('corez',ty.core_estimate[2],(ty.core_estimate[2] - 1.,ty.core_estimate[2] + 1.), 1.),
        FitParam('x0',0.,(0,100),1),
        FitParam('lambda',70., (0,100),1)
    ]
    return parlist

class LogXNParams:
    @staticmethod
    def get_event(parameters: np.ndarray) -> Event:
        '''This function maps the list of shower parameters to the Shower
        Event data container.
        '''
        nmax = np.exp(parameters[1])
        return Event(
        E=E(nmax),
        Xmax=np.exp(parameters[0]),
        Nmax=nmax,
        zenith= parameters[2],
        azimuth= parameters[3],
        corex= parameters[4],
        corey= parameters[5],
        corez=parameters[6],
        X0=parameters[7],
        Lambda=parameters[8]
        )
    
    @staticmethod
    def get_params(evt: Event) -> np.ndarray:
        return np.array([np.log(evt.Xmax),
                        np.log(evt.Nmax),
                        evt.zenith,
                        evt.azimuth,
                        evt.corex,
                        evt.corey,
                        evt.corez,
                        evt.X0,
                        evt.Lambda])
    
    @staticmethod
    def adjust_guess(parlist: list[FitParam]) -> list[FitParam]:
        pardict = {p.name:p for p in parlist}
        pardict['xmax'].value = np.log(pardict['xmax'].value)
        pardict['xmax'].limits = (np.log(pardict['xmax'].limits[0]), np.log(pardict['xmax'].limits[1]))
        pardict['nmax'].value = np.log(pardict['nmax'].value)
        pardict['nmax'].limits = (np.log(pardict['nmax'].limits[0]), np.log(pardict['nmax'].limits[1]))
        return list(pardict.values())

class BasicParams:
    @staticmethod
    def get_event(parameters: np.ndarray) -> Event:
        '''This function maps the list of shower parameters to the Shower
        Event data container.
        '''
        nmax = parameters[1]
        return Event(
        E=E(nmax),
        Xmax=parameters[0],
        Nmax=nmax,
        zenith= parameters[2],
        azimuth= parameters[3],
        corex=parameters[4],
        corey=parameters[5],
        corez=parameters[6],
        X0=parameters[7],
        Lambda=parameters[8]
        )
    
    @staticmethod
    def get_params(evt: Event) -> np.ndarray:
        return np.array([evt.Xmax,
                        evt.Nmax,
                        evt.zenith,
                        evt.azimuth,
                        evt.corex,
                        evt.corey,
                        evt.corez,
                        evt.X0,
                        evt.Lambda])
    
    @staticmethod
    def adjust_guess(parlist: list[FitParam]) -> list[FitParam]:
        return parlist

def do_feature_fit(feature: FitFeature, guess_pars: list[FitParam]) -> Minuit:
    '''This function is a wrapper for the procedure of doing a fit with iminuit.
    '''
    ls = LeastSquares(np.arange(feature.output_size), 
                      feature.real_values, 
                      feature.error, 
                      feature.model,
                      verbose=1)
    m = Minuit(ls, 
               [par.value for par in guess_pars], 
               name = [par.name for par in guess_pars])
    for par in guess_pars:
        m.limits[par.name] = par.limits
        m.errors[par.name] = par.error
    m.fixed = True
    for parname in feature.target_parameters:
        m.fixed[parname] = False
    m.tol = .0001
    m.simplex()
    return m

def init_minuit(feature: FitFeature, guess_pars: list[FitParam]) -> Minuit:
    '''This function is a wrapper for the procedure of doing a fit with iminuit.
    '''
    ls = LeastSquares(np.arange(feature.output_size), 
                      feature.real_values, 
                      feature.error, 
                      feature.model,
                      verbose=1)
    m = Minuit(ls, 
               [par.value for par in guess_pars], 
               name = [par.name for par in guess_pars])
    for par in guess_pars:
        m.limits[par.name] = par.limits
        m.errors[par.name] = par.error
    m.fixed = True
    for parname in feature.target_parameters:
        m.fixed[parname] = False
    return m

def update_guess(m: Minuit) -> list[FitParam]:
    '''This function takes a minuit object and maps the parameters back to
    a list of my custom FitParam objects.
    '''
    return [FitParam(p.name,p.value,(p.lower_limit, p.upper_limit),p.error) for p in m.params]

def fit_procedure(ty: TyroFit, pf: NichePlane) -> Minuit:
    '''This function is the full procedure for fitting a NICHE event.
    '''

if __name__ == '__main__':
    # from datafiles import *
    import matplotlib.pyplot as plt
    plt.ion()
    from utils import plot_event, plot_generator, get_data_files, preceding_noise_file
    data_date_and_time = '20190504034237'
    data_files = get_data_files(data_date_and_time)
    noise_files = [preceding_noise_file(f) for f in data_files]
    cfg = CounterConfig(data_files, noise_files)
    
    pars = [500.,2.e6,np.deg2rad(40.),np.deg2rad(315.), 450., -660.,-25.7,0,70]
    ev = BasicParams.get_event(pars)
    pe = ProcessEvents(cfg, frozen_noise=False)
    
    xmax = []
    nmax = []
    zenith = []
    azimuth = []
    corex = []
    corey = []
    chi2=[]
    for i in range(100):
        real_nfits = pe.gen_nfits_from_event(ev)
        pf = NichePlane(real_nfits)
        ty = tyro(real_nfits)

        guess = make_guess(ty, pf)
        guess = LogXNParams.adjust_guess(guess)

        pt = PeakTimes(real_nfits, LogXNParams, cfg)
        pt.target_parameters = ['zenith','azimuth']
        m = init_minuit(pt, guess)
        m.tol = .0001
        m.simplex()
        tpguess = update_guess(m)

        pw = PulseWidth(real_nfits, LogXNParams, cfg)
        pw.target_parameters = ['xmax']
        m = init_minuit(pw, tpguess)
        m.simplex(ncall=5)
        xmaxguess = update_guess(m)

        pa = PulseArea(real_nfits, LogXNParams, cfg)
        pa.target_parameters = ['nmax']
        m = init_minuit(pa, xmaxguess)
        m.simplex(ncall=5)
        nmaxguess = update_guess(m)

        pa = NormalizedPulseArea(real_nfits, LogXNParams, cfg)
        pa.target_parameters = ['xmax','nmax','corex','corey']
        m = init_minuit(pa, nmaxguess)
        m.tol = .0001
        m.simplex()
        coreguess = update_guess(m)

        # m = init_minuit(pw, coreguess)
        # coreguess = update_guess(m)

        at = AllTunka(real_nfits, LogXNParams, cfg)
        at.target_parameters = ['xmax','nmax','corex','corey']
        m = init_minuit(at, coreguess)
        m.simplex()
        allguess = update_guess(m)

        # p = Peak(real_nfits, BasicParams, cfg)
        # m = init_minuit(p, allguess)

        fitpars = [p.value for p in m.params]
        e = LogXNParams.get_event(fitpars)
        xmax.append(e.Xmax)
        nmax.append(e.Nmax)
        zenith.append(e.zenith)
        azimuth.append(e.azimuth)
        corex.append(e.corex)
        corey.append(e.corey)
        chi2.append(at.chi2(fitpars))

    plt.figure()
    plt.hist(xmax,bins=50)
    plt.title('xmax')

    plt.figure()
    plt.hist(nmax,bins=50)
    plt.title('nmax')

    plt.figure()
    plt.hist(corex,bins=50)
    plt.title('corex')

    plt.figure()
    plt.hist(corey,bins=50)
    plt.title('corey')

    plt.figure()
    plt.hist(zenith,bins=50)
    plt.title('zenith')

    plt.figure()
    plt.hist(azimuth,bins=50)
    plt.title('azimuth')

    plt.figure()
    plt.hist(chi2,bins=50)
    plt.title('chi2')

    # ngrid = 11
    # real_nfits = pe.gen_nfits_from_event(ev)
    # pf = NichePlane(real_nfits)
    # ty = tyro(real_nfits)

    # guess_pars = pars.copy()
    # # guess_pars[0] = 500.
    # # guess_pars[1] = 2.e6
    # # guess_pars[2] = pf.theta
    # # guess_pars[3] = pf.phi
    # guess_pars[4] = ty.core_estimate[0]
    # guess_pars[5] = ty.core_estimate[1]

    # # xmaxs = norm.rvs(size=ngrid,loc=500.,scale=10.)
    # # nmaxs = norm.rvs(size=ngrid,loc=2.e6,scale=1.e5)
    # # xn = [[xm, nm,*guess_pars[2:]] for xm,nm in zip(xmaxs,nmaxs)]
    # xmaxs = np.linspace(450.,550.,ngrid)
    # nmaxs = np.linspace(1.e6,3.e6,ngrid)
    # x,n = np.meshgrid(xmaxs,nmaxs)
    # xn = [[xm, nm,*guess_pars[2:]] for xm,nm in zip(x.flatten(),n.flatten())]
    # p = NormalizedPulseArea(real_nfits, BasicParams, cfg)
    # xncosts = np.array(run_multiprocessing(p.chi2,xn,1))
    # plt.figure()
    # plt.contourf(xmaxs,nmaxs,xncosts.reshape(ngrid,ngrid),xncosts.min() + np.arange(50)**2,cmap='binary')
    # plt.colorbar()

    # xmaxs = np.linspace(450.,550.,ngrid)
    # nmaxs = np.linspace(1.e6,3.e6,ngrid)
    # x,n = np.meshgrid(xmaxs,nmaxs)
    # xn = [[xm, nm,*guess_pars[2:]] for xm,nm in zip(x.flatten(),n.flatten())]
    # pa = PulseArea(real_nfits, BasicParams, cfg)
    # xncosts = np.array(run_multiprocessing(pa.chi2,xn,1))
    # plt.figure()
    # plt.contourf(xmaxs,nmaxs,xncosts.reshape(ngrid,ngrid),xncosts.min() + np.arange(10)**2,cmap='binary')
    # plt.colorbar()

    # dpos = 15.
    # xs = np.linspace(pars[4]-dpos,pars[4]+dpos,ngrid)
    # ys = np.linspace(pars[5]-dpos,pars[5]+dpos,ngrid)
    # xc,yc = np.meshgrid(xs,ys)
    # xy = [[*guess_pars[:4],xm,ym,*guess_pars[-3:]] for xm,ym in zip(xc.flatten(),yc.flatten())]
    # npa = PulseArea(real_nfits, BasicParams, cfg)
    # xycosts = np.array(run_multiprocessing(npa.chi2,xy,1))
    # plt.figure()
    # plt.contourf(xs,ys,xycosts.reshape(ngrid,ngrid),xycosts.min() + np.arange(10)**2, cmap = 'binary')
    # plt.xlabel('corex')
    # plt.ylabel('corey')
    # plt.colorbar(label='chi_square')

    # dang = np.deg2rad(2.)
    # ts = np.linspace(pars[2]-np.deg2rad(1),pars[2]+np.deg2rad(1),ngrid)
    # ps = np.linspace(pars[3]-np.deg2rad(1),pars[3]+np.deg2rad(1),ngrid)
    # t,p = np.meshgrid(ts,ps)
    # tp = [[*guess_pars[:2],tm,pm,*guess_pars[4:]] for tm,pm in zip(t.flatten(),p.flatten())]
    # pt = PeakTimes(real_nfits, BasicParams, cfg)
    # tpcosts = np.array(run_multiprocessing(pt.chi2,tp,1))
    # plt.figure()
    # plt.contourf(np.rad2deg(ts),np.rad2deg(ps),tpcosts.reshape(ngrid,ngrid),tpcosts.min() + np.arange(10)**2,cmap='binary')
    # plt.xlabel('zenith')
    # plt.ylabel('azimuth')
    # plt.colorbar(label='chi_square')



    # parlist = [
    #     FitParam('xmax', 600., (300., 600.), 50., False),
    #     FitParam('nmax', 1.e7, (1.e5, 1.e7), 1.e5, False),
    #     FitParam('zenith', pars[2], (pf.theta -.1, pf.theta +.1), np.deg2rad(1.), True),
    #     FitParam('azimuth', pars[3], (pf.phi -.1, pf.phi +.1), np.deg2rad(1.), True),
    #     FitParam('corex',ty.core_estimate[0],(ty.core_estimate[0] - 50.,ty.core_estimate[0] + 50.), 5., True),
    #     FitParam('corey',ty.core_estimate[1],(ty.core_estimate[1] - 50.,ty.core_estimate[1] + 50.), 5., True),
    #     FitParam('corez',ty.core_estimate[2],(ty.core_estimate[2] - 1.,ty.core_estimate[2] + 1.), 1., True),
    #     FitParam('x0',0.,(0,100),1,True),
    #     FitParam('lambda',70., (0,100),1,True)
    # ]

    # ls = LeastSquares(np.arange(len(ef.nfits)),ef.pas,ef.pa_error,ef.pamodel,verbose=1)
    # guess_pars = [f.value for f in parlist]
    # names = [f.name for f in parlist]
    # m = Minuit(ls,guess_pars,name=names)

    # for f in parlist:
    #     m.limits[f.name] = f.limits
    #     m.errors[f.name] = f.error

    # m.fixed = True
    # m.fixed['xmax'] = False
    # m.fixed['nmax'] = False
    # m.fixed['corex'] = False
    # m.fixed['corey'] = False
    # m.tol = .001
    # m = m.simplex()

    # xmaxs = np.linspace(400.,600.,ngrid)
    # nmaxs = np.linspace(1.e6,3.e6,ngrid)
    # x,n = np.meshgrid(xmaxs,nmaxs)
    # xn = [[xm, nm,*guess_pars[2:]] for xm,nm in zip(x.flatten(),n.flatten())]
    # xncosts = np.array(run_multiprocessing(ef.chi_square,xn,1))
    # plt.figure()
    # plt.contourf(xmaxs,nmaxs,xncosts.reshape(ngrid,ngrid),xncosts.min() + np.arange(20)**2)
    # plt.xlabel('xmax')
    # plt.ylabel('nmax')
    # plt.semilogy()
    # plt.colorbar(label='chi_square')

    # xmax = []
    # nmax = []
    # zenith = []
    # azimuth = []
    # corex = []
    # corey = []
    # chi2=[]

    # for i in range(100):
    #     real_nfits = pe.gen_nfits_from_event(ev)
    #     ef = EventFit(real_nfits, LogXNCoreParams, cfg)
    #     pf = NichePlane(real_nfits)
    #     ty = tyro(real_nfits)


    #     parlist = [
    #         FitParam('xmax', np.log(300.), (np.log(300.), np.log(600.)), np.log(50.), False),
    #         FitParam('nmax+xmax', np.log(1.e5+300.), (np.log(1.e5+300.), np.log(1.e7+600.)), np.log(1.e5), False),
    #         FitParam('zenith', pf.theta, (pf.theta -.1, pf.theta +.1), np.deg2rad(1.), True),
    #         FitParam('azimuth', pf.phi, (pf.phi -.1, pf.phi +.1), np.deg2rad(1.), True),
    #         FitParam('corex',np.log(ty.core_estimate[0]),(np.log(ty.core_estimate[0] - 50.),np.log(ty.core_estimate[0] + 50.)), 5., True),
    #         FitParam('corey',np.log(-ty.core_estimate[1]),(np.log(-ty.core_estimate[1] - 50.),np.log(-ty.core_estimate[1] + 50.)), 5., True),
    #         FitParam('corez',ty.core_estimate[2],(ty.core_estimate[2] - 1.,ty.core_estimate[2] + 1.), 1., True),
    #         FitParam('x0',0.,(0,100),1,True),
    #         FitParam('lambda',70., (0,100),1,True)
    #     ]

    #     ls = LeastSquares(np.arange(len(ef.nfits)),ef.real_pulse_widths,ef.real_pulse_width_error,ef.pwmodel,verbose=1)
    #     guess_pars = [f.value for f in parlist]
    #     names = [f.name for f in parlist]
    #     m = Minuit(ls,guess_pars,name=names)

    #     for f in parlist:
    #         m.limits[f.name] = f.limits
    #         m.errors[f.name] = f.error

    #     m.fixed = True
    #     m.fixed['zenith'] = False
    #     m.fixed['azimuth'] = False
    #     m = m.simplex()

    #     m.fixed = True
    #     m.fixed['xmax'] = False
    #     m.fixed['nmax+xmax'] = False
    #     m.fixed['corex'] = False
    #     m.fixed['corey'] = False
    #     # m.fixed['corez'] = False
    #     m.tol=(.001)
    #     m = m.simplex()

    #     m.fixed = True
    #     m.fixed['xmax'] = False
    #     m.fixed['nmax+xmax'] = False

    #     # m.values['xmax'] = np.log(600.)
    #     # m.values['nmax+xmax'] = np.log(1.e7 + 600.)
    #     m = m.simplex()

    #     m.fixed = True
    #     # m.fixed['xmax'] = False
    #     # m.fixed['nmax+xmax'] = False
    #     m.fixed['corex'] = False
    #     m.fixed['corey'] = False
    #     # m.fixed['corez'] = False
    #     m.tol=(.001)
    #     m = m.simplex()

    #     m.fixed = True
    #     m.fixed['xmax'] = False
    #     m.fixed['nmax+xmax'] = False
    #     m.fixed['corex'] = False
    #     m.fixed['corey'] = False
    #     # m.fixed['zenith'] = False
    #     # m.fixed['azimuth'] = False
    #     # m.fixed['corez'] = False
    #     m.tol=(.001)
    #     m = m.simplex()


    #     pars = np.array([par.value for par in m.params])
    #     e = get_event(pars)
    #     xmax.append(e.Xmax)
    #     nmax.append(e.Nmax)
    #     zenith.append(e.zenith)
    #     azimuth.append(e.azimuth)
    #     corex.append(e.corex)
    #     corey.append(e.corey)
    #     chi2.append(ef.chi_square(pars))

    # plt.figure()
    # plt.hist(xmax,bins=50)
    # plt.title('xmax')

    # plt.figure()
    # plt.hist(nmax,bins=50)
    # plt.title('nmax')

    # plt.figure()
    # plt.hist(corex,bins=50)
    # plt.title('corex')

    # plt.figure()
    # plt.hist(corey,bins=50)
    # plt.title('corey')

    # plt.figure()
    # plt.hist(zenith,bins=50)
    # plt.title('zenith')

    # plt.figure()
    # plt.hist(azimuth,bins=50)
    # plt.title('azimuth')

    # plt.figure()
    # plt.hist(chi2,bins=50)
    # plt.title('chi2')




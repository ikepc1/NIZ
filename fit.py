from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from functools import cached_property
from typing import Protocol, Callable
from abc import ABC, abstractmethod
import sys

from utils import run_multiprocessing
from tyro_fit import tyro, TyroFit
from niche_plane import NichePlane, NicheFit
from gen_ckv_signals import Event, get_ckv, CherenkovOutput, GetCkv
from process_showers import ProcessEvents
from config import COUNTER_NO, NAMES, COUNTER_POSITIONS, WAVEFORM_SIZE, TRIGGER_POSITION, NICHE_TIMEBIN_SIZE, COUNTER_QE, COUNTER_FADC_PER_PE
from counter_config import CounterConfig
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

def ckv_signal_dict(ckv: CherenkovOutput, t_offset: float = 0.) -> tuple[dict[str,np.ndarray],np.ndarray]:
    '''This function compiles the waveforms from CHASM into a dictionary where
    the keys are the counter names.
    '''
    fadc, times = rawphotons2fadc(ckv, t_offset)
    return {name:f for name, f in zip(ckv.cfg.active_counters, fadc)}, times

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

def integrate_from_fit(wf: np.ndarray, pbs: np.ndarray) -> float:
    intstart = int(np.floor(pbs[0] - 5.*pbs[2]))
    intend   = int(np.ceil(pbs[0] + 5.*pbs[3]))
    intsignal = wf[intstart:intend+1].sum()
    return intsignal

class ParamMapper(Protocol):
    '''This protocol maps a given set of parameters to an actual shower
    event.
    '''

    def get_event(self, pars: np.ndarray) -> Event:
        ...

    def get_params(self, evt: Event) -> np.ndarray:
        ...

    def adjust_guess(self):
        ...

class FitFeature(ABC):
    '''This is the protocol for a fit feature.
    '''
    def __init__(self, nfits: list[NicheFit], param_mapper: ParamMapper, cfg: CounterConfig) -> None:
        self.nfits = nfits
        self.param_mapper = param_mapper
        self.cfg = cfg
        # self.ckv = GetCkv(cfg)
        self.target_parameters = []

    def ckv_from_params(self, parameters: np.ndarray) -> CherenkovOutput:
        '''This method returns the Cherenkov light in each counter at each time bin.
        '''
        ev = self.param_mapper.get_event(parameters)
        # print(ev)
        ckv = get_ckv(ev, self.cfg)
        # ckv = self.ckv.run(ev)
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
    
    @cached_property
    def peaktimes(self) -> np.ndarray:
        return np.array([self.adjust_data_peaktime(f) for f in self.nfits])

    @property
    def nfit_dict(self) -> dict[str, NicheFit]:
        '''This is a dictionary of the Nichefit objects in the real event.
        '''
        return {f.name:f for f in self.nfits}

    def model(self, input_indices: np.ndarray, *parameters) -> np.ndarray:
        '''This is the model to be supplied to minuit. The indices are ignored.
        '''
        parameters = np.array(parameters)
        return self.get_output(parameters)

    def chi2(self, parameters: np.ndarray) -> float:
        '''This method calculates the chi squared value for a run with the 
        parameters specified.
        '''
        output = self.get_output(parameters)
        return ((self.real_values - output)**2/self.error**2).sum()
    
    def lnlike(self, parameters: np.ndarray) -> float:
        return -.5 * self.chi2(parameters)

    @property
    @abstractmethod
    def real_values() -> np.ndarray:
        '''This property should return the values of the feature in the real data.
        '''

    @property
    @abstractmethod
    def real_inputs() -> np.ndarray:
        '''This property should return the inputs to the model.
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

    @abstractmethod
    def get_output(self, parameters: np.ndarray) -> np.ndarray:
        '''This method should return the simulated feature when run with the given
        parameters.
        '''

    def cost(self) -> LeastSquares:
        return LeastSquares(self.real_inputs, 
                            self.real_values, 
                            self.error, 
                            self.model,
                            verbose=0)

class PeakTimes(FitFeature):
    '''This is the container for the peaktimes fit feature.
    '''

    @property
    def real_values(self) -> np.ndarray:
        return self.peaktimes
    
    @property
    def error(self) -> np.ndarray:
        return np.array([NICHE_TIMEBIN_SIZE * f.epeaktime for f in self.nfits])
    
    @property
    def output_size(self) -> int:
        return len(self.nfits)
    
    @property
    def real_inputs(self) -> np.ndarray:
        return np.arange(self.output_size)
    
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
    
class OffsetPeakTimes(PeakTimes):
    def get_output(self, parameters: np.ndarray) -> np.ndarray:
        ckv = self.ckv_from_params(parameters)
        sigdict, times = ckv_signal_dict(ckv,parameters[-1])

        #do tunka fits
        pb_array = np.array([do_wf_fit(sigdict[name])[:-1] for name in self.nfit_dict])

        #tunka fit returns approximate index of peak, find times those corresponds to
        peaktimes = np.interp(pb_array[:,0], np.arange(len(times)), times)
        
        #adjust times so biggest counter peaktime is the start
        # peaktimes -= peaktimes[self.pas.argmax()]
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
    
    @property
    def real_inputs(self) -> np.ndarray:
        return np.arange(self.output_size)

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
    
    @property
    def real_inputs(self) -> np.ndarray:
        return np.arange(self.output_size)

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
        sigdict, times = ckv_signal_dict(ckv,parameters[-1])
        # sigdict, times = ckv_signal_dict(ckv)

        #do tunka fits
        pb_array = np.array([do_wf_fit(sigdict[name])[:-1] for name in self.nfit_dict])

        #tunka fit returns approximate index of peak, find times those corresponds to
        peaktimes = np.interp(pb_array[:,0], np.arange(len(times)), times)
        
        #adjust times so biggest counter peaktime is the start
        # peaktimes -= peaktimes[self.pas.argmax()]
        pb_array[:,0] = peaktimes
        return pb_array.flatten()
    
    @property
    def output_size(self) -> int:
        return 4 * len(self.nfits)
    
    @property
    def real_inputs(self) -> np.ndarray:
        return np.arange(self.output_size)

class MeasuredPulse(FitFeature):
    '''This is the container for all tunka parameters as features simultaneously.
    '''
    @staticmethod
    def measured_ns_diff(nfit: NicheFit) -> float:
        return (nfit.waveform.argmax() - TRIGGER_POSITION) * NICHE_TIMEBIN_SIZE

    @property
    def biggest_measured_difference(self) -> float:
        return self.measured_ns_diff(self.nfit_dict[self.biggest_counter])

    def adjust_measured_peaktime(self, nfit: NicheFit) -> float:
        '''This method calculates the peaktime of the 'nfit' counter relative to the peaktime of 
        the largest trigger.
        '''
        trigtime_delta = nfit.trigtime() - self.biggest_trigtime
        ns_diff = self.measured_ns_diff(nfit)
        peaktime_difference = trigtime_delta.astype('float64') + ns_diff - self.biggest_measured_difference
        return peaktime_difference
    
    @staticmethod
    def measured_peak(nfit: NicheFit) -> float:
        return nfit.waveform.max() - nfit.baseline
    
    @staticmethod
    def fwhm(wf: np.ndarray) -> float:
        hm = wf.max() / 2.
        diffs = np.abs(wf - hm)

        #find time of rising half-max
        closest_ib4 = diffs[:wf.argmax()].argmin()
        if wf[closest_ib4] < hm:
            before_i = closest_ib4
            after_i = closest_ib4 + 1
        else:
            before_i = closest_ib4 - 1
            after_i = closest_ib4
        tmax1 = np.interp(hm,[wf[before_i],wf[after_i]],[before_i,after_i])

        #find time of falling half-max
        closest_iafter = diffs[wf.argmax():].argmin() + wf.argmax()
        if wf[closest_iafter] > hm:
            before_i = closest_iafter
            after_i = closest_iafter + 1
        else:
            before_i = closest_iafter - 1
            after_i = closest_iafter
        tmax2 = np.interp(hm,[wf[before_i],wf[after_i]],[before_i,after_i])
        return tmax2 - tmax1

    @property
    def real_values(self) -> np.ndarray:
        '''This is the fitted parameters for each counter triggered in the event. 
        They are the output data points to be compared to the model.
        '''
        pars = [np.array([self.adjust_measured_peaktime(f), self.measured_peak(f), self.fwhm(f.waveform - f.baseline)]) for f in self.nfits]
        return np.hstack(pars)
    
    @property
    def error(self) -> np.ndarray:
        '''This is the fitted parameters for each counter triggered in the event. 
        They are the output data points to be compared to the model.
        '''
        errs = [np.array([.5 * NICHE_TIMEBIN_SIZE, f.baseline_error, np.sqrt(2)]) for f in self.nfits]
        return np.hstack(errs)

    def get_output(self, parameters: np.ndarray) -> np.ndarray:
        '''This method gives the data output for a set of shower parameters.
        '''
        ckv = self.ckv_from_params(parameters)
        sigdict, times = ckv_signal_dict(ckv,parameters[-1])

        #find waveform properties in simulated data
        pb_array = np.empty((len(self.nfits), 3))
        pb_array[:,0] = np.array([sigdict[name].argmax() for name in self.nfit_dict])
        pb_array[:,1] = np.array([sigdict[name].max() for name in self.nfit_dict])
        pb_array[:,2] = np.array([self.fwhm(sigdict[name]) for name in self.nfit_dict])

        #tunka fit returns approximate index of peak, find times those corresponds to
        peaktimes = np.interp(pb_array[:,0], np.arange(len(times)), times)
        
        #adjust times so biggest counter peaktime is the start
        # peaktimes -= peaktimes[self.pas.argmax()]
        pb_array[:,0] = peaktimes
        return pb_array.flatten()
    
    @property
    def output_size(self) -> int:
        return 3 * len(self.nfits)
    
    @property
    def real_inputs(self) -> np.ndarray:
        return np.arange(self.output_size)
    
# def fwhm(wf: np.ndarray) -> float:
#     hm = wf.max() / 2.

def fwhm(wf: np.ndarray) -> float:
    hm = wf.max() / 2.
    diffs = np.abs(wf - hm)

    #find time of rising half-max
    closest_ib4 = diffs[:wf.argmax()].argmin()
    if wf[closest_ib4] < hm:
        before_i = closest_ib4
        after_i = closest_ib4 + 1
    else:
        before_i = closest_ib4 - 1
        after_i = closest_ib4
    tmax1 = np.interp(hm,[wf[before_i],wf[after_i]],[before_i,after_i])

    #find time of falling half-max
    closest_iafter = diffs[wf.argmax():].argmin() + wf.argmax()
    if wf[closest_iafter] > hm:
        before_i = closest_iafter
        after_i = closest_iafter + 1
    else:
        before_i = closest_iafter - 1
        after_i = closest_iafter
    tmax2 = np.interp(hm,[wf[before_i],wf[after_i]],[before_i,after_i])
    return tmax2 - tmax1

class FWHM(FitFeature):
    @cached_property
    def real_values(self) -> np.ndarray:
        return np.array([fwhm(f.waveform - f.baseline) for f in self.nfits])
    
    @cached_property
    def error(self) -> np.ndarray:
        return np.full(len(self.nfits),np.sqrt(2))
    
    def get_output(self, parameters: np.ndarray) -> np.ndarray:
        '''This method gives the data output for a set of shower parameters.
        '''
        ckv = self.ckv_from_params(parameters)
        sigdict, times = ckv_signal_dict(ckv,parameters[-1])
        fwhms = np.array([fwhm(sigdict[name]) for name in self.nfit_dict])
        return fwhms
    
    @property
    def output_size(self) -> int:
        return len(self.nfits)
    
    @property
    def real_inputs(self) -> np.ndarray:
        return np.arange(self.output_size)


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
    
    @property
    def real_inputs(self) -> np.ndarray:
        return np.arange(self.output_size)

class NormalizedPulseArea(PulseArea):
    '''This is the implementation of normalized pulse areas fit feature.
    '''
    @property
    def output_size(self) -> int:
        return len(self.nfits)
    
    @property
    def real_inputs(self) -> np.ndarray:
        return np.arange(self.output_size)

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

class TimesWidthsAreas(FitFeature):

    def __init__(self, nfits: list[NicheFit], param_mapper: ParamMapper, cfg: CounterConfig) -> None:
        super().__init__(nfits, param_mapper, cfg)
        self.peaktimes = PeakTimes(nfits, param_mapper, cfg)
        self.pulse_widths = PulseWidth(nfits, param_mapper, cfg)
        self.pulse_areas = NormalizedPulseArea(nfits, param_mapper, cfg)

    @property
    def output_size(self) -> int:
        return 3*len(self.nfits)
    
    @property
    def real_inputs(self) -> np.ndarray:
        return np.arange(self.output_size)
    
    @cached_property
    def real_values(self) -> np.ndarray:
        return np.hstack((self.peaktimes.real_values, self.pulse_widths.real_values, self.pulse_areas.real_values))
    
    @cached_property
    def error(self) -> np.ndarray:
        return np.hstack((self.peaktimes.error, self.pulse_widths.error, self.pulse_areas.error))
    
    def get_output(self, parameters: np.ndarray) -> np.ndarray:
        ckv = self.ckv_from_params(parameters)
        sigdict, times = ckv_signal_dict(ckv,parameters[-1])
        #do tunka fits
        pb_array = np.array([do_wf_fit(sigdict[name])[:-1] for name in self.nfit_dict])
        #tunka fit returns approximate index of peak, find times those corresponds to
        peaktimes = np.interp(pb_array[:,0], np.arange(len(times)), times)
        #adjust times so biggest counter peaktime is the start
        # peaktimes -= peaktimes[self.pas.argmax()]
        # pa_array = np.array([do_pulse_integration(sigdict[name]) for name in self.nfit_dict])
        pa_array = np.array([integrate_from_fit(sigdict[name],pbs) for name, pbs in zip(self.nfit_dict,pb_array)])
        pulse_widths = pb_array[:,2] + pb_array[:,3]
        return np.hstack((peaktimes,pulse_widths,pa_array/pa_array.sum()))

class AllSamples(FitFeature):
    wf_cushion = 4
    wf_start = TRIGGER_POSITION - wf_cushion
    wf_end = TRIGGER_POSITION + wf_cushion

    @cached_property
    def biggest_trig2peak_diff(self) -> float:
        return (self.nfit_dict[self.biggest_counter].waveform.argmax() - TRIGGER_POSITION) * NICHE_TIMEBIN_SIZE

    def data_trigtime2peaktime(self, nfit: NicheFit) -> float:
        '''This method calculates the peaktime of the 'nfit' counter relative to the peaktime of 
        the largest trigger.
        '''
        trigtime_delta = nfit.trigtime() - self.biggest_trigtime
        ns_diff = (nfit.waveform.argmax() - TRIGGER_POSITION) * NICHE_TIMEBIN_SIZE
        return trigtime_delta.astype('float64') + ns_diff - self.biggest_trig2peak_diff
    
    def get_real_times(self, fit: NicheFit) -> np.ndarray:
        '''This method calculates the time values for each waveform sample in a real data NicheFit.
        '''
        times = np.arange(NICHE_TIMEBIN_SIZE*WAVEFORM_SIZE, step = NICHE_TIMEBIN_SIZE)
        # times = np.arange(0.,float(WAVEFORM_SIZE)) * NICHE_TIMEBIN_SIZE
        peaktime = self.data_trigtime2peaktime(fit)
        times -= times[fit.waveform.argmax()]
        times += peaktime
        return times

    @cached_property
    def real_times_array(self) -> list[np.ndarray]:
        ''''''
        return [self.get_real_times(f)[f.start_rise:f.end_fall] for f in self.nfits]

    @cached_property
    def real_inputs(self) -> np.ndarray:
        # return self.real_times_array.flatten()
        return np.hstack(self.real_times_array)
        # return np.arange(len(self.real_times_array.flatten()))
    
    @cached_property
    def real_values(self) -> np.ndarray:
        return np.hstack([(f.waveform - f.baseline)[f.start_rise:f.end_fall] for f in self.nfits])
    
    def calc_sample_errors(self, f: NicheFit) -> np.ndarray:
        wf = (f.waveform - f.baseline)[f.start_rise:f.end_fall]
        wf[wf<0.] = 0.
        sqrtn_error = np.sqrt(wf) 
        return np.sqrt(sqrtn_error**2 + f.baseline_error**2)
    
    @cached_property
    def error(self) -> np.ndarray:
        # return np.hstack([self.calc_sample_errors(f) for f in self.nfits])
        return np.hstack([np.full(f.end_fall-f.start_rise,f.baseline_error) for f in self.nfits])
    
    @staticmethod
    def trace_at_times(sim_wf: np.ndarray, sim_times: np.ndarray, real_trace_times: np.ndarray) -> np.ndarray:
        return sim_wf[np.searchsorted(sim_times,real_trace_times, side='left')]
        # return np.interp(real_trace_times,sim_times,sim_wf, left=0., right=0.)

    @staticmethod
    def wf_baseline_parabola(wf: np.ndarray, tol:float=1.e-6) -> np.ndarray:
        wf_start = np.argmax(wf[wf<1.e-6])
        wf_rev = wf[::-1]
        wf_end = np.argmax(wf_rev[wf_rev<1.e-6])
        if wf_end == 0 or wf_start == 0:
            return wf
        wf[:wf_start] -= np.arange(wf_start)[::-1]**2
        wf[-wf_end:] -= np.arange(wf_end)**2
        return wf

    def get_output(self, parameters: np.ndarray) -> np.ndarray:
        ckv = self.ckv_from_params(parameters)
        sigdict, times = ckv_signal_dict(ckv, parameters[-1])
        # sigdict, times = ckv_signal_dict(ckv, 0.)
        for c in self.nfit_dict:
            sigdict[c][sigdict[c]>self.nfit_dict[c].max_value] = self.nfit_dict[c].max_value
        # sigdict, times = ckv_signal_dict(ckv)
        # sigdict = {name:self.wf_baseline_parabola(sigdict[name]) for name in sigdict}
        # times -= times[sigdict[self.biggest_counter].argmax()]
        wfs_at_real_times = [self.trace_at_times(sigdict[name],times,realtimes) for name, realtimes in zip(self.nfit_dict,self.real_times_array)]
        # wfs_at_real_times[wfs_at_real_times<0.] = 0.
        return np.hstack(wfs_at_real_times)
    
class Samples(FitFeature):
    @cached_property
    def real_inputs(self) -> np.ndarray:
        return np.arange(100*len(self.nfits))
    
    def get_pulse(self, full_wf: np.ndarray) -> np.ndarray:
        max = full_wf.argmax()
        return full_wf[max-50:max+50]

    @cached_property
    def real_values(self) -> np.ndarray:
        return np.array([self.get_pulse(f.waveform-f.baseline) for f in self.nfits]).flatten()
    
    @cached_property
    def error(self) -> np.ndarray:
        return np.array([np.full(100,f.baseline_error) for f in self.nfits]).flatten()

    def get_output(self, parameters: np.ndarray) -> np.ndarray:
        ckv = self.ckv_from_params(parameters)
        # sigdict, times = ckv_signal_dict(ckv, parameters[-1])
        sigdict, times = ckv_signal_dict(ckv)
        wfs = np.array([self.get_pulse(sigdict[name]) for name in self.nfit_dict])
        wfs[wfs<0.] = 0.
        return wfs.flatten()

@dataclass
class FitParam:
    '''This is a wrapper for a minuit parameter for easy management.
    '''
    name: str
    value: float
    limits: tuple[float,float]
    error: float
    fixed: bool = False

    def parguess(self, size: int) -> np.ndarray:
        return np.random.normal(self.value, .1*self.error, size = size)

def make_guess(ty: TyroFit, pf: NichePlane, cfg: CounterConfig) -> list[FitParam]:
    '''This function makes a guess for the fit parameters.
    '''
    corez = cfg.counter_bottom[2]
    parlist = [
        FitParam('xmax', 500., (400., 1000.), 50.),
        FitParam('nmax', 1.e6, (1.e4, 1.e7), 1.e5),
        FitParam('zenith', pf.theta, (0., pf.theta +.1), np.deg2rad(1.)),
        FitParam('azimuth', pf.phi, (pf.phi -.1, pf.phi +.1), np.deg2rad(1.)),
        FitParam('corex',ty.core_estimate[0],ty.xlimits, 5.),
        FitParam('corey',ty.core_estimate[1],ty.ylimits, 5.),
        # FitParam('corez',ty.core_estimate[2],(ty.core_estimate[2] - 1.,ty.core_estimate[2] + 1.), 1.),
        # FitParam('corez',-25.7,(ty.core_estimate[2] - 1.,ty.core_estimate[2] + 1.), 1.),
        FitParam('corez',corez,(corez,corez), 1., fixed=True),
        FitParam('x0',0.,(-1.e3,1.e3),1, fixed=True),
        FitParam('lambda',70., (40.,100.),1, fixed=True),
        FitParam('t_offset', 0., (-4.5e2, 4.5e2), 10.,fixed=False)
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
    # ls = LeastSquares(feature.real_inputs, 
    #                   feature.real_values, 
    #                   feature.error, 
    #                   feature.model,
    #                   verbose=1)
    m = Minuit(feature.cost(), 
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
    # ls = LeastSquares(np.arange(feature.output_size), 
    #                   feature.real_values, 
    #                   feature.error, 
    #                   feature.model,
    #                   verbose=0)
    m = Minuit(feature.cost(), 
               *[par.value for par in guess_pars], 
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

def update_guess_values(guess: list[FitParam], m: Minuit) -> list[FitParam]:
    '''This function takes a minuit object and maps the parameters back to
    a list of my custom FitParam objects.
    '''
    return [FitParam(p.name,p.value,(p.lower_limit, p.upper_limit),gp.error,gp.fixed) for p, gp in zip(m.params,guess)]

class FitProcedure:

    def __init__(self, cfg: CounterConfig, nfits: list[NicheFit]) -> None:
        self.cfg = cfg
        self.nfits = nfits
        self.chi2ndof = 1.e10
        

    def fit_procedure(self, guess: list[FitParam]) -> list[FitParam]:
        '''This function is the full procedure for fitting a NICHE event.
        '''

        pt = PeakTimes(self.nfits, BasicParams, self.cfg)
        pt.target_parameters = ['zenith','azimuth']
        m = init_minuit(pt, guess)
        m.tol = .1
        m.simplex()

        # guess = update_guess_values(guess, m)
        # pw = PulseWidth(self.nfits, BasicParams, self.cfg)
        # pw.target_parameters = ['xmax']
        # m = init_minuit(pw, guess)
        # m.simplex(ncall=40)

        # guess = update_guess_values(guess, m)
        # pa = PulseArea(self.nfits, BasicParams, self.cfg)
        # pa.target_parameters = ['nmax']
        # m = init_minuit(pa, guess)
        # m.simplex(ncall=20)

        guess = update_guess_values(guess, m)
        pa = NormalizedPulseArea(self.nfits, BasicParams, self.cfg)
        pa.target_parameters = ['xmax','nmax','corex','corey']
        m = init_minuit(pa, guess)
        m.simplex()

        guess = update_guess(m)
        at = AllTunka(self.nfits, BasicParams, self.cfg)
        at.target_parameters = ['t_offset']
        m = init_minuit(at, guess)
        m.migrad()

        guess = update_guess(m)
        at = AllSamples(self.nfits, BasicParams, self.cfg)
        at.target_parameters = ['t_offset']
        m = init_minuit(at, guess)

        m.tol=.1
        m.fixed = True
        m.fixed['xmax'] = False
        m.fixed['nmax'] = False
        m.fixed['zenith'] = False
        m.fixed['azimuth'] = False
        m.fixed['corex'] = False
        m.fixed['corey'] = False
        m.fixed['x0'] = False
        m.fixed['lambda'] = False
        m.fixed['t_offset'] = False
        m.simplex()

        self.chi2ndof = at.chi2(np.array([p.value for p in m.params]))/m.ndof

        guess = update_guess_values(guess,m)
        return guess
    
    def minimal_fit_procedure(self, guess: list[FitParam]) -> list[FitParam]:
        '''This function is the minimal procedure for fitting a NICHE event.
        '''

        pt = PeakTimes(self.nfits, BasicParams, self.cfg)
        pt.target_parameters = ['zenith','azimuth']
        m = init_minuit(pt, guess)
        m.tol = .1
        m.simplex()

        # guess = update_guess_values(guess, m)
        # pw = PulseWidth(self.nfits, BasicParams, self.cfg)
        # pw.target_parameters = ['xmax']
        # m = init_minuit(pw, guess)
        # m.simplex(ncall=40)

        # guess = update_guess_values(guess, m)
        # pa = PulseArea(self.nfits, BasicParams, self.cfg)
        # pa.target_parameters = ['nmax']
        # m = init_minuit(pa, guess)
        # m.simplex(ncall=20)

        guess = update_guess_values(guess, m)
        pa = NormalizedPulseArea(self.nfits, BasicParams, self.cfg)
        pa.target_parameters = ['xmax','nmax','corex','corey']
        m = init_minuit(pa, guess)
        m.simplex()

        self.chi2ndof = pa.chi2(np.array([p.value for p in m.params]))/m.ndof

        guess = update_guess_values(guess,m)
        return guess
    
def dataframe_fit()

if __name__ == '__main__':
    ev_df_pkl = Path(sys.argv[1])






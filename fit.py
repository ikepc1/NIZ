from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.optimize import curve_fit
import numpy as np
from dataclasses import dataclass, field
import pandas as pd
from functools import cached_property

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
    corex=np.exp(parameters[4]),
    corey=-np.exp(parameters[5]),
    corez=parameters[6],
    X0=parameters[7],
    Lambda=parameters[8]
    )

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

@dataclass
class EventFit:
    '''This class is responible for compiling the features to be fit from
    real data.
    '''
    nfits: list[NicheFit]
    cfg: CounterConfig = field(repr=False)
    n_features: int = 4

    @property
    def pas(self) -> np.ndarray:
        return np.array([f.intsignal for f in self.nfits])

    @property
    def nfit_dict(self) -> dict[str, NicheFit]:
        '''This is a dictionary of the Nichefit objects in the real event.
        '''
        return {f.name:f for f in self.nfits}

    @property
    def biggest_counter(self) -> str:
        '''This is the name of the counter in the data with the biggest
        pulse.
        '''
        return self.nfits[self.pas.argmax()].name
    
    @cached_property
    def biggest_trigtime(self) -> np.datetime64:
        '''This is the time when the biggest trigger happened.
        '''
        return self.nfit_dict[self.biggest_counter].trigtime()
    
    @cached_property
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
    def pa_output(self) -> np.ndarray:
        '''This property is the pulse area seen in each real counter.
        '''
        return np.array([f.intsignal for f in self.nfits])
    
    @property
    def pa_error(self) -> np.ndarray:
        return np.array([f.eintsignal for f in self.nfits])
    
    def ckv_from_params(self, parameters: np.ndarray) -> np.ndarray:
        ev = get_event(parameters)
        print(ev)
        ckv = get_ckv(ev, self.cfg)
        return ckv

    def get_pa_output(self, parameters: np.ndarray) -> np.ndarray:
        '''This method gets the pulse area from a sim.
        '''
        ckv = self.ckv_from_params(parameters)
        sigdict, _ = ckv_signal_dict(ckv)
        pb_array = np.array([do_pulse_integration(sigdict[name]) for name in self.nfit_dict])
        return pb_array

    @cached_property
    def real_output(self) -> np.ndarray:
        '''This is the fitted parameters for each counter triggered in the event. 
        They are the output data points to be compared to the model.
        '''
        pars = [np.array([self.adjust_data_peaktime(f), f.peak, f.risetime, f.falltime]) for f in self.nfits]
        return np.hstack(pars)
    
    @cached_property
    def real_error(self) -> np.ndarray:
        '''This is the fitted parameters for each counter triggered in the event. 
        They are the output data points to be compared to the model.
        '''
        errs = [np.array([f.epeaktime * NICHE_TIMEBIN_SIZE, f.epeak, f.erisetime, f.efalltime]) for f in self.nfits]
        return np.hstack(errs)
    
    @property
    def input_indices(self) -> np.ndarray:
        '''This is the enumeration of the terms of the chi-squared statistic.
        '''
        return np.arange(self.n_features * len(self.nfits))

    def get_output(self, parameters: np.ndarray) -> np.ndarray:
        '''This method gives the data output for a set of shower parameters.
        '''
        ev = get_event(parameters)
        print(ev)
        ckv = get_ckv(ev, self.cfg)
        sigdict, times = ckv_signal_dict(ckv)

        #do tunka fits
        pb_array = np.array([do_wf_fit(sigdict[name])[:-1] for name in self.nfit_dict])

        #tunka fit returns approximate index of peak, find times those corresponds to
        peaktimes = np.interp(pb_array[:,0], np.arange(len(times)), times)
        
        #adjust times so biggest counter peaktime is the start
        peaktimes -= peaktimes[self.pas.argmax()]
        pb_array[:,0] = peaktimes
        return pb_array.flatten()

    def chi_square(self, parameters: np.ndarray) -> float:
        '''This is a direct calculation of the chi square statistic for a set of shower 
        parameters.
        '''
        output = self.get_output(parameters)
        return ((self.real_output - output)**2/self.real_error**2).sum()

    def model(self, chi_square_indices: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        '''This is the model to be supplied to minuit. The indices are ignored.
        '''
        return self.get_output(parameters)

@dataclass
class FitParam:
    '''This is a wrapper for a minuit parameter for easy management.
    '''
    name: str
    value: float
    limits: tuple[float]
    error: float
    fixed: bool

if __name__ == '__main__':
    # from datafiles import *
    import matplotlib.pyplot as plt
    plt.ion()
    from utils import plot_event, plot_generator, get_data_files, preceding_noise_file
    data_date_and_time = '20190504034237'
    data_files = get_data_files(data_date_and_time)
    noise_files = [preceding_noise_file(f) for f in data_files]
    cfg = CounterConfig(data_files, noise_files)
    
    pars = [np.log(500),np.log(2.e6),np.deg2rad(40.),np.deg2rad(315.), np.log(450.), np.log(660.),-25.,0,70]
    ev = get_event(pars)
    pe = ProcessEvents(cfg, frozen_noise=False)

    real_nfits = pe.gen_nfits_from_event(ev)
    

    xmax = []
    nmax = []
    zenith = []
    azimuth = []
    corex = []
    corey = []
    chi2=[]

    for i in range(1):
        real_nfits = pe.gen_nfits_from_event(ev)
        ef = EventFit(real_nfits, cfg)
        pf = NichePlane(real_nfits)
        ty = tyro(real_nfits)


        parlist = [
            FitParam('xmax', np.log(300.), (np.log(300.), np.log(600.)), np.log(50.), False),
            FitParam('nmax', np.log(1.e5), (np.log(1.e5), np.log(1.e7)), np.log(1.e5), False),
            FitParam('zenith', pf.theta, (pf.theta -.1, pf.theta +.1), np.deg2rad(1.), True),
            FitParam('azimuth', pf.phi, (pf.phi -.1, pf.phi +.1), np.deg2rad(1.), True),
            FitParam('corex',np.log(ty.core_estimate[0]),(np.log(ty.core_estimate[0] - 50.),np.log(ty.core_estimate[0] + 50.)), 5., True),
            FitParam('corey',np.log(-ty.core_estimate[1]),(np.log(-ty.core_estimate[1] - 50.),np.log(-ty.core_estimate[1] + 50.)), 5., True),
            FitParam('corez',ty.core_estimate[2],(ty.core_estimate[2] - 1.,ty.core_estimate[2] + 1.), 1., True),
            FitParam('x0',0.,(0,100),1,True),
            FitParam('lambda',70., (0,100),1,True)
        ]

        ls = LeastSquares(ef.input_indices,ef.real_output,ef.real_error,ef.model,verbose=1)
        guess_pars = [f.value for f in parlist]
        names = [f.name for f in parlist]
        m = Minuit(ls,guess_pars,name=names)

        for f in parlist:
            m.limits[f.name] = f.limits
            m.errors[f.name] = f.error

        m.fixed = True
        m.fixed['zenith'] = False
        m.fixed['azimuth'] = False
        m = m.simplex()

        m.fixed = True
        m.fixed['xmax'] = False
        m.fixed['nmax'] = False
        m.fixed['corex'] = False
        m.fixed['corey'] = False
        # m.fixed['corez'] = False
        m.tol=(.001)
        m = m.simplex()

        m.fixed = True
        m.fixed['xmax'] = False
        m.fixed['nmax'] = False

        m.values['xmax'] = np.log(600)
        m.values['nmax'] = np.log(1.e7)
        m = m.simplex()

        m.fixed = True
        m.fixed['xmax'] = False
        m.fixed['nmax'] = False
        m.fixed['corex'] = False
        m.fixed['corey'] = False
        # m.fixed['corez'] = False
        m.tol=(.001)
        m = m.simplex()

        pars = np.array([par.value for par in m.params])
        xmax.append(np.exp(pars[0]))
        nmax.append(np.exp(pars[1]))
        zenith.append(pars[2])
        azimuth.append(pars[3])
        corex.append(np.exp(pars[4]))
        corey.append(-np.exp(pars[5]))
        chi2.append(ef.chi_square(pars))

    # guess_pars = pars.copy()
    # guess_pars[0] = np.log(300.)
    # guess_pars[1] = np.log(1.e5)
    # guess_pars[2] = pf.theta
    # guess_pars[3] = pf.phi
    # guess_pars[4] = ty.core_estimate[0]
    # guess_pars[5] = ty.core_estimate[1]

    # names = ('xmax','nmax','zenith','azimuth','corex','corey','corez','x0','lambda')
    # ls = LeastSquares(ef.input_indices,ef.real_output,ef.real_error,ef.model,verbose=1)
    # m = Minuit(ls,guess_pars,name=names)
    # m.fixed = True
    # m.fixed[0] = False
    # m.fixed[1] = False
    # # m.fixed[2] = False
    # # m.fixed[3] = False
    # # m.fixed[4] = False
    # # m.fixed[5] = False
    # m.limits[0] = (np.log(300.),np.log(600.))
    # m.limits[1] = (np.log(1.e5),np.log(1.e7))
    # m.limits[2] = (0, np.pi/2)
    # m.limits[3] = (0, 2*np.pi)
    # m.limits[4] = (COUNTER_POSITIONS[:,0].min(), COUNTER_POSITIONS[:,0].max())
    # m.limits[5] = (COUNTER_POSITIONS[:,1].min(), COUNTER_POSITIONS[:,1].max())
    # m.errors[0] = np.log(50.)
    # m.errors[1] = np.log(1.e5)
    # m.errors[2] = np.deg2rad(1.)
    # m.errors[3] = np.deg2rad(1.)
    # m.errors[4] = 1.
    # m.errors[5] = 1.
    # m.tol=.001
    # m.simplex()

    # fitpars = np.array([par.value for par in m.params])

    # ngrid = 51

    # xmaxs = np.linspace(np.log(400),np.log(600),ngrid)
    # nmaxs = np.linspace(np.log(1.e6),np.log(3.e6),ngrid)
    # x,n = np.meshgrid(xmaxs,nmaxs)
    # xn = [[xm, nm,*guess_pars[2:]] for xm,nm in zip(x.flatten(),n.flatten())]
    # xncosts = np.array(run_multiprocessing(ef.chi_square,xn,1))
    # plt.figure()
    # plt.contourf(np.exp(xmaxs),np.exp(nmaxs),xncosts.reshape(ngrid,ngrid),xncosts.min() + np.arange(20)**2)
    # plt.xlabel('xmax')
    # plt.ylabel('nmax')
    # plt.semilogy()
    # plt.colorbar(label='chi_square')

    # dang = np.deg2rad(5.)
    # ts = np.linspace(guess_pars[2]-dang,guess_pars[2]+dang,ngrid)
    # ps = np.linspace(guess_pars[3]-dang,guess_pars[3]+dang,ngrid)
    # t,p = np.meshgrid(ts,ps)
    # tp = [[*guess_pars[:2],tm,pm,*guess_pars[4:]] for tm,pm in zip(t.flatten(),p.flatten())]
    # tpcosts = np.array(run_multiprocessing(ef.chi_square,tp,1))
    # plt.figure()
    # plt.contourf(np.rad2deg(ts),np.rad2deg(ps),tpcosts.reshape(ngrid,ngrid),tpcosts.min() + np.arange(20)**2)
    # plt.xlabel('zenith')
    # plt.ylabel('azimuth')
    # plt.colorbar(label='chi_square')

    # dpos = 5.
    # xs = np.linspace(guess_pars[4]-dpos,guess_pars[4]+dpos,ngrid)
    # ys = np.linspace(guess_pars[5]-dpos,guess_pars[5]+dpos,ngrid)
    # xc,yc = np.meshgrid(xs,ys)
    # xy = [[*guess_pars[:4],xm,ym,*guess_pars[-3:]] for xm,ym in zip(xc.flatten(),yc.flatten())]
    # xycosts = np.array(run_multiprocessing(ef.chi_square,xy,1))
    # plt.figure()
    # plt.contourf(xs,ys,xycosts.reshape(ngrid,ngrid),xycosts.min() + np.arange(20)**2)
    # plt.xlabel('corex')
    # plt.ylabel('corey')
    # plt.colorbar(label='chi_square')

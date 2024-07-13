from fit import *
from tyro_fit import tyro, TyroFit
from niche_plane import NichePlane
import numpy as np

from pymultinest.solve import solve

from process_showers import ProcessEvents
from config import CounterConfig
# from datafiles import *
import matplotlib.pyplot as plt
plt.ion()
from utils import plot_event, plot_generator, get_data_files, preceding_noise_file

class LogProbs:
    def __init__(self, ff: FitFeature, guess: list[FitParam]) -> None:
        self.fitfeature = ff
        self.guess = guess
        self.guessdict = {p.name:p for p in guess}
        self.varying_pars = [p.name for p in guess if not p.fixed]
        self.lls = np.array([self.guessdict[name].limits[0] for name in self.varying_pars])
        self.uls = np.array([self.guessdict[name].limits[1] for name in self.varying_pars])
        self.diffs = self.uls - self.lls

    def varpars_to_all(self, pars: np.ndarray) -> np.ndarray:
        '''This method takes the varying parameters and adds them to a dictionary of 
        all parameters.
        '''
        pardict = {p.name:p.value for p in self.guess}
        for par, name in zip(pars,self.varying_pars):
            pardict[name] = par
        pararray = np.array(list(pardict.values()))
        return pararray
    
    def prior(self, unitypars: np.ndarray) -> float:
        '''This maps unity interval to the parameter space of free parameters.
        '''
        outpars = self.lls + self.diffs * unitypars
        return outpars
    
    def lnlike(self, pars: np.ndarray) -> float:
        allpars = self.varpars_to_all(pars)
        return self.fitfeature.lnlike(allpars)

    

data_date_and_time = '20190504034237'
data_files = get_data_files(data_date_and_time)
noise_files = [preceding_noise_file(f) for f in data_files]
cfg = CounterConfig(data_files, noise_files)

pars = [500.,2.e6,np.deg2rad(40.),np.deg2rad(315.), 450., -660.,-25.7,0,70]
ev = BasicParams.get_event(pars)
pe = ProcessEvents(cfg, frozen_noise=False)
real_nfits = pe.gen_nfits_from_event(ev)
pf = NichePlane(real_nfits)
ty = tyro(real_nfits)

guess = make_guess(ty,pf,cfg)

pt = PeakTimes(pf.counters, BasicParams, cfg)
pt.target_parameters = ['zenith','azimuth']
m = init_minuit(pt, guess)
m.tol = .1
m.simplex()

guess = update_guess_values(guess, m)
pa = NormalizedPulseArea(pf.counters, BasicParams, cfg)
pa.target_parameters = ['xmax','nmax','corex','corey']
m = init_minuit(pa, guess)
m.simplex()

guess = update_guess(m)
at = AllTunka(pf.counters, BasicParams, cfg)
at.target_parameters = ['t_offset']
m = init_minuit(at, guess)
m.migrad()

guess = update_guess_values(guess,m)
guessdict = {p.name:p for p in guess}
guessdict['corez'].fixed = True
guessdict['x0'].fixed = True
guessdict['lambda'].fixed = True
s = AllSamples(real_nfits,BasicParams,cfg)

lp = LogProbs(s, guess)
n_params = len(lp.varying_pars)
prefix = 'multinest/'
result = solve(LogLikelihood=lp.lnlike, Prior=lp.prior, n_dims=n_params, outputfiles_basename=prefix, verbose=True)





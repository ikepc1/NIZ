import numpy as np
import emcee
import corner
from multiprocessing import Pool
from copy import deepcopy

from fit import AllSamples, make_guess, BasicParams, FitFeature, FitParam
from tyro_fit import tyro, TyroFit
from niche_plane import NichePlane, NicheFit

class LogProbs:
    def __init__(self, ff: FitFeature, guess: list[FitParam]) -> None:
        self.fitfeature = ff
        self.guess = guess
        self.guessdict = {p.name:p for p in guess}

    def allpars(self, mcmcpars: dict[str, float]) -> np.ndarray:
        '''This method returns all the parameters needed to run the model, adding
        in the fixed parameters.
        '''
        pardict = {p.name:p.value for p in self.guess}
        for parname in mcmcpars:
            pardict[parname] = mcmcpars[parname]
        allpars = np.array(list(pardict.values()))
        return allpars

    def lnprior(self, mcmcpars: dict[str, float]) -> float:
        '''This is simply the limits on the values.
        '''
        for parname in mcmcpars:
            fitpar = self.guessdict[parname]
            if fitpar.limits[0] >= mcmcpars[parname] or fitpar.limits[1] <= mcmcpars[parname]:
                return -np.inf
        return 0.0
    
    def lnprob(self, mcmcpars: dict[str, float]) -> float:
        '''This is the posterior probability.
        '''
        lp = self.lnprior(mcmcpars)
        if not np.isfinite(lp):
            return -np.inf
        allpars = self.allpars(mcmcpars)
        val = lp + self.fitfeature.lnlike(allpars)
        if np.isnan(val):
            return -np.inf
        return val
    
def main(ff: FitFeature, guess: list[FitParam], nwalkers: int = 32, niter: int = 100):
    lp = LogProbs(ff, guess)
    mcmcguess = [p for p in guess if not p.fixed]
    names = [p.name for p in mcmcguess]
    ndim = len(mcmcguess)
    p0 = np.array([p.parguess(nwalkers) for p in mcmcguess]).T

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lp.lnprob, pool=pool, parameter_names=names)
        pos, prob, state = sampler.run_mcmc(p0, niter, progress=True)

    return sampler, pos, prob, state

if __name__ == '__main__':
    from process_showers import ProcessEvents
    from config import CounterConfig
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
    real_nfits = pe.gen_nfits_from_event(ev)
    pf = NichePlane(real_nfits)
    ty = tyro(real_nfits)

    g = make_guess(ty,pf)
    s = AllSamples(real_nfits,BasicParams,cfg)

    sampler, pos, prob, state = main(s,g,niter=10000)
    flat_samples = sampler.get_chain(discard=1000,flat=True)
    labels = [p.name for p in g if not p.fixed]
    fig = corner.corner(flat_samples,labels=labels)


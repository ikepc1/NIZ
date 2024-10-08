import numpy as np
from niche_raw import NicheRaw
from scipy.optimize import curve_fit
from scipy.integrate import quad
from functools import cached_property

from config import COUNTER_POSITIONS_DICT, TRIGGER_POSITION, NICHE_TIMEBIN_SIZE

class NicheFit(NicheRaw):
    """
    Class for fit j-niche waveform data
    """

    def __init__(s,niche_raw_obj: NicheRaw):
        for attr in dir(niche_raw_obj):
            if not attr.startswith('__'):
                setattr(s,attr,getattr(niche_raw_obj,attr))

        s.position = np.array(COUNTER_POSITIONS_DICT[s.name])
        r = niche_raw_obj
        bl0 = (s.waveform[:20].mean()+s.waveform[-20:].mean())/2
        t00 = np.argmax(s.waveform)
        if r.trigPosition > 0:
            if np.abs(t00-(1024-r.trigPosition)) > 30:
                t00 = int(1024-r.trigPosition)
        pk0 = s.waveform[t00]-bl0
        rt0 = 2.
        ft0 = 4.
        p0 = (t00,pk0,rt0,ft0,bl0)
        t = np.arange(len(s.waveform))
        s.t = t
        # ev = 2.*np.ones_like(s.waveform)
        s.baseline_error = np.sqrt(np.mean([np.var(s.waveform[:t00-50]),np.var(s.waveform[t00+50:])]))
        ev = np.full_like(s.waveform,s.baseline_error)
        try:
            pb,pcov = curve_fit(s.tunka_fit,t,s.waveform,p0,ev)
        except RuntimeError:
            pb = np.zeros(5,dtype=float)
            pcov = np.zeros((5,5),dtype=float)
        s.peaktime = pb[0]
        s.peak     = pb[1]
        s.risetime = pb[2]
        s.falltime = pb[3]
        s.baseline = pb[4]
        s.epeaktime = np.sqrt(pcov[0,0])
        s.epeak     = np.sqrt(pcov[1,1])
        s.erisetime = np.sqrt(pcov[2,2])
        s.efalltime = np.sqrt(pcov[3,3])
        s.ebaseline = np.sqrt(pcov[4,4])
        s.intstart = int(np.floor(s.peaktime - 5.*s.risetime))
        s.intend   = int(np.ceil(s.peaktime + 5.*s.falltime))
        s.intsignal = s.waveform[s.intstart:s.intend+1].sum() - (s.intend+1-s.intstart)*s.baseline
        s.eintsignal = np.sqrt(s.intsignal)
        pb[4] = 0 # set baseline to 0 for integrating
        pf = tuple(pb)
        # s.intfit = quad(s.tunka_fit,0,len(s.waveform),pf)[0]

    @property
    def max_value(self) -> float:
        return 4096 - self.baseline

    @property
    def peak_datetime(self) -> np.timedelta64:
        '''This method returns the datetime object for the actual time the peak occurred.
        '''
        ns_diff = int(np.round((self.peaktime - TRIGGER_POSITION) * NICHE_TIMEBIN_SIZE))
        return self.trigtime() + np.timedelta64(ns_diff, 'ns')

    @property
    def ns_diff(self) -> float:
        return (self.peaktime - TRIGGER_POSITION) * NICHE_TIMEBIN_SIZE
    
    @cached_property
    def start_rise(self) -> int:
        level = 3 * self.baseline_error + self.baseline
        # level = self.waveform.max() / 4.
        peak = self.waveform.argmax()
        before_reversed = self.waveform[:peak][::-1]
        n_samples_before = (before_reversed<level).argmax()
        istart = peak - n_samples_before - 1
        return istart
    
    @cached_property
    def end_fall(self) -> int:
        level = 2 * self.baseline_error + self.baseline
        # level = self.waveform.max() / 4.
        peak = self.waveform.argmax()
        after = self.waveform[peak:]
        n_samples_after = (after<level).argmax()
        iend = peak + n_samples_after - 1
        return iend

    @staticmethod
    def tunka_fit(t,t0,pk,rt,ft,bl):
        try:
            nt = len(t)
        except TypeError:
            nt = 0
        if nt > 0:
            x = t-t0
            output = bl*np.ones_like(x)
            lh = x<=0
            lout = np.zeros_like(output[lh])
            f = np.abs(x[lh])/rt
            fl5 = f<5
            f5 = f[fl5]
            lout[fl5] += pk*np.exp(-f5**(2+f5/2))
            output[lh] += lout

            rh = ~lh
            rout = np.zeros_like(output[rh])
            g = x[rh]/ft
            h = 1.3*np.ones_like(g)
            h[g<0.8] = 1.7-0.5*g[g<0.8]
            gex = g**h
            gex700 = gex<700
            rout[gex700] += pk*np.exp(-gex[gex700])
            output[rh] += rout
            return output
        else:
            x = t-t0
            if x<=0:
                f = np.abs(x)/rt
                if f<5:
                    return bl+pk*np.exp(-f**(2+0.5*f))
                else:
                    return 0.
            else:
                g = x/ft
                h = 1.7-0.5*g if g<0.8 else 1.3
                if g**h<700:
                    return bl+pk*np.exp(-g**h)
                else:
                    return 0.
        
    
        

import numpy as np
from scipy.integrate import quad

class TunkaPMTPulse:
    '''This class normalizes the Tunka pmt pulse functional form as a pdf'''

    def __init__(self, t0=0., pk=1., rt=5., ft=10., bl=0.):
        self.C0 = 1.
        self.t0 = t0
        self.pk = pk
        self.rt = rt
        self.ft = ft
        self.bl = bl
        self.normalize()

    @property
    def ll(self):
        return self.t0 - 100.

    @property
    def ul(self):
        return self.t0 + 100.

    def tunka_fit(self, t: np.ndarray) -> np.ndarray:
        """
        Function which gives a Tunka style PMT pulse
        """
        try:
            nin = len(t)
        except TypeError:
            nin = 0
            t = np.array([t])

        x = t-self.t0
        output = self.bl*np.ones_like(x)
        lh = x<=0
        lout = np.zeros_like(output[lh])
        f = np.abs(x[lh])/self.rt
        fl5 = f<5
        f5 = f[fl5]
        lout[fl5] += self.pk*np.exp(-f5**(2+f5/2))
        output[lh] += lout

        rh = ~lh
        rout = np.zeros_like(output[rh])
        g = x[rh]/self.ft
        h = 1.3*np.ones_like(g)
        h[g<0.8] = 1.7-0.5*g[g<0.8]
        gex = g**h
        gex700 = gex<700
        rout[gex700] += self.pk*np.exp(-gex[gex700])
        output[rh] += rout

        if nin == 0:
            return output[0]
        else:
            return output

    def tunka_pdf(self, t: np.ndarray) -> np.ndarray:
        '''This method is the normalized Tunka pulse.
        '''
        return self.C0 * self.tunka_fit(t)

    def normalize(self) -> None:
        '''This method sets the normalization constant.
        '''
        self.C0 /= quad(self.tunka_pdf, self.ll, self.ul)[0]

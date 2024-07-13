import numpy as np

def tunka(t,t0,pk,rt,ft,bl):
    """
    Function which gives a Tunka style PMT pulse
    """
    try:
        nin = len(t)
    except TypeError:
        nin = 0
        t = np.array([t])

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

    if nin == 0:
        return output[0]
    else:
        return output

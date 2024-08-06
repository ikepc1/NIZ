import numpy as np

def tunka(t,t0,pk,rt,ft,bl):
    """
    The Tunka waveform-fitting function.

    Variable:
        t - The independant time variable. Can be a scalar or ndarray.

    Parameters:
        t0 - The time of the peak of the Tunka waveform.
        pk - The peak value of the waveform.
        rt - The risetime for the waveform. Typically about the 50-100% time
        ft - The falltime for the waveform. Typically about the 100-50% time
        bl - The baseline value upon which the waveform sits.

    Returns:
        The value for the waveform (if t is a scalar) or an array of waveform values (if t in an ndarray)

    This function calculates the Tunka waevform, and can be used for fitting waveforms on a background
    (the baseline). The pk and bl values should have the same units as the pk sits above the bl. The
    times: t0, rt, ft, and t should all be in the same units.

    The rise of the waveform begins with a very shapr rise relaxing into a gaussian at t=0. (The power in 
    the exponent goes from 5 to 2.) The fall of the waveform continues this relaxation, with the power in
    the waveform going from 1.7 down to 1.3 (at which it remains fixed).
    """

    try:
        nt = len(t)
    except TypeError:
        nt = 0
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
    if nt>0:
        return output
    else:
        return output[0]

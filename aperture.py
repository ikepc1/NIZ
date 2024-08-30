import numpy as np
import pandas as pd

def trig_prob_wilson_score(ns: int, n: int, alpha = .05) -> tuple[float, float]:
    '''This function calculates the trigger probability and confidence interval using the 
    wilson score method.
    '''
    p = ns / n
    z = 1 - (alpha/2)
    p_est = (1 / (1+(z**2/n))) * (p + z**2/(2*n))
    error = (1 / (1+(z**2/n))) * np.sqrt((p*(1-p)/n) + (z**2/(4*n**2)))
    return p_est, error

def calculate_aperture(radius: float, ns: int, n: int) -> tuple[float, float]:
    '''This function calculates the aperture based on the number of triggers
    and the total possible aperture.
    '''
    total = 2* np.pi**2 * radius**2
    p, p_err = trig_prob_wilson_score(ns,n)
    aperture = total * p
    err = total * p_err
    return aperture, err
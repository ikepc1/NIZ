import numpy as np

ANGLES = np.deg2rad(np.array([0., 5., 10., 15., 20., 25., 30., 35., 40., 41., 42., 43.]))
FRACTIONS = np.array([1., .993, .981, .968, .968, .942, .944, 901., .434, .221, .074, .009])

def filter(angles: np.ndarray) -> np.ndarray:
    '''This function interpolates in Omura's simulated angular response.
    '''
    return np.interp(angles, ANGLES, FRACTIONS)

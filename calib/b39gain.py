import numpy as np

TUBE_AREA = 1197. #effective tube area in m^2
TUBE_AREA_ERR = .03
DIST = 1237. #distance from flasher to center of camera face in m
DIST_ERR = 0.7
QE = .27 #reported tube quantum efficiency
QE_ERR = 0.064 #from Zundel's thesis
COL_EFF = .9 #reported tube collection efficiency, no error, reported from manufacturer
UV_FILTER_EFF = .89 #uv filter efficiency
UV_FILTER_EFF_ERR = .03
NPE_ERR = .045

def photons_per_ster_ns(npes: np.ndarray) -> np.ndarray:
    '''This function converts the number of photoelectrons seen by the b39 test camera
    to the number of photons per steradian per ns produced by the light source.
    '''
    return (npes*DIST**2) / (QE*COL_EFF*UV_FILTER_EFF*TUBE_AREA*100.)

def photons_err(npes: np.ndarray) -> np.ndarray:
    '''This function calculates the error in the photon measurement.
    '''
    over_qcua = 1 / (QE * COL_EFF * UV_FILTER_EFF * TUBE_AREA * 100.)
    nr2 = npes * DIST**2
    dgdn = (DIST**2) * over_qcua
    print(dgdn)
    dgdr = (2 * npes * DIST) * over_qcua
    print(dgdr)
    dgdq = -nr2 * over_qcua / QE
    print(dgdq)
    dgdu = -nr2 * over_qcua / UV_FILTER_EFF
    print(dgdu)
    dgda = -nr2 * over_qcua / TUBE_AREA
    print(dgda)
    err = np.sqrt((dgdn*NPE_ERR)**2 + 
                  (dgdr*DIST_ERR)**2 + 
                  (dgdq*QE_ERR)**2 + 
                  (dgdu*UV_FILTER_EFF_ERR)**2 + 
                  (dgda*TUBE_AREA_ERR)**2)
    return err

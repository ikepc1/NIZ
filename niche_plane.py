import numpy as np
import scipy.constants as spc
from dataclasses import dataclass
import pandas as pd

from niche_fit import NicheFit
from config import COUNTER_POSITIONS_DICT, NAMES

def get_event_from_df(df: pd.DataFrame, event_index: int) -> list[NicheFit]:
    '''This function returns a list of Niche events from the trigger
    dataframe.
    '''
    series = df.loc[event_index][12:]
    return series[series.notna()].tolist()

class NichePlane:
    """
    Class for j-niche events with plane timing fit
    """

    def __init__(s,nfits: list[NicheFit]):
        s.counters = nfits
        s.date = nfits[0].date
        s.meantrigtime = np.mean([5*c.counter for c in s.counters])
        ncntr = len(nfits)
        trigtime = np.empty(ncntr,dtype=float)
        cxyze = np.empty((ncntr,4),dtype=float)
        for i in range(ncntr):
            c = nfits[i]
            n = c.number
            trigtime[i] = 5*c.counter
            cxyze[i,0] = COUNTER_POSITIONS_DICT[c.name][0]
            cxyze[i,1] = COUNTER_POSITIONS_DICT[c.name][1]
            cxyze[i,2] = 5*spc.c*spc.nano * (
                c.peaktime-(len(c.waveform)-c.trigPosition)
                - c.risetime/2)
            cxyze[i,3] = 5*spc.c*spc.nano * np.sqrt(
                c.epeaktime**2+c.erisetime**2/4)
        cxyze[:,0] -= cxyze[:,0].mean()
        cxyze[:,1] -= cxyze[:,1].mean()
        cxyze[:,2] -= (trigtime-s.meantrigtime)*spc.c*spc.nano
        pf = s.planeFit(cxyze)
        s.nx,s.ny,s.z0 = pf
        h = s.planeChi2Hess(pf,cxyze)
        # s.covar = np.linalg.inv(h)
        s.covar = s.invert(h)
        s.enx,s.eny,s.ez0 = np.sqrt(np.diag(s.covar))
        s.theta = np.arcsin(np.sqrt(s.nx**2+s.ny**2))
        s.phi   = np.arctan2(s.ny,s.nx)

    def __str__(s):
        output = "%d.%09d\n"%(int(s.date),int(s.meantrigtime))
        for c in s.counters:
            output += "%s "%(c.name)
        output += '\n'
        output += "nx = %7.4f ny = %7.4f z0 = %5.1f\n"%(s.nx,s.ny,s.z0)
        output += "  +/-%7.4f   +/-%7.4f   +/-%5.1f\n"%(s.enx,s.eny,s.ez0)
        for i in range(3):
            output += "  (%9.2e %9.2e %9.2e)\n"%tuple(s.covar[i])
        output += "th = %7.4f ph = %7.4f"%(s.theta,s.phi)
        return output
    def __repr__(s):
        return s.__str__()
    
    @staticmethod
    def invert(matrix: np.ndarray) -> np.ndarray:
        '''This method wraps numpy invert to deal with the case of singular
        or un-invertible matrices.
        '''
        try:
            return np.linalg.inv(matrix)
        except:
            return np.zeros_like(matrix)

    @staticmethod
    def sigmaMatrix(cxyze: np.ndarray):
        """
        Calculate the Sigma values to be used in linear fitting.
        cxyze is an array with x,y,z,ez in rows and one row per counter.
        Return tuple (S,Sx,Sy,Sxx,Sxy,Syy,Sz,Sxz,Syz)
        """
        x = cxyze[:,0]
        y = cxyze[:,1]
        z = cxyze[:,2]
        ez = cxyze[:,3]
        S   = (1/ez**2).sum()
        Sx  = (x/ez**2).sum()
        Sy  = (y/ez**2).sum()
        Sxx = (x**2/ez**2).sum()
        Sxy = (x*y/ez**2).sum()
        Syy = (y**2/ez**2).sum()
        Sz  = (z/ez**2).sum()
        Sxz = (x*z/ez**2).sum()
        Syz = (y*z/ez**2).sum()
        return S,Sx,Sy,Sxx,Sxy,Syy,Sz,Sxz,Syz
    
    @staticmethod
    def planeFit(cxyze: np.ndarray):
        """
        Do plane fit to counter x,y,z,ez. cxyze is an array with
        x,y,z,ez in rows and one row per counter. Return (nx,ny,z0) the
        unit vector normal (nx,ny) of the plane that best describes the
        data and the offset. Does the linear fitting algorithm.
        """
        S,Sx,Sy,Sxx,Sxy,Syy,Sz,Sxz,Syz = NichePlane.sigmaMatrix(cxyze)
        a = np.array([[Sxx,Sxy,Sx],[Sxy,Syy,Sy],[Sx,Sy,S]])
        b = np.array([Sxz,Syz,Sz])
        try:
            return np.linalg.solve(a,b)
        except:
            return np.zeros_like(b)
        
    @staticmethod
    def planeChi2(p: np.ndarray,cxyze: np.ndarray):
        """
        Return the chi2 from fitting a plane to a set of data in clist.
        The idea is that r.n is the distance between the point (counter)
        and the plane being fit. z is ct, and this is measured perpendicular
        to the fit-plane not to the xy-palne of the counters. If we were to 
        include the altitude of the counters, one would have to add another
        row to cxyze for z and change the current z to ct. r = (xi,yi,0), so
        nz is not used.
        """
        nx = p[0]
        ny = p[1]
        z0 = p[2]
        x = cxyze[:,0]
        y = cxyze[:,1]
        z = cxyze[:,2]
        ez = cxyze[:,3]
        f = z0 + x*nx + y*ny
        ch2 = ((z-f)/ez)**2
        return ch2.sum()

    @staticmethod
    def planeChi2Jac(p: np.ndarray,cxyze: np.ndarray):
        """
        Return the gradient of the chi2 from fitting a plane to a set of
        data in clist
        """
        nx = p[0]
        ny = p[1]
        z0 = p[2]
        S,Sx,Sy,Sxx,Sxy,Syy,Sz,Sxz,Syz = NichePlane.sigmaMatrix(cxyze)
        dc_dz0 = -2*(Sz -z0*S -nx*Sx -ny*Sy)
        dc_dnx = -2*(Sxz-z0*Sx-nx*Sxx-ny*Sxy)
        dc_dny = -2*(Syz-z0*Sy-nx*Sxy-ny*Syy)
        return np.array((dc_dnx,dc_dny,dc_dz0))

    @staticmethod
    def planeChi2Hess(p: np.ndarray,cxyze: np.ndarray):
        """
        Return the 2nd deriviative matrix of the chi2 from fitting a plane
        to a set of data in clist
        """
        nx = p[0]
        ny = p[1]
        z0 = p[2]
        S,Sx,Sy,Sxx,Sxy,Syy,Sz,Sxz,Syz = NichePlane.sigmaMatrix(cxyze)
        # dc_dz0 = -2*(Sz -z0*S -nx*Sx -ny*Sy)
        # dc_dnx = -2*(Sxz-z0*Sx-nx*Sxx-ny*Sxy)
        # dc_dny = -2*(Syz-z0*Sy-nx*Sxy-ny*Syy)
        d2c_dz02   = 2*S
        d2c_dz0dnx = 2*Sx
        d2c_dz0dny = 2*Sy
        d2c_dnx2   = 2*Sxx
        d2c_dnxdny = 2*Sxy
        d2c_dny2   = 2*Syy
        return np.array([[d2c_dnx2,  d2c_dnxdny,d2c_dz0dnx],
                         [d2c_dnxdny,d2c_dny2,  d2c_dz0dny],
                         [d2c_dz0dnx,d2c_dz0dny,d2c_dz02  ]])
    

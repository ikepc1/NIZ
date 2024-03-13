from abc import ABC, abstractmethod
import numpy as np
from scipy.constants import value,nano
from scipy.integrate import cumtrapz
from scipy.spatial.transform import Rotation as R
from typing import Protocol
from importlib.resources import as_file, files
from functools import cached_property
from dataclasses import dataclass, field

from config import AxisConfig

class Shower(ABC):
    '''This is the abstract class containing the needed methods for creating
    a shower profile
    '''
    # X0 = 0. #Default value for X0

    @property
    def X_max(self):
        '''X_max getter'''
        return self._X_max

    @X_max.setter
    def X_max(self, X_max):
        '''X_max property setter'''
        # if X_max <= self.X0:
        #     raise ValueError("X_max cannot be less than X0")
        self._X_max = X_max

    @property
    def N_max(self):
        '''N_max getter'''
        return self._N_max

    @N_max.setter
    def N_max(self, N_max):
        '''N_max property setter'''
        if N_max <= 0.:
            raise ValueError("N_max must be positive")
        self._N_max = N_max

    @property
    def X0(self):
        '''X0 getter'''
        return self._X0

    @X0.setter
    def X0(self, X0):
        '''X0 property setter'''
        self._X0 = X0

    def stage(self, X, X0=36.62):
        '''returns stage as a function of shower depth'''
        return (X-self.X_max)/X0

    def age(self, X: np.ndarray) -> np.ndarray:
        '''Returns shower age at X'''
        return 3*X / (X + 2*self.X_max)

    def dE_dX_at_age(self, s: np.ndarray) -> np.ndarray:
        '''This method computes the avg ionization loss rate at shower age s.
        Parameters:
        s: shower age
        returns: dE_dX in MeV / (g / cm^2)
        '''
        t1 = 3.90883 / ((1.05301 + s)**9.91717)
        t3 = 0.13180 * s
        return t1 + 2.41715 + t3

    def dE_dX(self, X: np.ndarray) -> np.ndarray:
        '''This method computes the avg ionization loss rate at shower depth X.
        Parameters:
        X: Shower depth
        returns: dE_dX in MeV / (g / cm^2)
        '''
        return self.dE_dX_at_age(self.age(X))

    @abstractmethod
    def profile(self,*args,**kwargs):
        '''returns the number of charged particles as a function of depth'''

class MakeGHShower(Shower):
    '''This is the implementation where a shower profile is computed by the
    Gaisser-Hillas function'''

    def __init__(self, X_max: float, N_max: float, X0: float, Lambda: float):
        self.X_max = X_max
        self.N_max = N_max
        self.X0 = X0
        self.Lambda = Lambda

    def __repr__(self):
        return "GHShower(X_max={:.2f} g/cm^2, N_max={:.2f} particles, X0={:.2f} g/cm^2, Lambda={:.2f})".format(
        self.X_max, self.N_max, self.X0, self.Lambda)

    @property
    def Lambda(self):
        '''Lambda property getter'''
        return self._Lambda

    @Lambda.setter
    def Lambda(self, Lambda):
        '''Lambda property setter'''
        if Lambda <= 0.:
            raise ValueError("Negative Lambda")
        self._Lambda = Lambda

    def profile(self, X: np.ndarray):
        '''Return the size of a GH shower at a given depth.
        Parameters:
        X: depth

        Returns:
        # of charged particles
        '''
        x =         (X-self.X0)/self.Lambda
        g0 = x>0.
        m = (self.X_max-self.X0)/self.Lambda
        n = np.zeros_like(x)
        n[g0] = np.exp( m*(np.log(x[g0])-np.log(m)) - (x[g0]-m) )
        return self.N_max * n

def vector_magnitude(vectors: np.ndarray) -> np.ndarray:
    '''This method computes the length of an array of vectors'''
    return np.sqrt((vectors**2).sum(axis = -1))

class Counters(ABC):
    '''This is the class containing the neccessary methods for finding the
    vectors from a shower axis to a user defined array of Cherenkov detectors
    with user defined size'''

    def __init__(self, input_vectors: np.ndarray, input_radius: np.ndarray):
        self.vectors = input_vectors
        self.input_radius = input_radius

    @property
    def vectors(self) -> np.ndarray:
        '''Vectors to user defined Cherenkov counters getter'''
        return self._vectors

    @vectors.setter
    def vectors(self, input_vectors: np.ndarray):
        '''Vectors to user defined Cherenkov counters setter'''
        if type(input_vectors) != np.ndarray:
            input_vectors = np.array(input_vectors)
        if input_vectors.shape[1] != 3 or len(input_vectors.shape) != 2:
            raise ValueError("Input is not an array of vectors.")
        self._vectors = input_vectors

    @property
    def input_radius(self) -> np.ndarray | float:
        '''This is the input counter radius getter.'''
        return self._input_radius

    @input_radius.setter
    def input_radius(self, input_value: np.ndarray | float):
        '''This is the input counter radius setter.'''
        if type(input_value) != np.ndarray:
            input_value = np.array(input_value)
        # if np.size(input_value) == np.shape(self.vectors)[0] or np.size(input_value) == 1:
        #     self._input_radius = input_value
        # if np.size(input_value) != np.shape(self.vectors)[0] or np.size(input_value) != 1:
        #     raise ValueError('Counter radii must either be a single value for all detectors, or a list with a radius corresponding to each defined counter location.')
        if np.size(input_value) == 1:
            self._input_radius = np.full(self.vectors.shape[0], input_value)
        else:
            self._input_radius = input_value
            

    @property
    def N_counters(self) -> int:
        '''Number of counters.'''
        return self.vectors.shape[0]

    @property
    def r(self) -> np.ndarray:
        '''distance to each counter property definition'''
        return vector_magnitude(self.vectors)

    @abstractmethod
    def area(self, *args, **kwargs) -> np.ndarray:
        '''This is the abstract method for the detection surface area normal to
        the axis as seen from each point on the axis. Its shape must be
        broadcastable to the size of the travel_length array, i.e. either a
        single value or (# of counters, # of axis points)'''

    @abstractmethod
    def omega(self, *args, **kwargs) -> np.ndarray:
        '''This abstract method should compute the solid angle of each counter
        as seen by each point on the axis'''

    def area_normal(self) -> np.ndarray:
        '''This method returns the full area of the counting aperture.'''
        return np.pi * self.input_radius**2

    @staticmethod
    def law_of_cosines(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        '''This method returns the angle C across from side c in triangle abc'''
        cos_C = (c**2 - a**2 - b**2)/(-2*a*b)
        cos_C[cos_C > 1.] = 1.
        cos_C[cos_C < -1.] = -1.
        return np.arccos(cos_C)

    def travel_vectors(self, axis_vectors) -> np.ndarray:
        '''This method returns the vectors from each entry in vectors to
        the user defined array of counters'''
        return self.vectors.reshape(-1,1,3) - axis_vectors

    def travel_length(self, axis_vectors) -> np.ndarray:
        '''This method computes the distance from each point on the axis to
        each counter'''
        return vector_magnitude(self.travel_vectors(axis_vectors))

    def cos_Q(self, axis_vectors) -> np.ndarray:
        '''This method returns the cosine of the angle between the z-axis and
        the vector from the axis to the counter'''
        travel_n = self.travel_vectors(axis_vectors) / self.travel_length(axis_vectors)[:,:,np.newaxis]
        return np.abs(travel_n[:,:,-1])

    def travel_n(self, axis_vectors) -> np.ndarray:
        '''This method returns the unit vectors pointing along each travel
        vector.
        '''
        return self.travel_vectors(axis_vectors) / self.travel_length(axis_vectors)[:,:,np.newaxis]

    def calculate_theta(self, axis_vectors) -> np.ndarray:
        '''This method calculates the angle between the axis and counters'''
        travel_length = self.travel_length(axis_vectors)
        axis_length = np.broadcast_to(vector_magnitude(axis_vectors), travel_length.shape)
        counter_length = np.broadcast_to(self.r, travel_length.T.shape).T
        return self.law_of_cosines(axis_length, travel_length, counter_length)
    
class Timing(ABC):
    '''This is the abstract base class which contains the methods needed for
    timing photons from their source points to counting locations. Each timing
    bin corresponds to the spatial/depth bins along the shower axis.
    '''
    c = value('speed of light in vacuum')

    # def __init__(self, axis: Axis, counters: Counters):
    #     self.axis = axis
    #     self.counters = counters
    #     # self.counter_time = self.counter_time()

    def counter_time(self) -> np.ndarray:
        '''This method returns the time it takes after the shower starts along
        the axis for each photon bin to hit each counter

        The size of the returned array is of shape:
        (# of counters, # of axis points)
        '''
        # shower_time = self.axis_time[self.shower.profile(axis.X) > 100.]
        # return shower_time + self.travel_time + self.delay
        return self.axis_time + self.travel_time + self.delay()

    @property
    def travel_time(self) -> np.ndarray:
        '''This method calculates the the time it takes for something moving at
        c to go from points on the axis to the counters.

        The size of the returned array is of shape:
        (# of counters, # of axis points)
        '''
        return self.counters.travel_length(self.axis.vectors) / self.c / nano

    @property
    @abstractmethod
    def axis_time(self) -> np.ndarray:
        '''This method calculates the time it takes the shower (moving at c) to
        progress to each point on the axis

        The size of the returned array is of shape: (# of axis points,)
        '''

    @abstractmethod
    def delay(self) -> np.ndarray:
        '''This method calculates the delay a traveling photon would experience
        (compared to travelling at c) starting from given axis points to the
        detector locations

        The size of the returned array is of shape:
        (# of counters, # of axis points)
        '''

class Yield(Protocol):
    '''This is the protocol needed from a yield object.
    '''

    @property
    def l_mid(self) -> float:
        ...

class Attenuation(ABC):
    '''This is the abstract base class whose specific implementations will
    calculate the fraction of light removed from the signal at each atmospheric
    step.
    '''
    # atm = USStandardAtmosphere()

    with as_file(files('CHASM.data')/'abstable.npz') as file:
        abstable = np.load(file)

    ecoeff = abstable['ecoeff']
    l_list = abstable['wavelength']
    altitude_list = abstable['height']

    @property
    def yield_array(self) -> list[Yield]:
        return self._yield_array
    
    @yield_array.setter
    def yield_array(self, arr: list[Yield]):
        self._yield_array = arr

    # def __init__(self, axis: Axis, counters: Counters, yield_array: np.ndarray):
    #     self.axis = axis
    #     self.counters = counters
    #     self.yield_array = yield_array
    #     self.atm = axis.atm

    # def vertical_log_fraction(self) -> np.ndarray:
    #     '''This method returns the natural log of the fraction of light which
    #     survives each axis step if the light is travelling vertically.
    #
    #     The returned array is of size:
    #     # of yield bins, with each entry being on size:
    #     # of axis points
    #     '''
    #     log_fraction_array = np.empty_like(self.yield_array, dtype='O')
    #     N = self.atm.number_density(self.axis.h) / 1.e6 #convert to particles/cm^3
    #     dh = self.axis.dh * 1.e2 #convert to cm
    #     for i, y in enumerate(self.yield_array):
    #         cs = self.rayleigh_cs(self.axis.h, y.l_mid)
    #         log_fraction_array[i] = -cs * N * dh
    #     return log_fraction_array

    # def vertical_log_fraction(self) -> np.ndarray:
    #     '''This method returns the natural log of the fraction of light which
    #     survives each axis step if the light is travelling vertically.
    #
    #     The returned array is of size:
    #     # of yield bins, with each entry being of size:
    #     # of axis points
    #     '''
    #     log_fraction_array = np.empty_like(self.yield_array, dtype='O')
    #     for i, y in enumerate(self.yield_array):
    #         ecoeffs = self.ecoeff[np.abs(y.l_mid-self.l_list).argmin()]
    #         e_of_h = np.interp(self.axis.h, self.altitude_list, ecoeffs)
    #         frac_surviving = np.exp(-e_of_h)
    #         frac_step_surviving = 1. - np.diff(frac_surviving[::-1], append = 1.)[::-1]
    #         log_fraction_array[i] = np.log(frac_step_surviving)
    #     return log_fraction_array

    def vertical_log_fraction(self) -> np.ndarray:
        '''This method returns the natural log of the fraction of light which
        survives each axis step if the light is travelling vertically.

        The returned array is of size:
        # of yield bins, with each entry being of size:
        # of axis points
        '''
        return np.frompyfunc(self.calculate_vlf,1,1)(self.lambda_mids)

    def calculate_vlf(self, l):
        '''This method returns the natural log of the fraction of light which
        survives each axis step if the light is travelling vertically

        Parameters:
        y: yield object

        Returns:
        array of vertical-log-fraction values (size = # of axis points)
        '''
        ecoeffs = self.ecoeff[np.abs(l - self.l_list).argmin()]
        e_of_h = np.interp(self.axis.altitude, self.altitude_list, ecoeffs)
        frac_surviving = np.exp(-e_of_h)
        frac_step_surviving = 1. - np.diff(frac_surviving[::-1], append = 1.)[::-1]
        return np.log(frac_step_surviving)
    #

    @property
    def lambda_mids(self):
        '''This property is a numpy array of the middle of each wavelength bin'''
        l_mid_array = np.empty_like(self.yield_array)
        for i, y in enumerate(self.yield_array):
            l_mid_array[i] = y.l_mid
        return l_mid_array

    def nm_to_cm(self,l):
        return l*nano*1.e2

    def rayleigh_cs(self, h, l = 400.):
        '''This method returns the Rayleigh scattering cross section as a
        function of both the height in the atmosphere and the wavelength of
        the scattered light. This does not include the King correction factor.

        Parameters:
        h - height (m) single value or np.ndarray
        l - wavelength (nm) single value or np.ndarray

        Returns:
        sigma (cm^2 / particle)
        '''
        l_cm = self.nm_to_cm(l) #convert to cm
        N = self.atm.number_density(h) / 1.e6 #convert to particles/cm^3
        f1 = (24. * np.pi**3) / (N**2 * l_cm**4)
        n = self.atm.delta(h) + 1
        f2 = (n**2 - 1) / (n**2 + 2)
        return f1 * f2**2

    def fraction_passed(self):
        '''This method returns the fraction of light originating at each
        step on the axis which survives to reach each counter.

        The size of the returned array is of shape:
        # of yield bins, with each entry being on size:
        (# of counters, # of axis points)
        '''
        log_fraction_passed_array = self.log_fraction_passed()
        fraction_passed_array = np.empty_like(log_fraction_passed_array, dtype= 'O')
        for i, lfp in enumerate(log_fraction_passed_array):
            lfp[lfp>0.] = -100.
            fraction_passed_array[i] = np.exp(lfp)
        return fraction_passed_array

    @abstractmethod
    def log_fraction_passed(self) -> np.ndarray:
        '''This method should return the natural log of the fraction of light
        originating at each step on the axis which survives to reach the
        counter.

        The size of the returned array is of shape:
        # of yield bins, with each entry being on size:
        (# of counters, # of axis points)
        '''

class CurvedAtmCorrection(ABC):
    '''This is the abstract base class for performing a curved atmosphere
    correction.
    '''

    @abstractmethod
    def curved_correction(self, vert: np.ndarray) -> np.ndarray:
        '''This method ahould perform the integration of the quantity vert
        for either an upward or downward axis.
        '''

    def __call__(self, vert: np.ndarray) -> np.ndarray:
        return self.curved_correction(vert)

class MakeYield:
    '''This class interacts with the table of Cherenkov yield ratios'''
    # npz_files = {-3.5 : 'y_t_delta_lX_-4_to_-3.npz',
    #              -2.5 : 'y_t_delta_lX_-3_to_-2.npz',
    #              -1.5 : 'y_t_delta_lX_-2_to_-1.npz',
    #              -.5 : 'y_t_delta_lX_-1_to_0.npz',
    #              .5 : 'y_t_delta_lX_0_to_1.npz',
    #              -100. : 'y_t_delta.npz'}

    lXs = np.arange(-6,0)

    def __init__(self, l_min: float, l_max: float):
        self.l_min = l_min
        self.l_max = l_max
        self.l_mid = np.mean([l_min, l_max])

    def __repr__(self):
        return "Yield(l_min={:.2f} nm, l_max={:.2f} nm)".format(
        self.l_min, self.l_max)

    def find_nearest_interval(self, lX: float) -> tuple:
        '''This method returns the start and end points of the lX interval that
        the mesh falls within.
        '''
        index = np.searchsorted(self.lXs[:-1], lX)
        if index == 0:
            return self.lXs[0], self.lXs[1]
        else:
            return self.lXs[index-1], self.lXs[index]

    def get_npz_file(self, lX: float) -> str:
        '''This method returns the gg array file for the axis' particular
        log(moliere) interval.
        '''
        start, end = self.find_nearest_interval(lX)
        return f'y_t_delta_lX_{start}_to_{end}.npz'

    # def get_npz_file(self, lX: float):
    #     # lX_midbin_array = np.array(list(self.npz_files.keys()))
    #     # lX_key = lX_midbin_array[np.abs(lX - lX_midbin_array).argmin()]
    #     # return self.npz_files[lX_key]
    #     return 'y_t_delta.npz'

    def set_yield_attributes(self, y: dict[str,np.ndarray]):
        '''This method sets the yield (as a function of stage and delta)
        attributes from the specified file.
        '''

        self.y_delta_t = y['y_t_delta'] * self.lambda_interval(self.l_min, self.l_max)
        self.delta = y['ds']
        self.t = y['ts']

    def set_yield_at_lX(self, lX: float):
        self.set_yield_attributes(self.get_npz_file(lX))

    def lambda_interval(self, l_min, l_max):
        '''This method returns the factor that results from integrating the
        1/lambda^2 factor in the Frank Tamm formula
        '''
        return 1 / (l_min * nano) - 1 / (l_max * nano)

    def y_of_t(self, t: float):
        '''This method returns the array of yields (for each delta) of the
        tabulated stage nearest to the given t
        '''
        return self.y_delta_t[np.abs(t - self.t).argmin()]

    def y(self, d: float, t: float):
        '''This method returns the average Cherenkov photon yield per meter
        per charged particle at a given stage and delta.
        '''
        return np.interp(d, self.delta, self.y_of_t(t))

    def y_list(self, t_array: np.ndarray, delta_array: np.ndarray):
        '''This method returns a list of average Cherenkov photon yields
        corresponding to a list of stages and deltas.
        Parameters:
        t_array: numpy array of stages
        delta_array: numpy array of corresponding deltas

        Returns: numpy array of yields
        '''
        y_array = np.empty_like(t_array)
        for i, (t,d) in enumerate(zip(t_array,delta_array)):
            y_array[i] = self.y(delta_array[i], t)
        return y_array

@dataclass
class Yield:
    '''This is the implementation of the yield element'''
    l_min: float
    l_max: float
    N_bins: int = 10
    element_type: str = field(init=False, default='yield', repr=False)

    def make_lambda_bins(self):
        '''This method creates a list of bin low edges and a list of bin high
        edges'''
        bin_edges = np.linspace(self.l_min, self.l_max, self.N_bins+1)
        return bin_edges[:-1], bin_edges[1:]

    def create(self) -> list[MakeYield]:
        '''This method returns an instantiated yield object'''
        bin_minimums, bin_maximums = self.make_lambda_bins()
        yield_array = []
        for i, (min, max) in enumerate(zip(bin_minimums, bin_maximums)):
            yield_array.append(MakeYield(min, max))
        return yield_array

@dataclass
class AxisParams:
    zenith: float
    azimuth: float
    ground_level: float

class Axis(ABC):
    '''This is the abstract base class which contains the methods for computing
    the cartesian vectors and corresponding slant depths of an air shower'''

    earth_radius = 6.371e6 #meters
    lX = -100. #This is the default value for the distance to the axis in log moliere units (in this case log(-inf) = 0, or on the axis)
    

    def __init__(self, params: AxisParams):
        self.config = AxisConfig()
        self.atm = self.config.ATM
        self.ground_level = params.ground_level
        self.zenith = params.zenith
        self.azimuth = params.azimuth
        self.altitude = self.set_initial_altitude()

    @property
    def zenith(self) -> float:
        '''polar angle  property getter'''
        return self._zenith

    @zenith.setter
    def zenith(self, zenith):
        '''zenith angle property setter'''
        if zenith >= np.pi/2:
            raise ValueError('Zenith angle cannot be greater than pi / 2')
        if zenith < 0.:
            raise ValueError('Zenith angle cannot be less than 0')
        self._zenith = zenith

    @property
    def azimuth(self) -> float:
        '''azimuthal angle property getter'''
        return self._azimuth

    @azimuth.setter
    def azimuth(self, azimuth):
        '''azimuthal angle property setter'''
        if azimuth >= 2 * np.pi:
            raise ValueError('Azimuthal angle must be less than 2 * pi')
        if azimuth < 0.:
            raise ValueError('Azimuthal angle cannot be less than 0')
        self._azimuth = azimuth

    @property
    def ground_level(self) -> float:
        '''ground level property getter'''
        return self._ground_level

    @ground_level.setter
    def ground_level(self, value):
        '''ground level property setter'''
        if value > self.atm.maximum_height:
            raise ValueError('Ground level too high')
        self._ground_level = value

    def set_initial_altitude(self) -> np.ndarray:
        '''altitude property definition'''
        return np.linspace(self.ground_level, self.atm.maximum_height, self.config.N_POINTS)

    @property
    def dh(self) -> np.ndarray:
        '''This method sets the dh attribute'''
        dh = self.h[1:] - self.h[:-1]
        return np.concatenate((np.array([0]),dh))

    @property
    def h(self) -> np.ndarray:
        '''This is the height above the ground attribute'''
        hs = self.altitude - self.ground_level
        hs[0] = 1.e-5
        return hs

    @property
    def delta(self) -> np.ndarray:
        '''delta property definition (index of refraction - 1)'''
        return self.atm.delta(self.altitude)

    @property
    def density(self) -> np.ndarray:
        '''Axis density property definition (kg/m^3)'''
        return self.atm.density(self.altitude)

    @property
    def moliere_radius(self) -> np.ndarray:
        '''Moliere radius property definition (m)'''
        return 96. / self.density

    def h_to_axis_R_LOC(self,h,theta) -> np.ndarray:
        '''Return the length along the shower axis from the point of Earth
        emergence to the height above the surface specified

        Parameters:
        h: array of heights (m above ground level)
        theta: polar angle of shower axis (radians)

        returns: r (m) (same size as h), an array of distances along the shower
        axis_sp.
        '''
        cos_EM = np.cos(np.pi-theta)
        R = self.earth_radius + self.ground_level
        r_CoE= h + R # distance from the center of the earth to the specified height
        rs = R*cos_EM + np.sqrt(R**2*cos_EM**2-R**2+r_CoE**2)
        # rs -= rs[0]
        # rs[0] = 1.
        return rs # Need to find a better way to define axis zero point, currently they are all shifted by a meter to prevent divide by zero errors

    @classmethod
    def theta_normal(cls,h,r) -> np.ndarray:
        '''This method calculates the angle the axis makes with respect to
        vertical in the atmosphere (at that height).

        Parameters:
        h: array of heights (m above sea level)
        theta: array of polar angle of shower axis (radians)

        Returns:
        The corrected angles(s)
        '''
        cq = (r**2 + h**2 + 2*cls.earth_radius*h)/(2*r*(cls.earth_radius+h))
        cq[cq>1.] = 1.
        cq[cq<-1.] = -1.
        # cq = ((cls.earth_radius+h)**2+r**2-cls.earth_radius**2)/(2*r*(cls.earth_radius+h))
        return np.arccos(cq)

    @property
    def theta_difference(self) -> np.ndarray:
        '''This property is the difference between the angle a vector makes with
        the z axis and the angle it makes with vertical in the atmosphere at
        all the axis heights.
        '''
        return self.zenith - self.theta_normal(self.h, self.r)

    @property
    def dr(self) -> np.ndarray:
        '''This method sets the dr attribute'''
        dr = self.r[1:] - self.r[:-1]
        return np.concatenate((np.array([0]),dr))

    @property
    def vectors(self) -> np.ndarray:
        '''axis vector property definition

        returns a vector from the origin to a distances r
        along the axis'''
        ct = np.cos(self.zenith)
        st = np.sin(self.zenith)
        cp = np.cos(self.azimuth)
        sp = np.sin(self.azimuth)
        axis_vectors = np.empty([np.shape(self.r)[0],3])
        axis_vectors[:,0] = self.r * st * cp
        axis_vectors[:,1] = self.r * st * sp
        axis_vectors[:,2] = self.r * ct
        return axis_vectors

    def depth(self, r: np.ndarray) -> np.ndarray:
        '''This method is the depth as a function of distance along the shower
        axis'''
        return np.interp(r, self.r, self.X)

    def get_timing(self, curved_correction: CurvedAtmCorrection) -> Timing:
        '''This function returns an instantiated timing object appropriate for
        the axis implementation.'''
        return self.get_timing_class()(curved_correction)

    def get_attenuation(self, curved_correction: CurvedAtmCorrection, y: list[MakeYield]) -> Attenuation:
        '''This function returns an instantiated attenuation object appropriate for
        the axis implementation.'''
        return self.get_attenuation_class()(curved_correction, y)
    
    def get_curved_atm_correction(self, counters: Counters) -> CurvedAtmCorrection:
        return self.get_curved_atm_correction_class()(self, counters)

    @property
    @abstractmethod
    def r(self) -> np.ndarray:
        '''r property definition'''
        # return self.h_to_axis_R_LOC(self.h, self.zenith)

    @property
    @abstractmethod
    def X(self) -> np.ndarray:
        '''This method sets the depth along the shower axis attribute'''

    @abstractmethod
    def slant_depth_integrand(self, h: float | np.ndarray) -> float | np.ndarray:
        '''This method is the integrand as a function of altitude for calculating slant
        depth.
        '''

    @abstractmethod
    def distance(self, X: np.ndarray) -> np.ndarray:
        '''This method is the distance along the axis as a function of depth'''

    @abstractmethod
    def theta(self) -> np.ndarray:
        '''This method computes the ACUTE angles between the shower axis and the
        vectors toward each counter'''

    @abstractmethod
    def get_timing_class(self) -> Timing:
        '''This method should return the specific timing factory needed for
        the specific axis type (up or down)
        '''

    @abstractmethod
    def get_attenuation_class(self) -> Attenuation:
        '''This method should return the specific attenuation factory needed for
        the specific axis type (up or down)
        '''

    @abstractmethod
    def get_gg_file(self) -> str:
        '''This method should return the gg array for the particular axis type.
        For linear axes, it should return the regular gg array. For a mesh axis,
        it should return the gg array for the axis' particular log(moliere)
        interval.
        '''

    @abstractmethod
    def reset_for_profile(self, shower: Shower) -> None:
        '''This method re-defines the height attribute depending on where the shower 
        falls along the axis. The purpose of this is to avoid running expensive 
        calculations where no significant amount of particles exist.
        '''

    @abstractmethod
    def get_curved_atm_correction_class(self) -> CurvedAtmCorrection:
        '''This method returns the curved atm correction class for the speccific axis 
        implementation.
        '''

class MakeSphericalCounters(Counters):
    '''This is the implementation of the Counters abstract base class for
    CORSIKA IACT style spherical detection volumes.

    Parameters:
    input_vectors: rank 2 numpy array of shape (# of counters, 3) which is a
    list of vectors (meters).
    input_radius: Either a single value for an array of values corresponding to
    the spherical radii of the detection volumes (meters).
    '''

    def __repr__(self):
        return "SphericalCounters({:.2f} Counters with average area~ {:.2f})".format(
        self.vectors.shape[0], self.area_normal().mean())

    def area(self) -> np.ndarray:
        '''This is the implementation of the area method, which calculates the
        area of spherical counters as seen from the axis.
        '''
        return self.area_normal()

    def omega(self, axis: Axis) -> np.ndarray:
        '''This method computes the solid angle of each counter as seen by
        each point on the axis'''
        return (self.area() / (self.travel_length(axis).T)**2).T
    
class LateralSpread:
    '''This class interacts with the table of NKG universal lateral distributions
    '''
    with as_file(files('CHASM.data')/'n_t_lX_of_t_lX.npz') as file:
        lx_table = np.load(file)

    t = lx_table['ts']
    lX = lx_table['lXs']
    n_t_lX_of_t_lX = lx_table['n_t_lX_of_t_lX']

    @classmethod
    def get_t_indices(cls, input_t: np.ndarray) -> np.ndarray:
        '''This method returns the indices of the stages in the table closest to
        the input stages.
        '''
        return np.abs(input_t[:, np.newaxis] - cls.t).argmin(axis=1)

    @classmethod
    def get_lX_index(cls, input_lX: float) -> np.ndarray:
        '''This method returns the index closest to the input lX within the 5
        tabulated lX values.
        '''
        return np.abs(input_lX - cls.lX).argmin()

    @classmethod
    def nch_fractions(cls, input_ts: np.ndarray, input_lX: float) -> np.ndarray:
        '''This method returns the fraction of charged particles at distance
        exp(lX) moliere units from the shower axis at an array of stages
        (input_ts).
        '''
        t_indices = cls.get_t_indices(input_ts)
        lX_index = cls.get_lX_index(input_lX)
        return cls.n_t_lX_of_t_lX[t_indices,lX_index]
    
def axis_to_mesh(lX: float, axis: Axis, shower: Shower, N_ring: int = 20) -> tuple:
    '''This function takes an shower axis and creates a 3d mesh of points around
    the axis (in coordinates where the axis is the z-axis)
    Parameters:
    axis: axis type object
    shower: shower type object
    Returns:
    an array of vectors to points in the mesh
    size (, 3)
    The corresponding # of charged particles at each point
    The corresponding array of stages (for use in universality calcs)
    The corresponding array of deltas (for use in universality calcs)
    The corresponding array of shower steps (dr) in m (for Cherenkov yield calcs)
    The corresponding array of altitudes (for timing calcs)

    '''
    X = np.exp(lX) #number of moliere units for the radius of the ring
    X_to_m = X * axis.moliere_radius
    X_to_m[X_to_m>axis.config.MAX_RING_SIZE] = axis.config.MAX_RING_SIZE
    axis_t = shower.stage(axis.X)
    total_nch = shower.profile(axis.X) * LateralSpread.nch_fractions(axis_t,lX)
    axis_d = axis.delta
    axis_dr = axis.dr
    axis_altitude = axis.altitude
    r = axis.r
    ring_theta = np.arange(0,N_ring) * 2 * np.pi / N_ring
    ring_x = X_to_m[:, np.newaxis] * np.cos(ring_theta)
    ring_y = X_to_m[:, np.newaxis] * np.sin(ring_theta)
    x = ring_x.flatten()
    y = ring_y.flatten()
    z = np.repeat(r, N_ring)
    t = np.repeat(axis_t, N_ring)
    d = np.repeat(axis_d, N_ring)
    nch = np.repeat(total_nch / N_ring, N_ring)
    dr = np.repeat(axis_dr, N_ring)
    a = np.repeat(axis_altitude, N_ring)
    return np.array((x,y,z)).T, nch, t, d, dr, a

def rotate_mesh(mesh: np.ndarray, theta: float, phi: float) -> np.ndarray:
    '''This function rotates an array of vectors by polar angle theta and
    azimuthal angle phi.

    Parameters:
    mesh: numpy array of axis vectors shape = (# of vectors, 3)
    theta: axis polar angle (radians)
    phi: axis azimuthal angle (radians)
    Returns a numpy array the same shape as the original list of vectors
    '''
    theta_rot_axis = np.array([0,1,0]) #y axis
    phi_rot_axis = np.array([0,0,1]) #z axis
    theta_rot_vector = theta * theta_rot_axis
    phi_rot_vector = phi * phi_rot_axis
    theta_rotation = R.from_rotvec(theta_rot_vector)
    phi_rotation = R.from_rotvec(phi_rot_vector)
    mesh_rot_by_theta = theta_rotation.apply(mesh)
    mesh_rot_by_theta_then_phi = phi_rotation.apply(mesh_rot_by_theta)
    return mesh_rot_by_theta_then_phi

class MeshAxis(Axis):
    '''This class is the implementation of an axis where the sampled points are
    spread into a mesh.
    '''
    lXs = np.arange(-6,0) #Log moliere radii corresponding to the bin edges of the tabulated gg files.

    def __init__(self, lX_interval: tuple, linear_axis: Axis, shower: Shower):
        self.lX_interval = lX_interval
        self.lX = np.mean(lX_interval)
        self.linear_axis = linear_axis
        self.config = linear_axis.config
        self.atm = linear_axis.atm
        self.zenith = linear_axis.zenith
        self.azimuth = linear_axis.azimuth
        self.ground_level = linear_axis.ground_level
        self.shower = shower
        mesh, self.nch, self._t, self._d, self._dr, self._a  = axis_to_mesh(self.lX, 
                                                                            self.linear_axis, 
                                                                            self.shower,
                                                                            N_ring=self.config.N_IN_RING)
        self.meshX = np.repeat(self.X,self.config.N_IN_RING)
        self.rotated_mesh = rotate_mesh(mesh, linear_axis.zenith, linear_axis.azimuth)

    @property
    def lX_inteval(self) -> tuple:
        '''lX_inteval property getter'''
        return self._lX_inteval

    @lX_inteval.setter
    def lX_inteval(self, interval):
        '''lX_inteval angle property setter'''
        # if type(interval) != tuple:
        #     raise ValueError('lX interval needs to be a tuple')
        # if interval not in zip(self.lXs[:-1], self.lXs[1:]):
        #     raise ValueError('lX interval not in tabulated ranges.')
        self._lX_inteval = interval

    @property
    def delta(self) -> np.ndarray:
        '''Override of the delta property so each one corresponds to its
        respective mesh point.'''
        return self._d

    @property
    def vectors(self) -> np.ndarray:
        '''axis vector property definition

        returns vectors from the origin to mesh axis points.
        '''
        return self.rotated_mesh

    @property
    def r(self) -> np.ndarray:
        '''overrided r property definition'''
        return vector_magnitude(self.rotated_mesh)

    @property
    def dr(self) -> np.ndarray:
        '''overrided dr property definition'''
        return self._dr

    @property
    def altitude(self) -> np.ndarray:
        '''overrided h property definition'''
        return self._a

    @property
    def X(self) -> np.ndarray:
        '''This method sets the depth along the shower axis attribute'''
        return self.linear_axis.X
    
    def slant_depth_integrand(self, h: float | np.ndarray) -> float | np.ndarray:
        return self.linear_axis.slant_depth_integrand(h)

    def distance(self, X: np.ndarray) -> np.ndarray:
        '''This method is the distance along the axis as a function of depth'''
        return self.linear_axis.distance(X)

    def theta(self, axis_vectors, counters: Counters) -> np.ndarray:
        '''This method computes the ACUTE angles between the shower axis and the
        vectors toward each counter'''
        return self.linear_axis.theta(axis_vectors, counters)

    def get_timing_class(self) -> Timing:
        '''This method should return the specific timing factory needed for
        the specific axis type (up or down)
        '''
        return self.linear_axis.get_timing_class()

    def get_attenuation_class(self) -> Attenuation:
        '''This method should return the specific attenuation factory needed for
        the specific axis type (up or down)
        '''
        return self.linear_axis.get_attenuation_class()

    def get_gg_file(self) -> str:
        '''This method returns the gg array file for the axis' particular
        log(moliere) interval.
        '''
        start, end = self.find_nearest_interval()
        return f'gg_t_delta_theta_lX_{start}_to_{end}.npz'

    def find_nearest_interval(self) -> tuple:
        '''This method returns the start and end points of the lX interval that
        the mesh falls within.
        '''
        index = np.searchsorted(self.lXs[:-1],self.lX)
        if index == 0:
            return self.lXs[0], self.lXs[1]
        else:
            return self.lXs[index-1], self.lXs[index]

    # def get_gg_file(self):
    #     '''This method returns the original gg array file.
    #     '''
    #     # return 'gg_t_delta_theta_lX_-6_to_-5.npz'
    #     # return 'gg_t_delta_theta_mc.npz'
    #     return 'gg_t_delta_theta_2020_normalized.npz'

    def reset_for_profile(self, shower: Shower) -> None:
        return self.linear_axis.reset_for_profile(shower)
    
    def get_curved_atm_correction_class(self) -> CurvedAtmCorrection:
        return self.linear_axis.get_curved_atm_correction_class()
    
class MeshShower(Shower):
    '''This class is the implementation of a shower where the shower particles are
    distributed to a mesh axis rather than just the longitudinal axis.
    '''

    def __init__(self, mesh_axis: MeshAxis):
        self.mesh_axis = mesh_axis

    def stage(self, X: np.ndarray) -> np.ndarray:
        '''This method returns the corresponding stage of each mesh point.
        '''
        return self.mesh_axis._t

    def profile(self, X: np.ndarray) -> np.ndarray:
        '''This method returns the number of charged particles at each mesh
        point
        '''
        return self.mesh_axis.nch
    
class MakeDownwardAxis(Axis):
    '''This is the implementation of an axis for a downward going shower'''

    @cached_property
    def X(self) -> np.ndarray:
        '''This method sets the depth attribute, depths are added along the axis
        in the downward direction'''
        # depths = np.zeros_like(self.r)
        # A = self.altitude[:-1]
        # B = self.altitude[1:]
        # depths[:-1] = np.array([quad(self.slant_depth_integrand,a,b)[0] for a,b in zip(A,B)])
        # return np.cumsum(depths[::-1] / 10.)[::-1]
        return -cumtrapz(self.slant_depth_integrand(self.altitude[::-1]) / 10.,self.altitude[::-1], initial=0)[::-1]

    def distance(self, X: np.ndarray) -> np.ndarray:
        '''This method is the distance along the axis as a function of depth'''
        return np.interp(X, self.X[::-1], self.r[::-1])

    def theta(self, axis_vectors, counters: Counters) -> np.ndarray:
        '''This method returns the angle between the axis and the vector going
        to the counter, in this case it's the internal angle'''
        return counters.calculate_theta(axis_vectors)
    
    def reset_for_profile(self, shower: Shower) -> None:
        '''This method resets the attributes of the class based on where the shower
        occurs on the axis. We dont need to run universality calculations where
        there's no shower.self
        '''
        ids = shower.profile(self.X) >= self.config.MIN_CHARGED_PARTICLES
        a = self.altitude[::-1]
        self.altitude = a[np.argmax(ids[::-1]):][::-1]
        x = self.X[::-1]
        self.X = x[np.argmax(ids[::-1]):][::-1]
        # self.altitude = self.altitude[ids]

class MakeDownwardAxisFlatPlanarAtm(MakeDownwardAxis):
    '''This is the implementation of a downward going shower axis with a flat
    planar atmosphere.'''

    def __repr__(self):
        return "DownwardAxisFlatPlanarAtm(theta={:.2f} rad, phi={:.2f} rad, ground_level={:.2f} m)".format(
        self.zenith, self.azimuth, self.ground_level)

    def slant_depth_integrand(self, alt: float | np.ndarray) -> float | np.ndarray:
        '''This is the integrand needed to calculate slant depth.
        '''
        return self.atm.density(alt) / np.cos(self.zenith)

    @property
    def r(self) -> np.ndarray:
        '''This is the axis distance property definition'''
        return self.h / np.cos(self.zenith)

    def get_timing_class(self) -> Timing:
        '''This method returns the flat atm downward timing class
        '''
        return DownwardTiming

    def get_attenuation_class(self) -> Attenuation:
        '''This method returns the flat atmosphere attenuation object for downward
        axes'''
        return DownwardAttenuation

    def get_gg_file(self) -> str:
        '''This method returns the original gg array file.
        '''
        return 'gg_t_delta_theta_mc.npz'
        # return 'gg_t_delta_theta_lX_-2_to_-1.npz'
    
    def get_curved_atm_correction_class(self) -> CurvedAtmCorrection:
        return NoCurvedCorrection
    
class NoCurvedCorrection(CurvedAtmCorrection):
    '''This class is does nothing but pass the axis and counters object.
    '''
    def __init__(self, axis: Axis, counters: Counters) -> None:
        self.axis = axis
        self.counters = counters
        self.cQ = counters.cos_Q(axis.vectors)

    def curved_correction(self, vert: np.ndarray) -> None:
        pass

class DownwardTiming(Timing):
    '''This is the implementation of timing for a downward going shower with no
    correction for atmospheric curveature
    '''

    def __init__(self, curved_correction: NoCurvedCorrection):
        self.curved_correction = curved_correction
        self.axis = curved_correction.axis
        self.counters = curved_correction.counters
        # self.counter_time = self.counter_time()

    def __repr__(self):
        return f"DownwardTiming(axis=({self.axis.__repr__}), \
                                counters=({self.counters.__repr__}))"

    def vertical_delay(self) -> np.ndarray:
        '''This is the delay a vertically travelling photon would experience
        compared to something travelling at c
        '''
        return np.cumsum((self.axis.delta*self.axis.dh))/self.c/nano

    @property
    def axis_time(self) -> np.ndarray:
        '''This is the implementation of the axis time property

        This method calculates the time it takes the shower (moving at c) to
        progress to each point on the axis

        The size of the returned array is of size: (# of axis points,)
        '''
        return -self.axis.r / self.c / nano

    def delay(self) -> np.ndarray:
        '''This is the implementation of the delay property
        '''
        return self.vertical_delay() / self.curved_correction.cQ#self.counters.cos_Q(self.axis.vectors)

class DownwardAttenuation(Attenuation):
    '''This is the implementation of signal attenuation for an downward going air
    shower with a flat atmosphere.
    '''

    def __init__(self, curved_correction: NoCurvedCorrection, yield_array: np.ndarray):
        self.curved_correction = curved_correction
        self.axis = curved_correction.axis
        self.counters = curved_correction.counters
        self.yield_array = yield_array
        self.atm = self.axis.atm

    def log_fraction_passed(self) -> np.ndarray:
        '''This method returns the natural log of the fraction of light
        originating at each step on the axis which survives to reach the
        counter.

        The size of the returned array is of shape:
        # of yield bins, with each entry being on size:
        (# of counters, # of axis points)
        '''
        vert_log_fraction_list = self.vertical_log_fraction()
        log_frac_passed_list = np.empty_like(vert_log_fraction_list, dtype='O')
        # cQ = self.counters.cos_Q(self.axis.vectors)
        for i, v_log_frac in enumerate(vert_log_fraction_list):
            log_frac_passed_list[i] = np.cumsum(v_log_frac / self.curved_correction.cQ, axis=1)
        return log_frac_passed_list
    
class CherenkovPhotonArray:
    """A class for using the full array of CherenkovPhoton values
    at a series of stages, t, and atmospheric delta values.
    """

    def __init__(self, gg:dict[str,np.ndarray]):
        """Create a CherenkovPhotonArray object from a npz file. The
        npz file should contain the Cherenkov angular distributions
        for a set of stages and delta values. It should also contain
        arrays of the values for t, delta, and theta.

        Parameters:
            npzfile: The input file containing the angular distributions
                and values.

        The npz file should have exactly these keys: "gg_t_delta_theta",
        "t", "delta", and "theta".
        """
        # with as_file(files(f'{npzfile}')) as file:
        #     gg = np.load(file)

        self.gg_t_delta_theta = gg['gg_t_delta_theta']
        self.t = gg['t']
        self.delta = gg['delta']
        self.theta = gg['theta']

    def angular_distribution(self,t,delta):
        it = np.abs(self.t - t).argmin()
        id = np.abs(self.delta - delta).argmin()
        gg = self.gg_t_delta_theta[it,id]
        return gg

@dataclass
class ShowerSignal:
    '''This is a data container for a shower simulation's Cherenkov 
    Photons, arrival times and counting locations.
    '''
    counters: Counters #counters object
    axis: Axis #axis object
    shower: Shower #shower object
    source_points: np.ndarray = field(repr=False) #vectors to axis points
    wavelengths: np.ndarray = field(repr=False) #wavelength of each bin, shape = (N_wavelengths)
    photons: np.ndarray = field(repr=False) #number of photons from each step to each counter, shape = (N_counters, N_wavelengths, N_axis_points)
    times: np.ndarray = field(repr=False) #arrival times of photons from each step to each counter, shape = (N_counters, N_axis_points)
    charged_particles: np.ndarray = field(repr=False)
    depths: np.ndarray = field(repr=False)
    total_photons: np.ndarray = field(repr=False)
    cos_theta: np.ndarray = field(repr=False)

class Signal:
    '''This class calculates the Cherenkov signal from a given shower axis at
    given counters
    '''

    def __init__(self, shower: Shower, axis: Axis, counters: Counters, yield_array: list[MakeYield], gg: dict[str,np.ndarray], y: dict[str,np.ndarray]):
        self.shower = shower
        self.axis = axis
        self.table_file = axis.get_gg_file()
        self.gga = CherenkovPhotonArray(gg)
        self.y_at_lx = y
        self.counters = counters
        self.yield_array = yield_array
        self.t = self.shower.stage(self.axis.X)
        self.t[self.t>14.] = 14.
        self.Nch = self.shower.profile(self.axis.X)
        self.theta = self.axis.theta(axis.vectors, counters)
        self.omega = self.counters.omega(self.axis.vectors)
        # self.ng = self.calculate_ng()
        # self.ng_sum = self.ng.sum(axis = 1)

    def __repr__(self):
        return f"Signal({self.shower.__repr__()}, {self.axis.__repr__()}, {self.counters.__repr__()})"

    def calculate_gg(self):
        '''This funtion returns the interpolated values of gg at a given deltas
        and thetas

        returns:
        the angular distribution values at the desired thetas
        The returned array is of size:
        (# of counters, # of axis points)
        '''
        gg = np.empty_like(self.theta)
        for i in range(gg.shape[1]):
            gg_td = self.gga.angular_distribution(self.t[i], self.axis.delta[i])
            gg[:,i] = np.interp(self.theta[:,i], self.gga.theta, gg_td)
        return gg
    
    @property
    def photon_array_shape(self) -> tuple:
        '''This property is the shape of the outputted photons array.
        '''
        return (self.counters.N_counters,len(self.yield_array),self.axis.r.size)

    # def calculate_gg(self):
    #     '''This funtion returns the interpolated values of gg at a given deltas
    #     and thetas

    #     returns:
    #     the angular distribution values at the desired thetas
    #     The returned array is of size:
    #     (# of counters, # of axis points)
    #     '''
    #     gg = np.empty_like(self.theta)
    #     for i in range(gg.shape[0]):
    #         gg[i] = self.gga.gg_of_t_delta_theta(self.t,self.axis.delta,self.theta[i])
    #     return gg

    def calculate_yield(self, y: MakeYield):
        ''' This function returns the total number of Cherenkov photons emitted
        at a given stage of a shower per all solid angle.

        returns: the total number of photons per all solid angle
        size: (# of axis points)
        '''
        Y = y.y_list(self.t, self.axis.delta)
        return 2. * self.Nch * self.axis.dr * Y

    # def calculate_ng(self):
    #     '''This method returns the number of Cherenkov photons going toward
    #     each counter from every axis bin

    #     The returned array is of size:
    #     # of yield bins, with each entry being on size:
    #     (# of counters, # of axis points)
    #     '''
    #     gg = self.calculate_gg()
    #     ng_array = np.empty_like(self.yield_array, dtype='O')
    #     for i, y in enumerate(self.yield_array):
    #         y.set_yield_at_lX(self.axis.lX)
    #         ng_array[i] = gg * self.calculate_yield(y) * self.omega
    #     return ng_array
    
    def calculate_ng(self) -> np.ndarray:
        '''This method returns the number of Cherenkov photons going toward
        each counter from every axis bin

        The returned array is of size:
        (# of counters, # of yield bins, # of axis points)
        '''
        gg = self.calculate_gg()
        ng_array = np.empty(self.photon_array_shape)
        for i, y in enumerate(self.yield_array):
            y.set_yield_attributes(self.y_at_lx)
            ng_array[:,i,:] = gg * self.calculate_yield(y) * self.omega
        return ng_array

@dataclass
class Simulation:
    shower:Shower
    axis:Axis
    counters:Counters
    y: list[MakeYield]

    def __post_init__(self) -> None:
        self.lXs = np.arange(-6,0)
        self.lX_intervals = list(zip(self.lXs[:-1], self.lXs[1:]))
        self.lX_mids = np.array([np.mean(interval) for interval in self.lX_intervals])
        self.yield_files = {interval:np.load(f'y_t_delta_lX_{interval[0]}_to_{interval[1]}.npz') for interval in self.lX_intervals}
        self.gg_files = {interval:np.load(f'gg_t_delta_theta_lX_{interval[0]}_to_{interval[1]}.npz') for interval in self.lX_intervals}
        self.N_c = self.counters.N_counters
        self.axis.reset_for_profile(self.shower)


    def find_nearest_interval(self, lX: float) -> tuple:
        '''This method returns the start and end points of the lX interval that
        the mesh falls within.
        '''
        index = np.searchsorted(self.lXs[:-1], lX)
        if index == 0:
            return self.lXs[0], self.lXs[1]
        else:
            return self.lXs[index-1], self.lXs[index]
        
    @staticmethod
    def get_attenuated_photons_array(signal: Signal, curved_correction: CurvedAtmCorrection):
        '''This method returns the attenuated number of photons going from each
        step to each counter.

        The returned array is of size:
        # of yield bins, with each entry being on size:
        (# of counters, # of axis points)
        '''
        attenuation = signal.axis.get_attenuation(curved_correction,signal.yield_array)
        fraction_array = attenuation.fraction_passed()
        photons_array = signal.calculate_ng()
        attenuated_photons = np.zeros_like(photons_array)
        for i_a, fractions in enumerate(fraction_array):
            attenuated_photons[:,i_a,:] = photons_array[:,i_a,:] * fractions
        return attenuated_photons

    def get_mesh_signal(self) -> ShowerSignal:
        '''This method returns a ShowerSignal object with the photons calculated
        using mesh sampling.
        '''
        lXs = np.linspace(-6,1,15)
        lX_intervals = list(zip(lXs[:-1], lXs[1:]))
        N_axis_points = self.axis.config.N_IN_RING * self.axis.r.size
        axis_vectors = np.empty((len(lX_intervals), N_axis_points, 3))
        photons_array = np.empty((self.N_c, len(self.y), len(lX_intervals), N_axis_points))
        cQ_array = np.empty((self.N_c, len(self.y), len(lX_intervals), N_axis_points))
        times_array = np.empty((self.N_c, len(lX_intervals), N_axis_points))
        charged_particle_array = np.empty((len(lX_intervals), N_axis_points))
        depth_array = np.empty_like(charged_particle_array)

        #calculate signal at each mesh ring
        for i, lX in enumerate(lX_intervals):
            meshaxis = MeshAxis(lX, self.axis, self.shower)
            meshshower = MeshShower(meshaxis)
            nearest_lX = self.find_nearest_interval(np.mean(lX))

            signal = Signal(meshshower,meshaxis,self.counters,self.y, self.gg_files[nearest_lX], self.yield_files[nearest_lX])
            curved_correction = meshaxis.get_curved_atm_correction(self.counters)

            axis_vectors[i,:] = meshaxis.vectors

            photons_array[:,:,i] = self.get_attenuated_photons_array(signal, curved_correction)

            times_array[:,i] = meshaxis.get_timing(curved_correction).counter_time()
            cQ_array[:,:,i] = curved_correction.cQ[:,np.newaxis,:]

            #also save profile info for charged particles distributed into the rings
            charged_particle_array[i] = meshaxis.nch
            depth_array[i] = meshaxis.meshX
        
        #sum photons at each depth step
        tot_at_X = photons_array.sum(axis=2).sum(axis=1).sum(axis=0).reshape(self.axis.r.size,-1).sum(axis=1)

        #flatten over mesh rings
        photons_array = photons_array.reshape((photons_array.shape[0],photons_array.shape[1],-1))
        cQ_array = cQ_array.reshape((photons_array.shape[0],photons_array.shape[1],-1))
        times_array = times_array.reshape((times_array.shape[0],-1))
        axis_vectors = axis_vectors.reshape((-1,3))    
        charged_particle_array = charged_particle_array.flatten()
        depth_array = depth_array.flatten()
        
        return ShowerSignal(self.counters, 
                            self.axis,
                            self.shower,
                            axis_vectors, 
                            np.array([y.l_mid for y in self.y]),
                            photons_array,
                            times_array,
                            charged_particle_array,
                            depth_array,
                            tot_at_X,
                            cQ_array[:,0,:]) #just take the first wavelength entry, the angles are the same for all

if __name__ =='__main__':
    import cProfile

    shower = MakeGHShower(500,2.e6,0.,70.)
    axis = MakeDownwardAxisFlatPlanarAtm(AxisParams(0,0,0))
    counters = MakeSphericalCounters(np.array([[0.,0.,0.],[1.,1.,1.]]),1.)
    y = Yield(300.,450.).create()
    sim = Simulation(shower,axis,counters,y)
    cProfile.run('sim.get_mesh_signal()')
    # sig = sim.get_mesh_signal()

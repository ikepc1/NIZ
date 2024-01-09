import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from iminuit import Minuit
from iminuit.cost import LeastSquares

from utils import run_multiprocessing
from tyro_fit import tyro, TyroFit
from niche_plane import get_nfits_from_df, NichePlane, NicheFit
from gen_ckv_signals import Event
from process_showers import ProcessEvents
from config import CounterConfig, COUNTER_NO, NAMES, COUNTER_POSITIONS, WAVEFORM_SIZE, TRIGGER_POSITION, NICHE_TIMEBIN_SIZE

def compile_nfit_inputs(df: pd.DataFrame) -> list[list[NicheFit]]:
    '''This function gets a list of lists of niche fits to pass to multiprocessing
    '''
    nfit_list = []
    indices = []
    for i in range(len(df)):
        if has_multiple_triggers(df,i):
            nfit_list.append(get_nfits_from_df(df, i))
            indices.append(i)
    return nfit_list, indices

def has_multiple_triggers(df: pd.DataFrame, index: int) -> bool:
    '''This function checks if more than one counter triggered
    for the event at index.
    '''
    return len(df.loc[index][12:][df.loc[index][12:].notna()]) > 1

def run_dataframe_recon(df: pd.DataFrame) -> None:
    '''This function reconstructs each event in the dataframe.
    '''
    inputs, indices = compile_nfit_inputs(df)
    print('Adding Tyros...')
    tyros = run_multiprocessing(tyro, inputs)
    print('Adding plane fits...')
    pfs = run_multiprocessing(NichePlane, inputs)
    for index, tyr, pf in zip(indices, tyros, pfs):
        df.at[index,'Fit'] = tyr
        df.at[index,'Plane Fit'] = pf
        if tyr.has_contained_core:
            core = tyr.core_estimate
            df.at[index,'corex'] = core[0]
            df.at[index,'corey'] = core[1]
            df.at[index,'corez'] = core[2]

def E(Nmax: float) -> float:
    return Nmax / 1.3e-9

def base_fit_params() -> pd.DataFrame:
    '''This function creates a small dataframe with the fit parameter values to be fed to
    minuit.
    '''
    l = []
    l.append({'name':'Xmax', 'initial_value': 400., 'limits':(3.e2,1.e3), 'fixed': False})
    l.append({'name':'Nmax', 'initial_value': 1.e6, 'limits':(1.e5,1.e9), 'fixed': False})
    l.append({'name':'zenith', 'initial_value': 0., 'limits':(0.,np.pi/2), 'fixed': True})
    l.append({'name':'azimuth', 'initial_value': 0., 'limits':(0.,2*np.pi), 'fixed': True})
    l.append({'name':'corex', 'initial_value': 0., 'limits':(COUNTER_POSITIONS[:,0].min(), COUNTER_POSITIONS[:,0].max()), 'fixed': True})
    l.append({'name':'corey', 'initial_value': 0., 'limits':(COUNTER_POSITIONS[:,1].min(), COUNTER_POSITIONS[:,1].max()), 'fixed': True})
    l.append({'name':'corez', 'initial_value': 0., 'limits':(COUNTER_POSITIONS[:,2].min(), COUNTER_POSITIONS[:,2].max()), 'fixed': True})
    l.append({'name':'X0', 'initial_value': 0., 'limits':(-500.,500.), 'fixed': True})
    l.append({'name':'Lambda', 'initial_value': 70., 'limits':(0.,100.), 'fixed': True})
    df = pd.DataFrame(l)
    df = df.set_index('name')
    return df

def full_wf_times(trigtime: float) -> np.ndarray:
    ''''''
    times_ns = np.arange(NICHE_TIMEBIN_SIZE*WAVEFORM_SIZE, step = NICHE_TIMEBIN_SIZE)
    times_ns -= times_ns[TRIGGER_POSITION]
    times_ns += trigtime
    return times_ns

def get_times_array(nfits: list[NicheFit]) -> np.ndarray:
    ''''''
    trigtimes = np.array([nfit.trigtime() for nfit in nfits])
    trigtimes = np.float64(trigtimes - trigtimes.min())
    return trigtimes

@dataclass
class EventFit:
    '''This is the container for an analysis of a real Niche event.
    '''
    plane_fit: NichePlane
    tyro: TyroFit
    pe: ProcessEvents
    waveform_cushion: int = 100

    def __post_init__(self) -> None:
        self.core = self.tyro.core_estimate
        self.wf_starti = TRIGGER_POSITION - self.waveform_cushion
        self.wf_stopi = TRIGGER_POSITION + self.waveform_cushion
        self.real_nfit_dict = {f.name:f for f in self.plane_fit.counters}
        self.data_pa_array = np.array([f.intsignal for f in self.plane_fit.counters])
        self.real_trigger_names = np.array([f.name for f in self.plane_fit.counters], dtype=str)
        self.biggest_trigger_name = self.real_trigger_names[np.argmax(self.data_pa_array)]
        self.biggest_trigger_time = self.real_nfit_dict[self.biggest_trigger_name].trigtime()
        self.active_counters = np.array(self.pe.cfg.active_counters, dtype=str)
        self.real_signal_array = self.signal_array(self.plane_fit.counters)[:,self.wf_starti:self.wf_stopi]
        self.real_signal_error = np.sqrt(self.real_signal_array)
        self.real_times_array = self.times_array(self.plane_fit.counters)[:,self.wf_starti:self.wf_stopi]
        self.params = self.init_fit_params()

    def init_fit_params(self) -> pd.DataFrame:
        params = base_fit_params()
        if self.plane_fit.phi < 0.:
            self.plane_fit.phi += 2*np.pi
        # Set theta and limits
        params.at['zenith','initial_value'] = self.plane_fit.theta
        params.at['zenith','limits'] = (self.plane_fit.theta, self.plane_fit.theta + .1)
        #set phi and limits
        params.at['azimuth','initial_value'] = self.plane_fit.phi
        params.at['azimuth','limits'] = (self.plane_fit.phi - .2, self.plane_fit.phi + .2)
        #set core
        params.at['corex','initial_value'] = self.core[0]
        params.at['corey','initial_value'] = self.core[1]
        params.at['corez','initial_value'] = self.core[2]
        return params
    
    def get_event(self, parameters: np.ndarray) -> Event:
        return Event(
        E=np.nan,
        Xmax=parameters[0],
        Nmax=parameters[1],
        zenith= parameters[2],
        azimuth= parameters[3],
        corex= parameters[4],
        corey=parameters[5],
        corez=parameters[6],
        X0=parameters[7],
        Lambda=parameters[8]
    )

    def signal_array(self, nfits: list[NicheFit]) -> np.ndarray:
        '''
        '''
        baselines = self.pe.noise.mean(axis=1)
        signals = np.array([np.full(WAVEFORM_SIZE, baseline) for baseline in baselines])
        for nfit in nfits:
            signals[self.active_counters == nfit.name] = nfit.waveform
        return signals
    
    def times_array(self, nfits: list[NicheFit]) -> np.ndarray:
        '''
        '''
        times = np.full((len(self.active_counters),WAVEFORM_SIZE),-1.e5)
        if self.biggest_trigger_name not in [nfit.name for nfit in nfits]:
            return times
        for nfit in nfits:
            trigtime = np.float64(nfit.trigtime() - self.biggest_trigger_time)
            times[self.active_counters == nfit.name] = full_wf_times(trigtime)
        return times
    
    def model(self, times: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        ev = self.get_event(parameters)
        sim_nfits = self.pe.gen_nfits_from_event(ev)
        sim_signal_array = self.signal_array(sim_nfits)
        sim_times_array = self.times_array(sim_nfits)
        output_signal_array = np.zeros_like(times)
        for i, (wf,t) in enumerate(zip(sim_signal_array, sim_times_array)):
            starti = i*2*self.waveform_cushion
            stopi = starti+2*self.waveform_cushion
            ctimes = times[starti:stopi]
            #find trace values in simulated waveforms at the times the real triggers occurred
            if ctimes.max() in t:
                output_signal_array[starti:stopi] = wf[np.searchsorted(t, ctimes)]
            else:
                output_signal_array[starti:stopi] = wf[:2*self.waveform_cushion]
        return output_signal_array.flatten()
    
    def fit(self) -> Minuit:
        names = tuple(self.params.index)
        values = tuple(self.params['initial_value'])
        fixed = tuple(self.params['fixed'])
        limits = tuple(self.params['limits'])
        least_squares_np = LeastSquares(self.real_times_array.flatten(), 
                                        self.real_signal_array.flatten(), 
                                        self.real_signal_error.flatten(), 
                                        self.model)
        m = Minuit(least_squares_np, values, name=names)
        for name, fix, lim in zip(names, fixed, limits):
            m.fixed[name] = fix
            m.limits[name] = lim
        return m

# @dataclass
# class EventFit:
#     '''This is the container for an analysis of a real Niche event.
#     '''
#     plane_fit: NichePlane
#     tyro: TyroFit
#     pe: ProcessEvents
#     waveform_cushion: int = 100

#     def __post_init__(self) -> None:
#         self.core = self.tyro.core_estimate
#         self.data_pa_array = np.array([f.intsignal for f in self.plane_fit.counters])
#         self.real_wf_array = np.array([f.waveform[TRIGGER_POSITION-self.waveform_cushion:TRIGGER_POSITION+self.waveform_cushion] for f in self.plane_fit.counters]).flatten()
#         self.real_nfit_dict = {f.name:f for f in self.plane_fit.counters}
#         self.real_trigger_ids = np.array([COUNTER_NO[f.name] for f in self.plane_fit.counters])
#         self.real_trigger_names = np.array([f.name for f in self.plane_fit.counters], dtype=str)
#         self.real_wf_error = np.sqrt(self.real_wf_array)
#         self.trigtimes = get_times_array(self.plane_fit.counters)
#         self.real_wf_times = self.construct_times()
#         self.first_trigger_name = self.real_trigger_names[np.argmin(self.trigtimes)]
#         self.biggest_trigger_name = self.real_trigger_names[np.argmax(self.data_pa_array)]
#         if self.plane_fit.phi < 0.:
#             self.plane_fit.phi += 2*np.pi
#         self.params = self.init_fit_params()
    
#     def get_waveform_times(self, trigtime: float) -> np.ndarray:
#         ''''''
#         times_ns = np.arange(NICHE_TIMEBIN_SIZE*2*self.waveform_cushion, step = NICHE_TIMEBIN_SIZE)
#         times_ns -= times_ns[self.waveform_cushion]
#         times_ns += trigtime
#         return times_ns

#     def construct_times(self) -> np.ndarray:
#         ''''''
#         times_array = np.array([self.get_waveform_times(time) for time in self.trigtimes])
#         return times_array.flatten()

#     def init_fit_params(self) -> pd.DataFrame:
#         params = base_fit_params()
#         # Set theta and limits
#         params.at['zenith','initial_value'] = self.plane_fit.theta
#         params.at['zenith','limits'] = (self.plane_fit.theta, self.plane_fit.theta + .1)
#         #set phi and limits
#         params.at['azimuth','initial_value'] = self.plane_fit.phi
#         params.at['azimuth','limits'] = (self.plane_fit.phi - .2, self.plane_fit.phi + .2)
#         #set core
#         params.at['corex','initial_value'] = self.core[0]
#         params.at['corey','initial_value'] = self.core[1]
#         params.at['corez','initial_value'] = self.core[2]
#         return params
    
#     def get_event(self, parameters: np.ndarray) -> Event:
#         return Event(
#         E=np.nan,
#         Xmax=parameters[0],
#         Nmax=parameters[1],
#         zenith= parameters[2],
#         azimuth= parameters[3],
#         corex= parameters[4],
#         corey=parameters[5],
#         corez=parameters[6],
#         X0=parameters[7],
#         Lambda=parameters[8]
#     )

#     # def get_times_dict(self, sim_nfits: dict[str,NicheFit]) -> dict[str,np.ndarray]:
#     #     '''Have the zero time be when the biggest real counter triggered, return a dictionary
#     #     of the time arrays.
#     #     '''
#     #     trigtimes = np.array([sim_nfits[name].trigtime() for name in sim_nfits])
#     #     trigtimes = np.float64(trigtimes - sim_nfits[self.biggest_trigger_name].trigtime())
#     #     return {name:full_wf_times(time) for name, time in zip(sim_nfits, trigtimes)}
    
#     def get_times_dict(self, sim_nfits: dict[str,NicheFit]) -> dict[str,np.ndarray]:
#         '''Have the zero time be when the first real counter triggered, return a dictionary
#         of the time arrays.
#         '''
#         trigtimes = np.array([sim_nfits[name].trigtime() for name in sim_nfits])
#         trigtimes = np.float64(trigtimes - sim_nfits[self.first_trigger_name].trigtime())
#         return {name:full_wf_times(time) for name, time in zip(sim_nfits, trigtimes)}

#     # def get_times_dict(self, sim_nfits: dict[str,NicheFit]) -> dict[str,np.ndarray]:
#     #     '''Have the zero time be when the first real counter triggered, return a dictionary
#     #     of the time arrays.
#     #     '''
#     #     trigtimes = np.array([sim_nfits[name].trigtime() for name in sim_nfits])
#     #     trigtimes = np.float64(trigtimes - sim_nfits[self.first_trigger_name].trigtime())
#     #     times_dict = {name:full_wf_times(time) for name, time in zip(sim_nfits, trigtimes)}
#     #     for name in self.pe.cfg.active_counters:
#     #         if name not in times_dict:
#     #             times_dict[name] = np.zeros(WAVEFORM_SIZE)
#     #     return times_dict

#     # def get_wf_dict(self, sim_nfits: dict[str,NicheFit]) -> dict[str,np.ndarray]:
#     #     '''Dictionary of the waveforms for all the triggered counters, zeros if it's an active counter that
#     #     didn't trigger.
#     #     '''
#     #     wf_dict = {name:sim_nfits[name].waveform for name in sim_nfits}
#     #     for name in self.pe.cfg.active_counters:
#     #         if name not in wf_dict:
#     #             wf_dict[name] = np.zeros(WAVEFORM_SIZE)
#     #     return wf_dict

#     def model(self, times: np.ndarray, parameters: np.ndarray) -> np.ndarray:
#         ev = self.get_event(parameters)
#         sim_nfits = self.pe.gen_nfits_from_event(ev)
#         sim_nfit_dict = {f.name:f for f in sim_nfits}
#         traces_at_times = np.zeros((len(self.real_trigger_ids),2*self.waveform_cushion))

#         #if the earliest trigger didn't occur, return zeroes b/c otherwise times can't be matched
#         # if self.biggest_trigger_name not in sim_nfit_dict:
#         #     return traces_at_times.flatten()
        
#         if self.first_trigger_name not in sim_nfit_dict:
#             return traces_at_times.flatten()
        
#         #if only one counter triggers nothing can be done
#         if len(sim_nfits) == 1:
#             return traces_at_times.flatten()

#         #otherwise the times can have the same zero, get the times for each waveform entry in each counter
#         sim_times_dict = self.get_times_dict(sim_nfit_dict)

#         for i,name in enumerate(self.real_trigger_names):
#             if name in sim_nfit_dict:
#                 starti = i*2*self.waveform_cushion
#                 ctimes = times[starti:starti+2*self.waveform_cushion]

#                 #find trace values in simulated waveforms at the times the real triggers occurred
#                 if ctimes.max() in sim_times_dict[name]:
#                     traces_at_times[i] = sim_nfit_dict[name].waveform[np.searchsorted(sim_times_dict[name], ctimes)]
#         return traces_at_times.flatten()
        
#     def fit(self) -> Minuit:
#         names = tuple(self.params.index)
#         values = tuple(self.params['initial_value'])
#         fixed = tuple(self.params['fixed'])
#         limits = tuple(self.params['limits'])
#         least_squares_np = LeastSquares(self.real_wf_times, self.real_wf_array, self.real_wf_error, self.model)
#         m = Minuit(least_squares_np, values, name=names)
#         for name, fix, lim in zip(names, fixed, limits):
#             m.fixed[name] = fix
#             m.limits[name] = lim
#         return m

if __name__ == '__main__':
    # from datafiles import *
    from utils import plot_event, plot_generator, get_data_files, preceding_noise_file
    data_date_and_time = '20190504034237'
    data_files = get_data_files(data_date_and_time)
    noise_files = [preceding_noise_file(f) for f in data_files]
    cfg = CounterConfig(data_files, noise_files)
    ns = pd.read_pickle('sample_ns_df.pkl')
    g = plot_generator(ns)

    pe = ProcessEvents(cfg, frozen_noise=True)
    ev = Event(0.,500,1.e7,np.deg2rad(40.),np.deg2rad(315.), 450., -660.,-25.,0,70)
    sim_nfits = ProcessEvents(cfg, frozen_noise=True).gen_nfits_from_event(ev)
    pf = NichePlane(sim_nfits)
    ty = tyro(sim_nfits)
    ef = EventFit(pf,ty,pe)

    # ev_id = 6
    # ef = EventFit(ns.loc[ev_id,'Plane Fit'], ns.loc[ev_id,'Fit'],ProcessEvents(cfg, frozen_noise=True))
    print('starting gradient descent...')
    migrad=ef.fit().migrad()
    pars = np.array([par.value for par in migrad.params])
    sim_ev = ef.get_event(pars)
    plot_event(tyro(ef.pe.gen_nfits_from_event(sim_ev)), 'sim')
    plot_event(ty, 'real')


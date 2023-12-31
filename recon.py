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
    l.append({'name':'Xmax', 'initial_value': 500., 'limits':(1.e2,1.e3), 'fixed': False})
    l.append({'name':'Nmax', 'initial_value': 1.e7, 'limits':(1.e4,1.e8), 'fixed': False})
    l.append({'name':'zenith', 'initial_value': 0., 'limits':(0.,np.pi/2), 'fixed': False})
    l.append({'name':'azimuth', 'initial_value': 0., 'limits':(0.,2*np.pi), 'fixed': False})
    l.append({'name':'corex', 'initial_value': 0., 'limits':(COUNTER_POSITIONS[:,0].min(), COUNTER_POSITIONS[:,0].max()), 'fixed': False})
    l.append({'name':'corey', 'initial_value': 0., 'limits':(COUNTER_POSITIONS[:,1].min(), COUNTER_POSITIONS[:,1].max()), 'fixed': False})
    l.append({'name':'corez', 'initial_value': 0., 'limits':(COUNTER_POSITIONS[:,2].min(), COUNTER_POSITIONS[:,2].max()), 'fixed': True})
    l.append({'name':'X0', 'initial_value': 0., 'limits':(-500.,500.), 'fixed': True})
    l.append({'name':'Lambda', 'initial_value': 70., 'limits':(0.,100.), 'fixed': True})
    df = pd.DataFrame(l)
    df = df.set_index('name')
    return df

def construct_times(trigtime: float) -> np.ndarray:
    ''''''
    times_ns = np.arange(NICHE_TIMEBIN_SIZE*WAVEFORM_SIZE, step = NICHE_TIMEBIN_SIZE)
    times_ns -= times_ns[TRIGGER_POSITION]
    times_ns += trigtime
    return times_ns

def get_times_array(nfits: list[NicheFit]) -> np.ndarray:
    ''''''
    trigtimes = np.array([nfit.trigtime() for nfit in nfits])
    trigtimes = np.float64(trigtimes - trigtimes.min())
    times_array = np.empty((len(nfits), WAVEFORM_SIZE))
    for i,time in enumerate(trigtimes):
        times_array[i] = construct_times(time)
    return times_array

def get_long_fadc_array(nfits: list[NicheFit], cushion: int = 50) -> np.ndarray:
    ''''''
    fadcs = []
    indices = []
    for nfit in nfits:
        fadcs.extend(nfit.waveform[TRIGGER_POSITION-cushion:TRIGGER_POSITION+cushion])


@dataclass
class EventFit:
    '''This is the container for an analysis of a real Niche event.
    '''
    plane_fit: NichePlane
    tyro: TyroFit
    pe: ProcessEvents

    def __post_init__(self) -> None:
        self.core = self.tyro.core_estimate
        self.data_pa_array = np.array([f.intsignal for f in self.plane_fit.counters])
        self.real_nfit_dict = {f.name:f for f in self.plane_fit.counters}
        self.real_trigger_ids = np.array([COUNTER_NO[f.name] for f in self.plane_fit.counters])
        self.real_pa_error = np.sqrt(self.data_pa_array)
        if self.plane_fit.phi < 0.:
            self.plane_fit.phi += 2*np.pi
        self.params = self.init_fit_params()

    def init_fit_params(self) -> pd.DataFrame:
        params = base_fit_params()
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

    def model(self, det_ids: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        '''This is the fit model to be passed to iminuit.
        '''
        ev = self.get_event(parameters)
        sim_nfits = self.pe.gen_nfits_from_event(ev)
        sim_nfit_dict = {f.name:f for f in sim_nfits}
        sim_pa_array = np.zeros_like(det_ids)
        for i,id in enumerate(det_ids):
            if NAMES[id] in sim_nfit_dict:
                sim_pa_array[i] = sim_nfit_dict[NAMES[id]].intsignal
        return sim_pa_array

    def fit(self) -> Minuit:
        names = tuple(self.params.index)
        values = tuple(self.params['initial_value'])
        fixed = tuple(self.params['fixed'])
        limits = tuple(self.params['limits'])
        least_squares_np = LeastSquares(self.real_trigger_ids, self.data_pa_array, self.real_pa_error, self.model)
        m = Minuit(least_squares_np, values, name=names)
        for name, fix, lim in zip(names, fixed, limits):
            m.fixed[name] = fix
            m.limits[name] = lim
        return m

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
    ev = Event(0.,500,5.e6,np.deg2rad(40.),np.deg2rad(315.), 450., -660.,-25.,0,70)
    sim_nfits = ProcessEvents(cfg, frozen_noise=True).gen_nfits_from_event(ev)
    pf = NichePlane(sim_nfits)
    ty = tyro(sim_nfits)
    ef = EventFit(pf,ty,pe)

    # ev_id = 6
    # ef = EventFit(ns.loc[ev_id,'Plane Fit'], ns.loc[ev_id,'Fit'],ProcessEvents(cfg, frozen_noise=True))
    migrad=ef.fit().migrad()
    pars = np.array([par.value for par in migrad.params])
    sim_ev = ef.get_event(pars)
    plot_event(tyro(ef.pe.gen_nfits_from_event(sim_ev)), 'sim')
    plot_event(ty, 'real')


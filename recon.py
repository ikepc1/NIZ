import pandas as pd
import numpy as np
from dataclasses import dataclass
from iminuit import Minuit
from iminuit.cost import LeastSquares

from utils import run_multiprocessing
from tyro_fit import tyro, TyroFit
from niche_plane import get_nfits_from_df, NichePlane, NicheFit
from gen_ckv_signals import Event
from process_showers import ProcessEvents
from config import CounterConfig, COUNTER_NO, NAMES

# def run_multiprocessing(func: Callable[[object],object], inputs: list[object], chunksize = 250) -> list[object]:
#     '''This function maps a function to imap with the use of context managers
#     and a progress bar.
#     '''
#     results = []
#     with alive_bar(len(inputs)) as bar:
#         with Pool(cpu_count()) as p:
#             for result in p.imap(func, inputs, chunksize=chunksize):
#                 results.append(result)
#                 bar()
#     return results

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
        

def initial_guess(tyro: TyroFit, plane: NichePlane) -> Event:
    '''This function takes the centroid and plane fit and constructs an initial
    guess for the shower parameters.
    '''
    core = tyro.core_estimate
    return Event(
        E=1.e15,
        Xmax=500.,
        Nmax=1.e6,
        zenith=plane.theta,
        azimuth=plane.phi,
        corex=core[0],
        corey=core[1],
        corez=core[2],
        X0=0.,
        Lambda=70.
    )

def E(Nmax: float) -> float:
    return Nmax / 1.3e-9

@dataclass
class EventFit:
    '''This is the container for an analysis of a real Niche event.
    '''
    plane_fit: NichePlane
    tyro: TyroFit
    pe: ProcessEvents
    E: float = 1.e15
    X0: float = 0.
    Lambda: float = 70.
    dXmax: float = 1
    dNmax: float = 1.e4

    def __post_init__(self) -> None:
        self.core = self.tyro.core_estimate
        self.data_pa_array = np.array([f.intsignal for f in self.plane_fit.counters])
        self.real_nfit_dict = {f.name:f for f in self.plane_fit.counters}
        self.real_trigger_ids = np.array([COUNTER_NO[f.name] for f in self.plane_fit.counters])
        self.real_pa_error = np.sqrt(self.data_pa_array)

    @property
    def error(self) -> np.ndarray:
        return np.sqrt(self.data_pa_array)

    def get_event(self, Xmax: float, Nmax: float) -> Event:
        if self.plane_fit.phi < 0.:
            self.plane_fit.phi += 2*np.pi
        return Event(
        E=self.E,
        Xmax=Xmax,
        Nmax=Nmax,
        zenith= self.plane_fit.theta,
        azimuth= self.plane_fit.phi,
        corex=self.core[0],
        corey=self.core[1],
        corez=self.core[2],
        X0=self.X0,
        Lambda=self.Lambda
    )

    def model(self, det_ids: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        '''This is the fit model to be passed to iminuit.
        '''
        ev = self.get_event(parameters[0], parameters[1])
        sim_nfits = self.pe.gen_nfits_from_event(ev)
        sim_nfit_dict = {f.name:f for f in sim_nfits}
        sim_pa_array = np.zeros_like(det_ids)
        for i,id in enumerate(det_ids):
            if NAMES[id] in sim_nfit_dict:
                sim_pa_array[i] = sim_nfit_dict[NAMES[id]].intsignal
        return sim_pa_array

    def minimize(self, init_xmax: float, init_nmax: float):
        least_squares_np = LeastSquares(self.real_trigger_ids, self.data_pa_array, self.real_pa_error, self.model)
        m = Minuit(least_squares_np, (init_xmax, init_nmax), name=('xmax', 'nmax'))
        m.limits = [(3.e2, 1.e3), (1.e5, 1.e7)]
        return m.migrad()

    # def cost(self, input: tuple) -> float:
    #     '''This function is the chi squared statistic for the result of a shower
    #     simulation.
    #     '''
    #     Xmax = input[0]
    #     Nmax = input[1]
    #     ev = self.get_event(Xmax, Nmax)
    #     sim_nfits = self.pe.gen_nfits_from_event(ev)
    #     sim_nfit_dict = {f.name:f for f in sim_nfits}
    #     sim_pa_array = np.zeros_like(self.data_pa_array)
    #     for i,name in enumerate(self.real_nfit_dict):
    #         if name in sim_nfit_dict:
    #             sim_pa_array[i] = sim_nfit_dict[name].intsignal
    #     return np.sum((self.data_pa_array - sim_pa_array)**2)
    
    # def gradient(self, Xmax: float, Nmax: float) -> tuple[float]:
    #     inputs = [(Xmax + self.dXmax/2, Nmax),
    #               (Xmax - self.dXmax/2, Nmax),
    #               (Xmax, Nmax + self.dNmax/2),
    #               (Xmax, Nmax - self.dNmax/2)]
    #     costs = run_multiprocessing(self.cost, inputs, chunksize=1)
    #     dc_dXmax = (costs[0] - costs[1])/self.dXmax
    #     dc_dNmax = (costs[2] - costs[3])/self.dNmax
    #     return dc_dXmax, dc_dNmax#, self.cost((Xmax, Nmax))
    
    # def minimize(self, Xmax: float, Nmax: float, n_iter=1000, tolerance=100.) -> tuple[float]:
    #     for i in range(n_iter):
    #         print(f'Epoch {i}')
    #         grad = self.gradient(Xmax, Nmax)
    #         print(Xmax, Nmax, grad)
    #         if np.abs(grad[0]) < 100 and np.abs(grad[1]) <1.e-2:
    #             return Xmax, Nmax
    #         Xmax -= 1.e-4 * grad[0]
    #         Nmax -= 1.e4 * grad[1]
    #     return Xmax, Nmax
    
if __name__ == '__main__':
    # from datafiles import *
    from utils import plot_event, plot_generator, get_data_files, preceding_noise_file
    data_date_and_time = '20190504034237'
    data_files = get_data_files(data_date_and_time)
    noise_files = [preceding_noise_file(f) for f in data_files]
    cfg = CounterConfig(data_files, noise_files)
    ns = pd.read_pickle('sample_ns_df.pkl')
    g = plot_generator(ns)

    ev_id = 17
    ef = EventFit(ns.loc[ev_id,'Plane Fit'], ns.loc[ev_id,'Fit'],ProcessEvents(cfg, frozen_noise=True))
    migrad=ef.minimize(400.,1.e6)
    sim_ev = ef.get_event(migrad.params[0].value,migrad.params[1].value)
    plot_event(tyro(ef.pe.gen_nfits_from_event(sim_ev)), 'sim')
    plot_event(ns.loc[ev_id,'Fit'], 'real')


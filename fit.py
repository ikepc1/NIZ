from iminuit import Minuit
from iminuit.cost import LeastSquares
import numpy as np
from dataclasses import dataclass
import pandas as pd

from utils import run_multiprocessing
from tyro_fit import tyro, TyroFit
from niche_plane import NichePlane, NicheFit
from gen_ckv_signals import Event
from process_showers import ProcessEvents
from config import CounterConfig, COUNTER_NO, NAMES, COUNTER_POSITIONS, WAVEFORM_SIZE, TRIGGER_POSITION, NICHE_TIMEBIN_SIZE
from trigger import TriggerSim
from noise import read_noise_file

def E(Nmax: float) -> float:
    return Nmax / 1.3e-9

def base_fit_params() -> pd.DataFrame:
    '''This function creates a small dataframe with the fit parameter values to be fed to
    minuit.
    '''
    l = []
    l.append({'name':'Xmax', 'initial_value': 500., 'limits':(300.,600.), 'error': 50, 'fixed': True})
    l.append({'name':'Nmax', 'initial_value': 1.e6, 'limits':(1.e5,1.1e6), 'error': 1.e4, 'fixed': True})
    l.append({'name':'zenith', 'initial_value': np.deg2rad(40.), 'limits':(0.,np.pi/2), 'error': np.deg2rad(1.), 'fixed': True})
    l.append({'name':'azimuth', 'initial_value':np.deg2rad(315.), 'limits':(0.,2*np.pi), 'error': np.deg2rad(1.), 'fixed': True})
    l.append({'name':'corex', 'initial_value': 450., 'limits':(COUNTER_POSITIONS[:,0].min(), COUNTER_POSITIONS[:,0].max()), 'error': 1., 'fixed': True})
    l.append({'name':'corey', 'initial_value': -660., 'limits':(COUNTER_POSITIONS[:,1].min(), COUNTER_POSITIONS[:,1].max()), 'error': 1., 'fixed': True})
    l.append({'name':'corez', 'initial_value': -25., 'limits':(COUNTER_POSITIONS[:,2].min(), COUNTER_POSITIONS[:,2].max()),  'error': 1.,'fixed': True})
    l.append({'name':'X0', 'initial_value': 0., 'limits':(-500.,500.), 'error': 1., 'fixed': True})
    l.append({'name':'Lambda', 'initial_value': 70., 'limits':(0.,100.), 'error': 1., 'fixed': True})
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

    def __post_init__(self) -> None:
        self.core = self.tyro.core_estimate
        self.active_counters = np.array(self.pe.cfg.active_counters, dtype=str)
        self.data_pa_array = np.array([f.intsignal for f in self.plane_fit.counters])
        self.real_trigger_names = np.array([f.name for f in self.plane_fit.counters], dtype=str)
        self.biggest_trigger_name = self.real_trigger_names[np.argmax(self.data_pa_array)]
        self.real_output, self.real_output_error = self.get_output(self.plane_fit.counters)
        self.params = self.init_fit_params()

    def init_fit_params(self) -> pd.DataFrame:
        params = base_fit_params()
        if self.plane_fit.phi < 0.:
            self.plane_fit.phi += 2*np.pi
        #Set theta and limits
        params.at['zenith','initial_value'] = self.plane_fit.theta
        params.at['zenith','limits'] = (self.plane_fit.theta - .2, self.plane_fit.theta + .2)
        #set phi and limits
        params.at['azimuth','initial_value'] = self.plane_fit.phi
        params.at['azimuth','limits'] = (self.plane_fit.phi - .2, self.plane_fit.phi + .2)
        #set core
        params.at['corex','initial_value'] = self.core[0]
        params.at['corex','limits'] = (self.core[0]-20., self.core[0]+50.)
        params.at['corey','initial_value'] = self.core[1]
        params.at['corey','limits'] = (self.core[1]-20., self.core[1]+50.)
        params.at['corez','initial_value'] = self.core[2]
        return params
    
    @staticmethod
    def get_event(parameters: np.ndarray) -> Event:
        nmax = parameters[1]
        return Event(
        E=E(nmax),
        Xmax=parameters[0],
        Nmax=nmax,
        zenith= parameters[2],
        azimuth= parameters[3],
        corex= parameters[4],
        corey=parameters[5],
        corez=parameters[6],
        X0=parameters[7],
        Lambda=parameters[8]
    )

    @property
    def input_indices(self) -> np.ndarray:
        return np.arange(len(self.pe.cfg.active_counters)*4)

    def get_pas(self, nfits: list[NicheFit]) -> tuple[np.ndarray]:
        '''
        '''
        pas = np.zeros(len(self.active_counters))
        paes = np.full(len(self.active_counters), 1.)
        for nfit in nfits:
            pas[self.active_counters == nfit.name] = nfit.intsignal
            paes[self.active_counters == nfit.name] = nfit.eintsignal
        return pas, paes
    
    @property
    def real_nfit_dict(self) -> dict[str,NicheFit]:
        return {f.name:f for f in self.plane_fit.counters}

    def get_trigtimes(self, nfits: list[NicheFit]) -> np.ndarray:
        nfit_dict = {f.name:f for f in nfits}
        ts = np.full(len(self.active_counters), 1.e-4)
        if self.biggest_trigger_name not in nfit_dict:
            return ts
        biggest_trigger_time = nfit_dict[self.biggest_trigger_name].trigtime()
        for i, name in enumerate(self.active_counters):
            if name in nfit_dict:
                trigtime = nfit_dict[name].trigtime()
                td = trigtime - biggest_trigger_time
                ts[i] = td.astype('float64')
        return ts
    
    def get_pulse_widths(self, nfits: list[NicheFit]) -> tuple[np.ndarray]:
        '''
        '''
        ws = np.zeros(len(self.active_counters))
        wes = np.full(len(self.active_counters), 2.5)
        for nfit in nfits:
            ws[self.active_counters == nfit.name] = nfit.risetime + nfit.falltime
            wes[self.active_counters == nfit.name] = np.sqrt(nfit.erisetime**2 + nfit.efalltime**2)
        return ws, wes
    
    def get_peaktimes(self, nfits: list[NicheFit]) -> tuple[np.ndarray]:
        ps = np.zeros(len(self.active_counters))
        pes = np.full(len(self.active_counters), 2.5)
        for nfit in nfits:
            ps[self.active_counters == nfit.name] = nfit.peaktime
            pes[self.active_counters == nfit.name] = nfit.epeaktime
        return ps, pes
    
    def get_output(self, nfits: list[NicheFit]) -> tuple[np.ndarray]:
        # output = np.empty_like(self.input_indices, dtype=np.float64)
        # output_error = np.empty_like(self.input_indices, dtype=np.float64)
        pas, paes = self.get_pas(nfits)
        ts = self.get_trigtimes(nfits)
        tes = np.full(len(self.active_counters), 2.5)
        ws, wes = self.get_pulse_widths(nfits)
        pts, ptes = self.get_peaktimes(nfits)
        output = np.hstack((pas,ts,ws,pts))
        output_error = np.hstack((paes,tes,wes,ptes))
        # output[:pas.size] = pas
        # output[pas.size:pas.size+pas.size] = ts
        # output[pas.size+pas.size:] = ws
        # output_error[:pas.size] = paes
        # output_error[pas.size:pas.size+pas.size] = tes
        # output_error[pas.size+pas.size:] = wes
        return output, output_error

    def cost(self, parameters: np.ndarray) -> float:
        initial_parameters = self.params['initial_value'].to_numpy()
        initial_parameters[:len(parameters)] = parameters
        ev = self.get_event(initial_parameters)
        print(ev)
        sim_nfits = self.pe.gen_nfits_from_event(ev)
        output, output_error = self.get_output(sim_nfits)
        cost = ((self.real_output - output)**2/self.real_output_error**2).sum()
        print(f'Chi square = {cost}')
        return cost
    
    def log10cost(self, parameters: np.ndarray) -> float:
        cost = np.log10(self.cost(parameters))
        return cost

    def model(self, indices: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        ev = self.get_event(parameters)
        sim_nfits = self.pe.gen_nfits_from_event(ev)
        output, _ = self.get_output(sim_nfits)
        return output

    def fit(self) -> Minuit:
        names = tuple(self.params.index)
        values = tuple(self.params['initial_value'])
        fixed = tuple(self.params['fixed'])
        limits = tuple(self.params['limits'])
        errors = tuple(self.params['error'])
        least_squares_np = LeastSquares(self.input_indices, 
                                        self.real_output, 
                                        self.real_output_error, 
                                        self.model,
                                        verbose=1)
        m = Minuit(least_squares_np, values, name=names)
        # m = Minuit(self.cost, values, name=names)
        for name, fix, lim, err in zip(names, fixed, limits, errors):
            m.fixed[name] = fix
            m.limits[name] = lim
            m.errors[name] = err
        return m
    
    @staticmethod
    def get_params(m: Minuit) -> np.ndarray:
        return np.array([par.value for par in m.params])

def initial_scan(m: Minuit, parname: str, ncall: int = 10) -> Minuit:
    m.fixed[parname] = False
    m.scan(ncall=ncall)
    m.fixed[parname] = True
    return m

def reset_limits(m: Minuit, parname: str, delta: float) -> Minuit:
    value = m.params[parname].value
    m.limits[parname] = (value - delta, value + delta)
    return m

def fit_procedure(m: Minuit) -> Minuit:
    m = initial_scan(m, 'Nmax', 20)
    m = reset_limits(m, 'Nmax', 2.e5)
    m = initial_scan(m,'zenith')
    m = reset_limits(m, 'zenith', .05)
    m = initial_scan(m,'corex')
    m = reset_limits(m, 'corex', 5)
    m = initial_scan(m,'corey')
    m = reset_limits(m, 'corey', 5)
    m = initial_scan(m, 'Xmax')
    m = reset_limits(m, 'Xmax', 50)
    for par in ['Xmax','Nmax','zenith']:
        m.fixed[par] = False
    m.scan(ncall=30)
    return m

def do_fit(real_nfits: list[NicheFit]) -> tuple[float]:
    pf = NichePlane(sim_nfits)
    ty = tyro(sim_nfits)


if __name__ == '__main__':
    # from datafiles import *
    import matplotlib.pyplot as plt
    plt.ion()
    from utils import plot_event, plot_generator, get_data_files, preceding_noise_file
    data_date_and_time = '20190504034237'
    data_files = get_data_files(data_date_and_time)
    noise_files = [preceding_noise_file(f) for f in data_files]
    cfg = CounterConfig(data_files, noise_files)
    # ns = pd.read_pickle('sample_ns_df.pkl')
    # g = plot_generator(ns)

    
    ev = Event(0.,500,1.e6,np.deg2rad(40.),np.deg2rad(315.), 450., -660.,-25.,0,70)
    sim_nfits = ProcessEvents(cfg, frozen_noise=True).gen_nfits_from_event(ev)
    pf = NichePlane(sim_nfits)
    ty = tyro(sim_nfits)

    xmax = []
    nmax = []
    zenith = []
    corex = []
    corey = []

    for i in range(10):
        pe = ProcessEvents(cfg, frozen_noise=True)
        ef = EventFit(pf,ty,pe)

        # print('starting param scan...')
        m=ef.fit()
        m = fit_procedure(m)
        # migrad.scan(ncall=10)
        # print('starting gradient descent...')
        # migrad.migrad()
        pars = np.array([par.value for par in m.params])
        xmax.append(pars[0])
        nmax.append(pars[1])
        zenith.append(pars[2])
        corex.append(pars[4])
        corey.append(pars[5])
        sim_ev = ef.get_event(pars)
        plot_event(tyro(ef.pe.gen_nfits_from_event(sim_ev)), f'sim {i}')
    plot_event(ty, f'real')

    # pe = ProcessEvents(cfg, frozen_noise=True)
    # ef = EventFit(pf,ty,pe)
    # init_pars = ef.params['initial_value'].tolist()

    # ngrid = 11

    # xmaxs = np.linspace(480,520,ngrid)
    # nmaxs = np.linspace(.9e6,1.1e6,ngrid)
    # x,n = np.meshgrid(xmaxs,nmaxs)
    # xn = list(zip(x.flatten(),n.flatten()))
    # xncosts = np.array(run_multiprocessing(ef.cost,xn,1))
    # plt.figure()
    # plt.contourf(xmaxs,nmaxs,xncosts.reshape(ngrid,ngrid),xncosts.min() + np.linspace(1,10,10)**2)
    # plt.xlabel('xmax')
    # plt.ylabel('nmax')
    # plt.colorbar(label='chi_square')

    # theta = np.deg2rad(np.linspace(35,45,ngrid))
    # phi = np.deg2rad(np.linspace(310,320,ngrid))
    # t,p = np.meshgrid(theta, phi)
    # parlist = [(init_pars[0], init_pars[1], tn, pn) for tn, pn in zip(t.flatten(),p.flatten())]
    # tpcosts = np.array(run_multiprocessing(ef.cost,parlist,1))
    # plt.figure()
    # plt.contourf(theta,phi,tpcosts.reshape(ngrid,ngrid),np.linspace(1,10,10)**2)
    # plt.xlabel('theta')
    # plt.ylabel('phi')
    # plt.colorbar(label='chi_square')

    # corex = np.linspace(450-50,450+50,ngrid)
    # corey = np.linspace(-660-50,-660+50,ngrid)
    # x,y = np.meshgrid(corex, corey)
    # parlist = [(init_pars[0], init_pars[1], init_pars[2], init_pars[3], xc, yc) for xc, yc in zip(x.flatten(),y.flatten())]
    # xycosts = np.array(run_multiprocessing(ef.cost,parlist,1))
    # plt.figure()
    # plt.contourf(corex,corey,xycosts.reshape(ngrid,ngrid),np.linspace(1,50,50)**2)
    # plt.xlabel('corex')
    # plt.ylabel('corey')
    # plt.colorbar(label='chi_square')


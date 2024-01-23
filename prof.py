import cProfile
from fit import *
from utils import get_data_files, preceding_noise_file

data_date_and_time = '20190504034237'
data_files = get_data_files(data_date_and_time)
noise_files = [preceding_noise_file(f) for f in data_files]
cfg = CounterConfig(data_files, noise_files)

pars = [np.log(500),np.log(1.e6),np.deg2rad(40.),np.deg2rad(315.), 450., -660.,-25.,0,70]
ev = get_event(pars)
sim_nfits = ProcessEvents(cfg, frozen_noise=True).gen_nfits_from_event(ev)
ef = EventFit(sim_nfits, cfg)

pars = [np.log(500),np.log(1.e6),np.deg2rad(40.),np.deg2rad(315.), 450., -660.,-25.,0,70]

cProfile.run('ef.get_output(pars)')
import cProfile
# from fit import *
from main import *

# data_date_and_time = '20190504034237'
# data_files = get_data_files(data_date_and_time)
# noise_files = [preceding_noise_file(f) for f in data_files]
# cfg = CounterConfig(data_files, noise_files)

# pars = [500.,2.e6,np.deg2rad(40.),np.deg2rad(315.), 450., -660.,-25.,0,70]
# ev = BasicParams.get_event(pars)
# real_nfits = ProcessEvents(cfg, frozen_noise=True).gen_nfits_from_event(ev)
# pt = AllSamples(real_nfits, BasicParams, cfg)

# cProfile.run('pt.model([0],*pars)')

cProfile.run("main('20230610070512',100)")
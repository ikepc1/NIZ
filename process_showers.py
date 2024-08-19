from multiprocessing import Pool, cpu_count
import pandas as pd
from alive_progress import alive_bar
import numpy as np

from trigger import NicheTriggers, gen_niche_trigger, generate_background, generate_zeros, get_thresholds
from gen_ckv_signals import get_ckv, Event, read_in_corsika, ckv_from_tilefile
from config import TRIGGER_WIDTH
from counter_config import CounterConfig
from niche_fit import NicheFit

class ProcessEvents:
    '''This class is responsible for mapping the detector configuration to
    the eventprocessing procedure.
    '''
    def __init__(self, cfg: CounterConfig, frozen_noise: bool = False, zero_noise: bool = False) -> None:
        self.cfg = cfg
        self.frozen_noise = frozen_noise
        self.zero_noise = zero_noise
        if frozen_noise:
            self.noise = generate_background(self.cfg.noise_open_files)
        elif zero_noise:
            self.noise = generate_zeros(self.cfg.noise_open_files)
            self.thresholds = get_thresholds(self.cfg.noise_open_files)

    def process_event(self, evt: Event) -> NicheTriggers:
        '''This function takes an event, generates a cherenkov signal'''
        if self.frozen_noise or self.zero_noise:
            return gen_niche_trigger(get_ckv(evt, self.cfg), self.noise)
        else:
            return gen_niche_trigger(get_ckv(evt, self.cfg), generate_background(self.cfg.noise_open_files))
    
    def process_ei_event(self, file: str) -> NicheTriggers:
        '''This function takes an event, generates a cherenkov signal'''
        if self.frozen_noise or self.zero_noise:
            return gen_niche_trigger(read_in_corsika(file, self.cfg), self.noise)
        else:
            return gen_niche_trigger(read_in_corsika(file, self.cfg), generate_background(self.cfg.noise_open_files))

    def process_tf_event(self, ei, shift) -> NicheTriggers:
        '''This function takes an event, generates a cherenkov signal'''
        if self.frozen_noise or self.zero_noise:
            return gen_niche_trigger(ckv_from_tilefile(ei, self.cfg, shift), self.noise)
        else:
            return gen_niche_trigger(ckv_from_tilefile(ei, self.cfg, shift), generate_background(self.cfg.noise_open_files))

    def pseudotrigger(self, nfit: NicheFit) -> bool:
        '''This method computes whether the waveform exceeds the specific counter's
        threshold.
        '''
        sums = np.cumsum(nfit.waveform)
        means = (sums[TRIGGER_WIDTH:] - sums[:-TRIGGER_WIDTH])/TRIGGER_WIDTH
        return (means > self.thresholds[nfit.name]).any()

    def gen_nfits_from_event(self, evt: Event) -> list[NicheFit]:
        trig = self.process_event(evt)
        nfits = [trig.cts[name].to_nfit() for name in trig.names]
        if self.zero_noise:
            return [f for f in nfits if self.pseudotrigger(f)]
        else:
            return nfits
        
    def gen_nfits_from_ei(self, file: str) -> list[NicheFit]:
        trig = self.process_ei_event(file)
        nfits = [trig.cts[name].to_nfit() for name in trig.names]
        if self.zero_noise:
            return [f for f in nfits if self.pseudotrigger(f)]
        else:
            return nfits
        
    def gen_nfits_from_tf(self, ei, shift) -> list[NicheFit]:
        trig = self.process_tf_event(ei, shift)
        nfits = [trig.cts[name].to_nfit() for name in trig.names]
        if self.zero_noise:
            return [f for f in nfits if self.pseudotrigger(f)]
        else:
            return nfits

def add_nfits_to_df(trig: NicheTriggers, df: pd.DataFrame, ev_index: int) -> None:
    '''This function adds counter triggers to the shower dataframe.
    '''
    for name in trig.names:
        df.loc[ev_index, name] = trig.cts[name].to_nfit()

def add_triggers_to_dataframe(shower_df: pd.DataFrame, cfg: CounterConfig) -> pd.DataFrame:
    '''This function adds to a dataframe: niche fitted CHASM signals from a dataframe of 
    shower parameters.
    '''
    evts = [Event(*row[:10]) for row in shower_df.itertuples(index=False, name=None)]
    pe = ProcessEvents(cfg)
    with alive_bar(len(evts)) as bar:
        with Pool(cpu_count()) as p:
            for i,trig in enumerate(p.imap(pe.process_event, evts, chunksize=1)):
                if trig.trigs.any():
                    print('trigger')
                    add_nfits_to_df(trig,shower_df,i)
                bar()
    return shower_df
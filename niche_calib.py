from niche_raw import NicheRaw
from read_flasher_logs import ampl_at_time
from utils import read_niche_file

from functools import cached_property
import numpy as np
from pathlib import Path

class CalibPulse(NicheRaw):
    '''This class is the container for the square calibration pulses.
    '''

    def __init__(self,niche_raw_obj: NicheRaw):
        for attr in dir(niche_raw_obj):
            if not attr.startswith('__'):
                setattr(self,attr,getattr(niche_raw_obj,attr))
        self.baseline = np.mean(self.waveform[:500])
        self.baseline_error = np.var(self.waveform[:500])

    @cached_property
    def min_pulse_level(self) -> float:
        _,bins = np.histogram(self.waveform)
        return bins[-3]

    # @cached_property
    # def start_rise(self) -> int:
    #     peak = self.waveform.argmax()
    #     before_reversed = self.waveform[:peak][::-1]
    #     under_level = (before_reversed<self.min_pulse_level)
    #     if not under_level.any():
    #         return 0
    #     n_samples_before = under_level.argmax()
    #     istart = peak - n_samples_before + 2
    #     return istart
    
    # @cached_property
    # def end_fall(self) -> int:
    #     peak = self.waveform.argmax()
    #     after = self.waveform[peak:]
    #     n_samples_after = (after<self.min_pulse_level).argmax()
    #     iend = peak + n_samples_after - 2
    #     return iend
    
    @cached_property
    def start_rise(self) -> int:
        return 524 + 10

    @cached_property
    def end_fall(self) -> int:
        return 610

    @cached_property
    def pulse(self) -> np.ndarray:
        return self.waveform[self.start_rise:self.end_fall] - self.baseline

    @cached_property
    def pulse_mean(self) -> float:
        if len(self.pulse) == 0:
            return 0.
        else:
            return self.pulse.mean()
        
    @cached_property
    def flash_ampl(self) -> float:
        return ampl_at_time(self.trigtime())

def read_calib_files(counter: str) -> list[CalibPulse]:
    calib_directory = Path('calib') / counter
    nraws = []
    for file in calib_directory.iterdir():
        if file.name.endswith('.bin'):
            nraws.extend(read_niche_file(file))
    return nraws


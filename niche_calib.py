from niche_raw import NicheRaw

from functools import cached_property
import numpy as np

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

    @cached_property
    def start_rise(self) -> int:
        peak = self.waveform.argmax()
        before_reversed = self.waveform[:peak][::-1]
        n_samples_before = (before_reversed<self.min_pulse_level).argmax()
        istart = peak - n_samples_before + 1
        return istart
    
    @cached_property
    def end_fall(self) -> int:
        peak = self.waveform.argmax()
        after = self.waveform[peak:]
        n_samples_after = (after<self.min_pulse_level).argmax()
        iend = peak + n_samples_after - 1
        return iend
    
    @cached_property
    def pulse(self) -> np.ndarray:
        return self.waveform[self.start_rise:self.end_fall] - self.baseline


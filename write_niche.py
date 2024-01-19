import numpy as np
from dataclasses import dataclass, field
import struct as st
from datetime import datetime

from niche_raw import NicheRaw
from niche_fit import NicheFit
from config import TRIGGER_VARIANCE, TRIGGER_WIDTH

@dataclass
class CounterInfo:
    '''This is a container for the pth and info for a counter.
    '''
    pressure: float = 0.
    temp: float = 0.
    trigPosition: float = 500.
    trigWidth: float = float(TRIGGER_WIDTH)
    bias: float = 0.
    gain: float = 0.
    trigVariance: float = TRIGGER_VARIANCE
    dacHV: float = 0.
    adcHV: float = 0.

    @property
    def pth(self) -> list[float]:
        return [self.pressure, self.temp]
    
    @property
    def info(self) -> list[float]:
        return [self.trigPosition,
                self.trigWidth,
                self.bias,
                self.gain,
                self.trigVariance,
                self.dacHV,
                self.adcHV]

@dataclass
class CounterTrigger:
    '''This is the container for a niche trigger on one counter.
    '''
    name: str
    waveform: np.ndarray = field(repr=False)
    times: np.ndarray = field(repr=False)
    datebytes: datetime = field(repr=False)

    @property
    def trigger_time_counter(self) -> float:
        '''This is the time when the counter triggered.
        '''
        return self.times[500]
        # return self.times.astype('uint32')[500]

    def counter_bytes(self) -> bytes:
        '''This method returns a bytearray of the hexadecimal 16 bit integer 
        representing the time in nanoseconds of the trigger.
        '''
        return st.pack('>i',int(self.trigger_time_counter))
    
    def waveform_bytes(self) -> bytes:
        '''This function returns the bytes of each entry in the waveform.
        '''
        return self.waveform.astype('>H').tobytes()
    
    def write_trigger_bytes(self) -> bytearray:
        '''This function returns the byte buffer for a trigger.
        '''
        bb = bytearray()
        bb.extend(self.datebytes)
        bb.extend(self.counter_bytes())
        bb.extend(self.waveform_bytes())
        return bb
    
    def to_nfit(self) -> NicheRaw:
        '''This method returns a NicheRaw object with the simulated data.
        '''
        ci = CounterInfo()
        return NicheFit(NicheRaw(self.name,ci.pth,ci.info,self.write_trigger_bytes()))
        

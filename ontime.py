from pathlib import Path
import numpy as np

from config import RAW_DATA_PATH
from counter_config import CounterConfig
from utils import read_niche_file

def ontime(cfg: CounterConfig) -> np.timedelta64:
    '''This function calculates the ontime for a specific data part
    as the difference between the first and last trigger.
    '''
    nraws = []
    for file in cfg.data_files:
        nraws.extend(read_niche_file(file))
    timestamps = np.array([nraw.trigtime() for nraw in nraws], dtype=np.datetime64)
    timestamps = timestamps[~np.isnan(timestamps)]
    timedelta = timestamps.max() - timestamps.min()
    return timedelta
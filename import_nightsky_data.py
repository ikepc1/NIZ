import pandas as pd
from pathlib import Path
import numpy as np

from niche_raw import NicheRaw
from niche_fit import NicheFit
from niche_plane import NichePlane
from niche_bin import bin_to_raw
from tyro_fit import tyro
from config import CounterConfig

def get_events_from_datafile(file: Path) -> list[NicheFit]:
    '''This function grabs the events in a niche nightsky datafile.
    '''
    with file.open('rb') as open_file:
        nraws = list(set(bin_to_raw(open_file.read(), file.parent.name, retfit=False)))
    return [NicheFit(nraw) for nraw in nraws]

def get_events(cfg: CounterConfig) -> dict[str, list[NicheRaw]]:
    '''This function creates a dictionary of nraw objects for all the events
    in each counter.
    '''
    event_dict = {}
    for data_file in cfg.data_files:
        event_dict[data_file.parent.name] = get_events_from_datafile(data_file)
    return event_dict

def nraw_list(events_dict: dict[str, list[NicheRaw]]) -> list[NicheRaw]:
    '''This function converts the dictionary of nraws to an array containing all
    of them for each counter.
    '''
    nraws = []
    for counter_name in events_dict:
        nraws.extend(events_dict[counter_name])
    return nraws

def match_times(events_dict: dict[str, list[NicheRaw]], ns_time_interval: int = 1000) -> list[np.ndarray]:
    '''This function takes the dictionary of all triggers for a night, and returns a 
    list of dictionaries where each entry is a time match.
    '''
    #declare max difference as numpy timedelta
    max_timedelta = np.timedelta64(ns_time_interval, 'ns')

    #make an array of all nicheraw events for a night
    nraws = nraw_list(events_dict)
    nraw_array = np.array(nraws, dtype = 'O')

    #get timestamp as numpy datetimes for every event
    ts_array = np.array([nraw.trigtime() for nraw in nraws])

    #find events whose timestamps are within max_timedelta of each other
    matches = []
    for ts in ts_array:
        timedelta_array = np.abs(ts - ts_array)
        matches.append(nraw_array[timedelta_array < max_timedelta])

    #remove duplicates
    return matches

def empty_row(cfg: CounterConfig) -> dict:
    '''This function returns a dictionary which is an empty row for the df.
    '''
    row = {}
    cols = ['E',
            'Xmax',     
            'Nmax',
            'zenith',
            'azimuth',
            'corex',
            'corey',
            'corez',
            'X0',
            'Lambda',
            'Fit',
            'Plane Fit']
    for counter in cfg.active_counters:
        cols.append(counter)
    for col in cols:
        row[col] = np.nan
    return row

def init_niche_nightsky_df(cfg: CounterConfig) -> pd.DataFrame:
    '''This function creates a dataframe where the matches are stored.
    '''
    events = get_events(cfg)
    matches = match_times(events)
    er = empty_row(cfg)
    rows = []
    for match in matches:
        row = er
        for nraw in match:
            row[nraw.name] = nraw
        if len(match) > 0:
            row['Plane Fit'] = NichePlane(list(match))
        rows.append(row)
    return pd.DataFrame(rows)
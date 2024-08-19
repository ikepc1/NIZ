import pandas as pd
from pathlib import Path
import numpy as np

from niche_raw import NicheRaw
from niche_fit import NicheFit
from niche_plane import NichePlane
from tyro_fit import tyro
from utils import run_multiprocessing, save_df, read_niche_file, get_data_files
from config import RAW_DATA_PATH, NIGHTSKY_DF_PATH
from counter_config import CounterConfig, init_config

def make_nfit(nraw: NicheRaw) -> NicheFit:
    return NicheFit(nraw)

def get_events_from_datafile(file: Path) -> list[NicheFit]:
    '''This function grabs the events in a niche nightsky datafile.
    '''
    nraws = read_niche_file(file)
    nfits = run_multiprocessing(NicheFit, nraws)
    return nfits
    # return nraws

def get_events(cfg: CounterConfig) -> dict[str, list[NicheRaw]]:
    '''This function creates a dictionary of nraw objects for all the events
    in each counter.
    '''
    event_dict = {}
    for data_file in cfg.data_files:
        # print(f'Getting pulse fits for {data_file.parent.name}')
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

def match_times(events_dict: dict[str, list[NicheRaw]], multiplicity: int = 5, ns_time_interval: int = 1000) -> list[np.ndarray]:
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
    ids = np.arange(len(nraws))
    # matches = []
    match_ids = []
    for ts in ts_array:
        timedelta_array = np.abs(ts - ts_array)
        mask = timedelta_array < max_timedelta
        # m = nraw_array[mask]
        id = ids[mask]
        if len(id) >= multiplicity:
            # matches.append(m)
            match_ids.append(tuple(id))

    #remove duplicates
    id_set = list(set(match_ids))
    matches = [nraw_array[np.array(i)] for i in id_set]
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
            'Plane_Fit',
            'weather']
    row['config'] = cfg
    for counter in cfg.active_counters:
        cols.append(counter)
    for col in cols:
        row[col] = np.nan
    return row

def get_weather(event: list[NicheFit], weather: dict) -> str:
    '''This function matches the timestamp of an event with the weather code
    recorded nearest to that time. (numpy array quacks like a list)
    '''
    event_time = np.min([nraw.trigtime() for nraw in event])
    return weather['codes'][np.argmin(np.abs(weather['times'] - event_time))]

def init_niche_nightsky_df(cfg: CounterConfig) -> pd.DataFrame:
    '''This function creates a dataframe where the matches are stored.
    '''
    weather = np.load('weather.npz')
    events = get_events(cfg)
    matches = match_times(events)
    rows = []
    for match in matches:
        row = empty_row(cfg)
        for nraw in match:
            row[nraw.name] = nraw
        row['Fit'] = tyro(list(match))
        row['Plane_Fit'] = NichePlane(list(match))
        row['weather'] = get_weather(match.tolist(), weather)
        rows.append(row)
    return pd.DataFrame(rows)

def process_night(night: Path) -> list[pd.DataFrame]:
    '''This function takes a NICHE night directory and time matches the events,
    appending them to a dataframe.
    '''
    alldata = get_data_files(night.name)
    allnsdata = [file for file in alldata if (file.name.endswith('.bin') and file.name[-5].isnumeric())]
    data_times = list(set([file.name[:-4] for file in allnsdata]))
    df_list = []
    for time in data_times:
        # ns_data_part = [file for file in allnsdata if file.name[:-4] == time]
        # noise_files = [preceding_noise_file(file) for file in ns_data_part]
        # cfg = CounterConfig(ns_data_part, noise_files)
        cfg = init_config(time)
        df = init_niche_nightsky_df(cfg)
        if not df.empty:
            save_df(df,time + '.pkl',NIGHTSKY_DF_PATH)
            df_list.append(df)
    return df_list

if __name__ == '__main__':
    path = Path(RAW_DATA_PATH)
    nights = [p for p in path.iterdir()] #if int(p.name) == 20200228]
    event_ncounters = []
    for nightpath in nights:
        print(nightpath.name)
        l = process_night(nightpath)
        if l:
            for df in l:
                event_ncounters.extend([len(row[1][row[1].notna()]) for row in df.iterrows()])
from datetime import datetime
from pathlib import Path
import numpy as np
from config import RAW_DATA_PATH

def isolate_digits(string: str) -> str:
    '''This function removes non integer characters from a string.
    '''
    return ''.join(c for c in string if c.isdigit())

def datetime_from_filename(file: Path) -> datetime:
    '''This function creates a datetime object from the date and time in the 
    filename of a niche datafile.
    '''
    return datetime.strptime(isolate_digits(file.name), '%Y%m%d%H%M%S')

def preceding_noise_file(datafile: Path) -> Path:
    '''This function finds the noise file immediately preceding the input data
    file.
    '''
    data_time = datetime_from_filename(datafile)
    noise_files = [file for file in datafile.parent.glob('*.bg.bin')]
    noise_times = np.array([datetime_from_filename(file) for file in noise_files]).astype(datetime)
    deltas = data_time - noise_times
    return noise_files[np.abs(deltas).argmin()]

def get_file_ts(file: Path) -> np.datetime64:
    '''This function gets the timestamp from the filename.
    '''
    y = file.name[:4]
    m = file.name[4:6]
    d = file.name[6:8]
    H = file.name[8:10]
    M = file.name[10:12]
    S = file.name[12:14]
    return np.datetime64(y+'-'+m+'-'+d+' '+H+':'+M+':'+S)

def get_data_files(timestr: str) -> list[str]:
    '''This function finds the corresponding data parts for each
    counter at a given time. The second wildcard in the glob makes
    this work for either noise or nightsky files.
    '''
    data_directory_path = Path(RAW_DATA_PATH) / timestr[:8]
    return sorted(data_directory_path.glob(f'*/{timestr}*.bin'))




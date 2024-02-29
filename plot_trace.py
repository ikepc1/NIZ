import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys

from noise import read_niche_file, trigger_times

CALIB_DATA_DIR = Path('/home/isaac/niche_calib_data/')

def get_date() -> str:
    '''This function gets the current date and returns it as a string.
    '''
    return datetime.now().strftime("%Y%m%d")

def copy_from_niche_laptop(counter_name: str) -> None:
    '''This function copies over all new calib data files from the niche laptop
    calib directory.
    '''
    target_dir = CALIB_DATA_DIR / counter_name
    cmd = f"sshpass -p 'niCherenkov' rsync --progress -avz -e ssh niche@192.168.183.10:/home/niche/calib/{counter_name}/* {str(target_dir)}/"
    os.system(cmd)

def get_recent_file(counter: str, date: str) -> Path:
    '''This function returns a path to the newest niche calib file.
    '''
    dir = CALIB_DATA_DIR / counter
    file_list = sorted(dir.glob(f'{date}*.bin'))
    return file_list[-1]

def plot_trace(file: Path, trace_no: int=0) -> None:
    '''This function plots a trace in a niche file.
    '''
    trace_array = read_niche_file(file)
    plt.figure()
    plt.plot(trace_array[trace_no])
    plt.ylabel('fadc counts')
    plt.xlabel('bin index')


if __name__ == '__main__':
    plt.ion()
    counter = str(sys.argv[1])
    date = str(sys.argv[2])

    copy_from_niche_laptop(counter)
    rf = get_recent_file(counter, date)
    plot_trace(rf)
    tt = trigger_times(rf)
    print(np.sort(tt))

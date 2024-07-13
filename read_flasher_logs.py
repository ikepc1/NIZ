from pathlib import Path
from datetime import datetime, timezone
import numpy as np

def parse_date(date: str) -> np.datetime64:
    '''This function takes a date from the logs and converts it to utc time then
    returns this as a numpy datetime.
    '''
    dt = datetime.strptime(date,"%m-%d-%y_%H:%M:%S").astimezone(timezone.utc)
    return np.datetime64(dt.replace(tzinfo=None))

def read_log_file(file: Path) -> tuple[np.ndarray]:
    '''This function reads the log file and returns a dictionary of the time each 
    amplitude was changed.
    '''
    timestamps = []
    ampls = []
    with open(file) as open_file:
        for line in open_file:
            if line[18:22] == 'AMPL':
                timestamps.append(parse_date(line[:17]))
                ampls.append(float(line[23:]))
    return np.array(timestamps), np.array(ampls)

def find_log_file(time: np.datetime64) -> Path:
    '''This function finds the log file for the given time.
    '''
    logdir = Path('calib/logs')
    logfiles = np.array([f for f in logdir.iterdir()], dtype='O')
    logtimes = np.array([parse_date(f.name[4:21]) for f in logdir.iterdir()])
    logfiles = logfiles[logtimes.argsort()]
    logtimes.sort()
    index = (logtimes > time).argmax() - 1
    return logfiles[index]

def ampl_at_time(time: np.datetime64) -> float:
    '''This function returns the amplitude the flasher had at a given time.
    '''
    logfile = find_log_file(time)
    timestamps, ampls = read_log_file(logfile)
    timestamps += np.timedelta64(5, 's')
    # timestamps += np.timedelta64(250, 'ms')
    index = (time<timestamps).argmax() - 1
    ampl = ampls[index]
    if np.abs(timestamps - time).min() < np.timedelta64(500, 'ms'):
        return 0.
    return ampl

if __name__ == '__main__':
    f = Path('calib/logs/log_04-12-24_20:45:48.txt')
    ts,amps = read_log_file(f)
    
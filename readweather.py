from pathlib import Path
import xml.etree.ElementTree as ET
import os
import numpy as np

from utils import isolate_digits

WEATHER_LOG_PATH = Path('/home/isaac/weather')

def read_weather(logpath: Path) -> dict[np.datetime64,str]:
    '''This function scrapes the weather codes from an fd logfile.
    '''
    y = logpath.name[1:5]
    m = logpath.name[6:8]
    d = logpath.name[9:11]
    tree = ET.parse(logpath)
    root = tree.getroot()
    times = [time.attrib['time'] for time in root.iter('weather')]
    times = [np.datetime64(f'{y}-{m}-{d}T{time}') for time in times]
    codes = [isolate_digits(time.text)[:7] for time in root.iter('weather')]
    dict = {t:c for t,c in zip(times,codes)}
    return dict

if __name__ == '__main__':
    alldict = {}
    for log in WEATHER_LOG_PATH.iterdir():
        alldict = {**alldict, **read_weather(log)}
    times = np.array([key for key in alldict],dtype=np.datetime64)
    codes = np.array([alldict[key] for key in alldict],dtype='str')
    np.savez('weather.npz',times=times,codes=codes)

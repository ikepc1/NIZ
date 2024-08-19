from pathlib import Path
import pandas as pd

def particle_maxi(file: Path) -> int:
    '''This function returns the final line of the particle data from
    a corsika longitudinal file.
    '''
    with file.open('r') as f:
        lines = f.readlines()
    for i,line in enumerate(lines[2:]):
        if not line[5].isnumeric():
            return int(i + 2)
    raise Exception('Probably not a longfile...')

def read_longfile(file: Path) -> pd.DataFrame:
    '''This function copies the particle information from a corsika 
    longitudinal file into a pandas dataframe.
    '''
    with file.open('r') as f:
        lines = f.readlines()
    colnames = lines[1].split()
    dictlist = []
    for line in lines[2:particle_maxi(file)]:
        dictlist.append({n:float(val) for n,val in zip(colnames,line.split())})
    return pd.DataFrame(dictlist)

def get_shower_angles(file: Path) -> tuple[float,float]:
    '''This function reads the output file for the angles.
    '''
    with file.open('r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith(' PRIMARY ANGLES ARE:'):
            words = line.split()
            return float(words[5]), float(words[9])
    raise Exception('Probably not an outfile...')
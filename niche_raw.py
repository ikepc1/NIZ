import numpy as np
import struct as st
from datetime import datetime

class NicheRaw:
    """
    Class for j-niche waveform data
    """

    names = {0:'curie',
             1:'dirac',
             2:'einstein',
             3:'feynman',
             4:'meitner',
             5:'newton',
             6:'noether',
             7:'rutherford',
             8:'wu',
             9:'yukawa',
             10: 'bardeen',
             11: 'bell',
             12: 'rossi',
             13: 'rubin'}
    
    counter_no = {'curie':     0,
                  'dirac':     1,
                  'einstein':  2,
                  'feynman':   3,
                  'meitner':   4,
                  'newton':    5,
                  'noether':   6,
                  'rutherford':7,
                  'wu':        8,
                  'yukawa':    9,
                  'bardeen':  10,
                  'bell':     11,
                  'rossi':    12,
                  'rubin':    13}
    
    positions = { 0:(392.8,-711.4,-24),
                  1:(574.2,-607.3,-26),
                  2:(489.2,-514.0,-23),
                  3:(577.4,-720.0,-27),
                  4:(489.6,-821.0,-27),
                  5:(379.5,-619.1,-26),
                  6:(389.1,-508.5,-25),
                  7:(489.2,-615.1,-29),
                  8:(290.4,-508.3,-26),
                  9:(483.0,-709.7,-25),
                 10:(283.3,-708.1,-24),
                 11:(592.0,-823.6,-23),
                 12:(286.0,-610.4,-21),
                 13:(397.3,-808.0,-26)}

    WAVELEN = 1024
    
    def __init__(s,name,pth,info,buf):
        s.name = name
        s.number = s.counter_no[name]
        if not pth is None:
            s.press = pth[0]
            s.temp  = pth[1]
        else:
            s.press = 0.
            s.temp  = 0.
        if not info is None:
            s.trigPosition = info[0]
            s.trigWidth    = info[1]
            s.bias         = info[2]
            s.gain         = info[3]
            s.trigVariance = info[4]
            s.dacHV        = info[5]
            s.adcHV        = info[6]
        else: 
            s.trigPosition = 0
            s.trigWidth    = 0
            s.bias         = 0.
            s.gain         = 0.
            s.trigVariance = 0
            s.dacHV        = 0
            s.adcHV        = 0.
        
        pos = 0
        s.date = st.unpack('14s',buf[pos:pos+14])[0]
        pos += 14
        s.counter = st.unpack('>i',buf[pos:pos+4])[0]
        pos += 4
        fmt = '>'+str(s.WAVELEN)+'H'
        s.waveform = np.array(st.unpack(fmt,buf[pos:pos+s.WAVELEN*2]),dtype='H')
        pos += s.WAVELEN*2

    def __str__(s):
        # output = "%14s_%08x"%(str(s.date,'ascii'),s.counter)
        output = "%14s_%08x"%(s.date,s.counter)
        return output

    def __repr__(s):
        return s.__str__()
    
    def trigtime(self) -> np.datetime64:
        '''This property is the time the counter triggered.
        '''
        y = self.__str__()[2:6]
        m = self.__str__()[6:8]
        d = self.__str__()[8:10]
        H = self.__str__()[10:12]
        M = self.__str__()[12:14]
        S = self.__str__()[14:16]
        ns = self.counter * 5
        if (y+m+d+H+M+S+str(ns)).isnumeric():
            try:
                return np.datetime64(y+'-'+m+'-'+d+' '+H+':'+M+':'+S) + np.timedelta64(ns, 'ns')
            except:
                return np.datetime64()
        else:
            return np.datetime64()

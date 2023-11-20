import numpy as np
import struct as st
from niche_raw import *
from niche_fit import *

TMSTLEN = 19
DATALEN = 2067
NBUNCH   = 256
BUNCHLEN = TMSTLEN + NBUNCH*DATALEN

def bin_to_raw(bb,name,retfit=False):
    """Return a list of niche_raw (and niche_fit) objects from
    a byte buffer bb"""
    nbb = len(bb)
    pos = 0
    dataFormat = st.unpack('i',bb[pos:pos+4])[0]
    pos += 4
    pth = None
    info = None
    if dataFormat >= 2:
        pth = st.unpack('3f',bb[pos:pos+3*4])
        pos += 3*4
        if dataFormat >= 3:
            # 32 words of information, 7 used
            info = st.unpack('7f',bb[pos:pos+7*4])
            pos += 7*4
            pos += (32-7)*4 # junk
            if dataFormat >= 4:
                local_IP = st.unpack('16s',bb[pos:pos+16])
                pos += 16
                cntr_name = st.unpack('16s',bb[pos:pos+16])
                pos += 16

    niche_list = []
    while pos < nbb:
        if pos+BUNCHLEN <= nbb and dataFormat!=0:
            bunch = np.frombuffer(bb,dtype='B',count=BUNCHLEN,offset=pos)
            nb = BUNCHLEN
        else:
            nb = nbb-pos
            bunch = np.frombuffer(bb,dtype='B',count=nb,offset=pos)
        pos += nb
        lb = len(bunch)

        ppos = 0
        if dataFormat != 0:
            fmt = str(TMSTLEN)+'s'
            timestamp = st.unpack(fmt,bunch[ppos:ppos+TMSTLEN])[0]
            ppos += TMSTLEN
        while ppos < nb:
            try:
                buf = np.frombuffer(bunch,dtype='B',count=DATALEN,offset=ppos)
                ppos += DATALEN
            except ValueError:
                break
            begining = st.unpack('3s',buf[:3])[0]
            try:
                begini = int(begining)
            except ValueError:
                continue
            # if begini < 199 or begini > 201:
            #     continue
            ######
            nraw = NicheRaw(name,pth,info,buf)
            if retfit:
                nfit = NicheFit(nraw)
                niche_list.append(nfit)
            else:
                niche_list.append(nraw)
            # niche_list.append(nraw)
    return niche_list

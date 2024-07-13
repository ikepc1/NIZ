import dstpy
import numpy as np

UVLED_TO_CAMERA_CENTER = 1.237 #meters


def read_dst_file_fraw1(file: str) -> np.ndarray:
    dstpy.addFile(file)
    dstpy.nextEvent(0)
    npe_list = []
    while dstpy.nextEvent():
        if len(dstpy.fraw1.m_fadc[0]) != 320:
            continue
        npes = np.empty((256,100))
        for it in range(256):
            npes[it] = np.array([ord(dstpy.fraw1.m_fadc[0][it][si]) for si in range(100)])
        ped = npes[:,:40].mean(axis = 1)
        npes = (npes.T - ped).T
        npes[npes<1.] = 0.
        npe_list.append(npes)
    dstpy.clearFiles()
    return np.array(npe_list)


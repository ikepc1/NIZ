import numpy as np

ROWSTARTS = np.arange(16) * 16
ROW_IDS = np.arange(16)

def row(ids: np.ndarray) -> np.ndarray:
    rows = np.empty_like(ids, dtype='int')
    for i, idx in enumerate(ids):
        rows[i] = ROW_IDS[idx>=ROWSTARTS][-1]
    return rows

def column(ids: np.ndarray) -> np.ndarray:
    r = row(ids)
    rowstarts = ROWSTARTS[r]
    return ids - rowstarts

def id_to_pos(id: np.ndarray) -> np.ndarray:
    '''This Function maps the tube id to its distance from the  
    center of the face of the camera.
    '''
    rows = row(id).astype('float')
    columns = column(id).astype('float')
    rows -= 7.5
    columns -= 7.5
    distance = np.sqrt(rows**2 + columns**2)
    return distance


def id_to_r(id: np.ndarray) -> np.ndarray:
    '''This function maps the tube id to how far its center is from
    the flasher.
    '''
    tube_positions = id_to_pos(id)
    tube_distances = np.sqrt(UVLED_TO_CAMERA_CENTER**2 + tube_positions**2)
    return tube_distances

def id_to_area(id: np.ndarray) -> np.ndarray:
    '''This function maps the tube id to its effective area as seen by the 
    flasher diffuser.
    '''
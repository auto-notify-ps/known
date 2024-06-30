import numpy as np
import cv2

__all__ = [
    'DTYPE',
    'VALID_CHANNELS',
    'DEFAULT_FILL',

    'create',
    'save',
    'load',
    'flip',
    'rotate',
    'fill',
    'region_fill',
    'region_copy',
    'extend',

]

DTYPE =             np.uint8
VALID_CHANNELS =    set([1, 3, 4])
DEFAULT_FILL =      255

# returns a new array
def create(h, w, c) -> None:
    # creates a new ndarray of specified height, width and channels with datatype specified in class (no of channels can be 1, 3 or 4)
    assert c in VALID_CHANNELS, f'[{c}] is not a valid number of channels, expecting {VALID_CHANNELS}'
    return np.zeros((int(h), int(w), int(c)), dtype=DTYPE) 

def save(pix, path):  return cv2.imwrite(path, pix)

# returns a new array
def load(path): 
    img =  cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.ndim==3:     pix = img.astype(DTYPE)
    elif img.ndim==2:   pix = np.expand_dims(img.astype(DTYPE), -1)
    else:               raise ValueError(f'expecting 2/3-D array but got {img.ndim}-D')
    return pix

# returns a view of the array
def flip(pix, horizontal=False, vertical=False): return pix[(slice(None, None, -1) if vertical else slice(None, None, None)), (slice(None, None, -1) if horizontal else slice(None, None, None)), :]

# returns a view of the array
def rotate(pix, clockwise=True):  return np.swapaxes(pix, 0, 1)[:, ::-1, :] if clockwise else np.swapaxes(pix, 0, 1)[::-1,:,:]

# in-place method
def fill(pix, i_row:int, i_col:int, n_rows:int, n_cols:int, color:tuple, channel=None): 
    # fills an area (region) of the image on all channels or a particular channel with values provided in color
    # i_row, i_col are the starting row and col
    # n_row, n_col are the number of rows and cols
    # if channel is provided, then color is assumed to be integer
    # otherwise color should be a tuple of the form (brga in 4 channel), (brg in 3 channel), (intensity in 1 channel)
    if channel is None: 
        ic = slice(None, None, None)
        color = [DEFAULT_FILL for _ in range(pix.shape[-1])] if color is None else color[0:pix.shape[-1]] # truncate to number of channels 
    else:
        ic = abs(int(channel)) % pix.shape[-1]
        color = DEFAULT_FILL if color is None else abs(int(color)) % 256
    if (n_rows is ...) or (n_rows is None): 
        if (n_cols is ...) or (n_cols is None):     pix[i_row:,             i_col:,             ic] =  color
        else:                                       pix[i_row:,             i_col:i_col+n_cols, ic] =  color
    else: 
        if (n_cols is ...) or (n_cols is None):     pix[i_row:i_row+n_rows, i_col:,             ic] =  color
        else:                                       pix[i_row:i_row+n_rows, i_col:i_col+n_cols, ic] =  color

# in-place method
def region_fill(pix, start_row:int, start_col:int, n_rows:int, n_cols:int, color:tuple):  fill(pix, start_row, start_col, n_rows, n_cols, color, channel=None)

# in-place method
def region_copy(pix_from, start_row_from, start_col_from, n_rows, n_cols, pix_to, start_row_to, start_col_to): pix_to[start_row_to:start_row_to+n_rows, start_col_to:start_col_to+n_cols,  :] = pix_from[start_row_from:start_row_from+n_rows, start_col_from:start_col_from+n_cols, :]

# returns a new array
def extend(pix, north, south, east, west, filler=None):
    # extends an image on all four sides (specified by number of pixels in north, south, east, west)
    new = create(pix.shape[0] + (abs(north) + abs(south)), pix.shape[1] + (abs(east) + abs(west)), pix.shape[-1])
    region_fill(new, 0, 0, None, None, filler)
    region_copy(
        pix_from = pix,   
        start_row_from = 0,
        start_col_from = 0,           
        n_rows = pix.shape[0],
        n_cols = pix.shape[1],
        pix_to = new,     
        start_row_to = abs(north),
        start_col_to = abs(west))
    return new


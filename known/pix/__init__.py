
__doc__=r"""pix module"""

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
def create(h:int, w:int, c:int):
    # creates a new ndarray of specified height, width and channels with datatype specified in class (no of channels can be 1, 3 or 4)
    assert c in VALID_CHANNELS, f'[{c}] is not a valid number of channels, expecting {VALID_CHANNELS}'
    return np.zeros((h, w, c), dtype=DTYPE) 

def save(pix, path):  return cv2.imwrite(path, pix)

# returns a new array
def load(path): 
    img =  cv2.imread(path, cv2.IMREAD_UNCHANGED) #<--- can throw error
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
    # otherwise color should be a tuple of the form (bgra in 4 channel), (bgr in 3 channel), (intensity in 1 channel)
    if channel is None: 
        ic = slice(None, None, None) 
        if not color: color = [ DEFAULT_FILL for _ in range(pix.shape[-1]) ]   
        else: color = color[:pix.shape[-1]]
    else:
        ic = channel # abs(int(channel)) % pix.shape[-1]
        if color is None: color = DEFAULT_FILL 

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


class Actions:

    @staticmethod
    def new(inputs, outputs, args):
        
        """ creates new images of given size and color """
        # --args=<int:height>,<int:width>,<int:channel>,<int:blue>,<int:green>,<int:red>,<int:alpha> 

        # --args=<int:height>,<int:width>,4            ,<int:blue>,<int:green>,<int:red>,<int:alpha> 
        # --args=<int:height>,<int:width>,3            ,<int:blue>,<int:green>,<int:red>             
        # --args=<int:height>,<int:width>,1            ,<int:intensity>                              
        
        args = [int(s) for s in args]
        for op in outputs:
            img = create(*args[0:3])
            fill(img, 0, 0, ..., ..., color=args[3:], channel=None)
            save(img, op)

    @staticmethod
    def crop(inputs, outputs, args):
        """ crops an image using bounding box (y, x, h, w) """
        # --args=<int:y-coord>,<int:x-coord>,<int:height>,<int:width>
    
        y, x, h, w = [int(s) for s in args]
        for ip,op in zip(inputs,outputs):
            org = load(ip)
            img = create(h, w, org.shape[-1])
            region_copy(org, y, x, h, w, img, 0, 0)
            save(img, op)

    @staticmethod
    def extend(inputs, outputs, args):
        """ extends an image using boundary distance """
        # --args=<int:north>,<int:south>,<int:east>,<int:west>,<int:blue>,<int:green>,<int:red>,<int:alpha>

        args = [int(s) for s in args]
        north, south, east, west = args[0:4]
        for ip,op in zip(inputs,outputs):
            save(extend(load(ip), north, south, east, west, filler=args[4:]), op)

    @staticmethod
    def flip(inputs, outputs, args):
        """ flip an image (horizontally, vertically)"""
        # --args=<bool:horizontally>,<bool:vertically>
        
        h, v = [bool(int(s)) for s in args]
        for ip,op in zip(inputs,outputs):
            org = load(ip)
            img = flip(org, horizontal=h, vertical=v)
            save(img, op)

    @staticmethod
    def rotate(inputs, outputs, args):
        """ rotate an image (clockwise or couter-clockwise)"""
        # --args=<bool:clockwise>

        c = [bool(int(s)) for s in args][0]
        for ip,op in zip(inputs,outputs):
            org = load(ip)
            img = rotate(org, clockwise=c)
            save(img, op)

    @staticmethod
    def convert(inputs, outputs, args):
        """ converts an image (as per output)"""
        # --input=<str:input-file.png> --output=<str:output-file.jpg>
        for ip,op in zip(inputs,outputs): save(load(ip), op)


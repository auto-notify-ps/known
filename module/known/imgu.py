__doc__=r"""
:py:mod:`known/imgu.py`
"""
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
__all__ = [
    'Pix', 'graphfromimage',
]#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#import datetime, os
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import cv2 # pip install opencv-python

from .basic import BaseConvert



CHANNELS = 4
DTYPE = np.uint8
CHANNEL_BRGA = (0, 1, 2, 3)
CHANNEL_RBGA = (1, 2, 0, 3)


class Pix(object):
    r""" abstracts a 4-channel (brga) image """

    def __init__(self, h, w, create=True) -> None:
        self.h, self.w = int(h), int(w)
        if create: self.i = np.zeros((self.h, self.w, CHANNELS), dtype=DTYPE) 
    
    @property
    def RBGA(self): return self.i[:, :, CHANNEL_RBGA]

    def plot_on(self, ax, grid): 
        ax.imshow(self.RBGA)
        if grid:
            xtick, ytick = np.arange(self.w), np.arange(self.h)
            ax.set_xticks(xtick-0.5)
            ax.set_xticklabels(xtick)
            ax.set_yticks(ytick-0.5)
            ax.set_yticklabels(ytick)
            ax.grid(axis='both')

    def clear(self, channel=None): 
        if channel is None:     self.i[:,:,:]       =0 # clear all channels
        else:                   self.i[:,:,channel] =0 # clear specified channel

    def get_color_at(self, row, col, normalize=False):
        b,g,r,a = (self.i[row, col, :]/255 if normalize else self.i[row, col, :])
        return (r, g, b, a)
    
    def set_color_at(self, row:int, col:int, rgba:tuple, normalize=False): 
        if normalize: rgba = [int(x*255) for x in rgba]
        r,g,b,a = rgba
        self.i[row, col, :] = (b, r, g, a) 

    def set_color_in(self, start_row:int, start_col:int, n_rows:int, n_cols:int, rgba:tuple, normalize=False): 
        if normalize: rgba = [int(x*255) for x in rgba]
        r,g,b,a = rgba
        self.i[start_row:start_row+n_rows, start_col:start_col+n_cols, :] = (b, r, g, a) 

    def set_hex_at(self, row:int, col:int, hex:str):
        if hex.startswith('#'): hex = hex[1:]
        hex = hex.upper()
        lenhex = len(hex)
        assert lenhex==6 or lenhex==8, f'expecting 6 or 8 chars but got {lenhex} :: {hex}'
        if lenhex==6: hex = 'FF' + hex # max alpha
        B,G,R,A = tuple(BaseConvert.int2base(num=BaseConvert.to_base_10(BaseConvert.SYM_HEX, hex), base=256, digs=4))
        return self.set_color_at(row,col,(R,G,B,A))

    def set_hex_in(self, start_row:int, start_col:int, n_rows:int, n_cols:int, hex:str):
        if hex.startswith('#'): hex = hex[1:]
        hex = hex.upper()
        lenhex = len(hex)
        assert lenhex==6 or lenhex==8, f'expecting 6 or 8 chars but got {lenhex} :: {hex}'
        if lenhex==6: hex = 'FF' + hex # max alpha
        B,G,R,A = tuple(BaseConvert.int2base(num=BaseConvert.to_base_10(BaseConvert.SYM_HEX, hex), base=256, digs=4))
        return self.set_color_in(start_row,start_col,n_rows,n_cols,(R,G,B,A))



    @staticmethod
    def save(pix, path):  
        return cv2.imwrite(path, pix.i)

    @staticmethod
    def load(path): 
        img =  cv2.imread(path, cv2.IMREAD_UNCHANGED)
        assert img.ndim==3, f'expecting 3-D array but got {img.ndim}-D'
        assert img.shape[-1]== CHANNELS, f'must be argb image with {CHANNELS} channels but got {img.shape[-1]} channels'
        h, w, _ = img.shape
        pix = __class__(h, w, False)
        pix.i = img.astype(DTYPE)
        return pix

    @staticmethod
    def plot(pix, ratio=0.75, grid=True):
        fig, ax = plt.subplots(1,1, figsize=(pix.w*ratio, pix.h*ratio))
        pix.plot_on(ax, grid)
        plt.show()

    def show_color_at(self, row, col, fs=1):
        rgba = self.get_color_at(row,col, normalize=False)
        hexc = self.get_hex_at(row, col)
        color = self.get_color_at(row, col, normalize=True)
        if not isinstance(fs, tuple): fs=(fs,fs)
        plt.figure(figsize=fs)
        plt.yticks([], [])
        plt.title(f'{rgba=}')
        plt.bar([f'{hexc}'], [1], color=color)
        plt.show()
        plt.close()

    @staticmethod
    def region(from_pix, start_row, start_col, n_rows, n_cols):
        r""" creates a new class object from a rectangular region with upper left corner at (x,y) and size (w,h)"""
        pix = __class__(n_rows, n_cols, False)
        pix.i = np.copy(from_pix.i[start_row:start_row+n_rows, start_col:start_col+n_cols,  :])
        return pix

    @staticmethod
    def copy_region(pix_from, start_row, start_col, n_rows, n_cols, pix_to, start_row_to, start_col_to):
        pix_to.i[start_row_to:start_row_to+n_rows, start_col_to:start_col_to+n_cols,  :] = pix_from.i[start_row:start_row+n_rows, start_col:start_col+n_cols,  :]

    @staticmethod
    def copy(pix_from, pix_to): pix_to.i[:, :, :] = pix_from.i[:, :, :]

    def clone(self):
        pix = self.__class__(self.h, self.w, create=False)
        pix.i = np.copy(self.i)
        return pix
        

def graphfromimage(img_path:str, pixel_choice:str='first', dtype=None) -> ndarray:
    r""" 
    Covert an image to an array (1-Dimensional)

    :param img_path:        path of input image 
    :param pixel_choice:    choose from ``[ 'first', 'last', 'mid', 'mean' ]``

    :returns: 1-D numpy array containing the data points

    .. note:: 
        * This is used to generate synthetic data in 1-Dimension. 
            The width of the image is the number of points (x-axis),
            while the height of the image is the range of data points, choosen based on their index along y-axis.
    
        * The provided image is opened in grayscale mode.
            All the *black pixels* are considered as data points.
            If there are multiple black points in a column then ``pixel_choice`` argument specifies which pixel to choose.

        * Requires ``opencv-python``

            Input image should be readable using ``cv2.imread``.
            Use ``pip install opencv-python`` to install ``cv2`` package
    """
    img= cv2.imread(img_path, 0)
    imgmax = img.shape[1]-1
    j = img*0
    j[np.where(img==0)]=1
    pixel_choice = pixel_choice.lower()
    pixel_choice_dict = {
        'first':    (lambda ai: ai[0]),
        'last':     (lambda ai: ai[-1]),
        'mid':      (lambda ai: ai[int(len(ai)/2)]),
        'mean':     (lambda ai: np.mean(ai))
    }
    px = pixel_choice_dict[pixel_choice]
    if dtype is None: dtype=np.float_
    return np.array([ imgmax-px(np.where(j[:,i]==1)[0]) for i in range(j.shape[1]) ], dtype=dtype)



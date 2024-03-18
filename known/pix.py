__doc__=r"""
:py:mod:`known/pix.py`
"""
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
__all__ = ['BaseConvertNumpy', 'Pix'] 
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
from typing import Any, Union, Iterable, Callable #, BinaryIO, cast, Dict, Optional, Type, Tuple, IO
from math import floor, log, ceil
import numpy as np
import matplotlib.pyplot as plt
import cv2 # pip install opencv-python


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class BaseConvertNumpy:
    
    r""" Number System Conversion (using numpy)
    
    A number is abstract concept that has many representations using sets of symbols

    A base-n number system uses a set of n digits to represent any number
    This is called the representation of the number

    Given one representation, we only need to convert to another

    """
    
    @staticmethod
    def convert(digits, base_from, base_to, reversed=True):
        r""" convers from one base to another 
        
        :param digits:      iterable of digits in base ```base_from```. NOTE: digits are Natural Numbers starting at 0. base 'b' will have digits between [0, b-1]
        :param base_from:   int - the base to convert from
        :param base_to:     int - the base to convert to
        :param reversed:    bool - if True, digits are assumed in reverse (human readable left to right)
                            e.g. if reversed is True then binary digits iterable [1,0,0] will represent [4] in decimal otherwise it will represent [1] in decimal
        """

        digits_from =  np.array(digits, dtype=np.uint32) # convert to int data-type
        if reversed: digits_from = digits_from[::-1]
        ndigits_from = len(digits_from)
        mult_from = np.array([base_from**i for i in range(ndigits_from)], dtype=np.uint32)
        repr_from = np.dot(digits_from , mult_from)

        #ndc = base_from**ndigits_from
        ndigits_to = ceil(log(repr_from,base_to))
        digits_to =  np.zeros((ndigits_to,), dtype=np.uint32)
        n = int(repr_from)
        for d in range(ndigits_to):
            digits_to[d] = n%base_to
            n=n//base_to

        if reversed: digits_to = digits_to[::-1]
        return tuple(digits_to)


    @staticmethod
    def ndigits(num:int, base:int): return ceil(log(num,base))

    @staticmethod
    def int2base(num:int, base:int, digs:int) -> list:
        r""" 
        Convert base-10 integer to a base-n list of fixed no. of digits 

        :param num:     base-10 number to be represented
        :param base:    base-n number system
        :param digs:    no of digits in the output

        :returns:       represented number as a list of ordinals in base-n number system

        .. seealso::
            :func:`~known.basic.base2int`
        """
        
        ndigits = digs if digs else ceil(log(num,base)) 
        digits =  np.zeros((ndigits,), dtype=np.uint32)
        n = num
        for d in range(ndigits):
            digits[d] = n%base
            n=n//base
        return digits

    @staticmethod
    def base2int(num:Iterable, base:int) -> int:
        """ 
        Convert an iterbale of digits in base-n system to base-10 integer

        :param num:     iterable of base-n digits
        :param base:    base-n number system

        :returns:       represented number as a integer in base-10 number system

        .. seealso::
            :func:`~known.basic.int2base`
        """
        res = 0
        for i,n in enumerate(num): res+=(base**i)*n
        return int(res)


    SYM_BIN = { f'{i}':i for i in range(2) }
    SYM_OCT = { f'{i}':i for i in range(8) }
    SYM_DEC = { f'{i}':i for i in range(10) }
    SYM_HEX = {**SYM_DEC , **{ s:(i+10) for i,s in enumerate(('A', 'B', 'C', 'D', 'E', 'F'))}}
    
    @staticmethod
    def n_syms(n): return { f'{i}':i for i in range(n) }

    @staticmethod
    def to_base_10(syms:dict, num:str):
        b = len(syms)
        l = [ syms[n] for n in num[::-1] ]
        return __class__.base2int(l, b)

    @staticmethod
    def from_base_10(syms:dict, num:int, joiner='', ndigs=None):
        base = len(syms)
        #print(f'----{num=} {type(num)}, {base=}, {type(base)}')
        if not ndigs: ndigs = (1 + (0 if num==0 else floor(log(num, base))))  # __class__.ndigs(num, base)
        ss = tuple(syms.keys())
        S = [ ss[i]  for i in __class__.int2base(num, base, ndigs) ]
        return joiner.join(S[::-1])


    @staticmethod
    def int2hex(num:int, joiner=''): return __class__.from_base_10(__class__.SYM_HEX, num, joiner)
  
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class Pix(object):
    r""" abstracts a 4-channel (brga) image """

    #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    CHANNELS = 4
    DTYPE = np.uint8
    CHANNEL_BRGA = (0, 1, 2, 3)
    CHANNEL_RBGA = (1, 2, 0, 3)
    #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    def __init__(self, h, w, create=True) -> None:
        self.h, self.w = int(h), int(w)
        if create: self.i = np.zeros((self.h, self.w, __class__.CHANNELS), dtype=__class__.DTYPE) 
    
    @property
    def RBGA(self): return self.i[:, :, __class__.CHANNEL_RBGA]

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
        B,G,R,A = tuple(BaseConvertNumpy.int2base(num=BaseConvertNumpy.to_base_10(BaseConvertNumpy.SYM_HEX, hex), base=256, digs=4))
        return self.set_color_at(row,col,(R,G,B,A))

    def set_hex_in(self, start_row:int, start_col:int, n_rows:int, n_cols:int, hex:str):
        if hex.startswith('#'): hex = hex[1:]
        hex = hex.upper()
        lenhex = len(hex)
        assert lenhex==6 or lenhex==8, f'expecting 6 or 8 chars but got {lenhex} :: {hex}'
        if lenhex==6: hex = 'FF' + hex # max alpha
        B,G,R,A = tuple(BaseConvertNumpy.int2base(num=BaseConvertNumpy.to_base_10(BaseConvertNumpy.SYM_HEX, hex), base=256, digs=4))
        return self.set_color_in(start_row,start_col,n_rows,n_cols,(R,G,B,A))



    @staticmethod
    def save(pix, path):  
        return cv2.imwrite(path, pix.i)

    @staticmethod
    def load(path): 
        img =  cv2.imread(path, cv2.IMREAD_UNCHANGED)
        assert img.ndim==3, f'expecting 3-D array but got {img.ndim}-D'
        assert img.shape[-1]== __class__.CHANNELS, f'must be argb image with {__class__.CHANNELS} channels but got {img.shape[-1]} channels'
        h, w, _ = img.shape
        pix = __class__(h, w, False)
        pix.i = img.astype(__class__.DTYPE)
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
        
    @staticmethod
    def graphfromimage(img_path:str, pixel_choice:str='first', dtype=None):
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

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=